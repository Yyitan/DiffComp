"""
DiffComp: Feature-space Diffusion for Compositional Zero-Shot Learning
扩散模型核心模块 —— 去噪网络、噪声计划、采样工具、合成特征库构建
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
#  时间步正弦嵌入
# ──────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    """将标量时间步 t 映射为 dim 维正弦-余弦位置嵌入。"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] long
        device = t.device
        half = self.dim // 2
        freq = math.log(10000) / (half - 1)
        freq = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -freq)
        emb  = t.float().unsqueeze(1) * freq.unsqueeze(0)   # [B, half]
        emb  = torch.cat([emb.sin(), emb.cos()], dim=-1)     # [B, dim]
        return emb


# ──────────────────────────────────────────────
#  特征扩散网络（FDN）
# ──────────────────────────────────────────────

class FeatureDiffusionNet(nn.Module):
    """
    轻量 Transformer Decoder 去噪网络。
    输入：带噪视觉特征 vt + 时间步 t + 条件 (c_attr, c_obj)
    输出：预测的加噪噪声 ε，与 vt 维度相同。

    架构：
      - 时间步嵌入：正弦位置编码 → 两层 MLP
      - vt + t_emb → input_proj → Query (长度=1)
      - concat(c_attr, c_obj) → cond_proj → Key/Value (长度=1)
      - n_layers 层 Transformer Decoder (Pre-Norm)
      - out_norm + out_proj（零初始化，保证训练初期稳定）
    """

    def __init__(
        self,
        feat_dim: int = 768,
        n_heads: int = 4,
        n_layers: int = 4,
        time_dim: int = 256,
        cfg_dropout: float = 0.1,
        ff_mult: int = 2,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim    = feat_dim
        self.cfg_dropout = cfg_dropout

        # 时间步嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim),
        )

        # 带噪特征投影（query 侧）
        self.input_proj = nn.Linear(feat_dim, feat_dim)

        # 条件投影：[c_attr | c_obj] → feat_dim（key/value 侧）
        self.cond_proj = nn.Linear(feat_dim * 2, feat_dim)

        # Transformer Decoder（Pre-Norm，更稳定）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feat_dim,
            nhead=n_heads,
            dim_feedforward=feat_dim * ff_mult,
            dropout=attn_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # 输出层
        self.out_norm = nn.LayerNorm(feat_dim)
        self.out_proj  = nn.Linear(feat_dim, feat_dim)

        self._init_weights()

    def _init_weights(self):
        """零初始化输出投影，保证训练初期 FDN 对主网络无扰动。"""
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        vt: torch.Tensor,
        t: torch.Tensor,
        c_attr: torch.Tensor,
        c_obj: torch.Tensor,
        drop_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            vt       : [B, D]  带噪视觉特征
            t        : [B]     时间步（整数）
            c_attr   : [B, D]  属性文本嵌入
            c_obj    : [B, D]  对象文本嵌入
            drop_mask: [B]     bool，True 表示该样本走无条件分支（CFG 训练时使用）
        Returns:
            eps_pred : [B, D]  预测噪声
        """
        # 全部转 float32，避免混精度问题
        vt     = vt.float()
        c_attr = c_attr.float()
        c_obj  = c_obj.float()

        # Classifier-Free Guidance：训练时随机 drop 条件
        if self.training and drop_mask is not None:
            zero   = torch.zeros_like(c_attr)
            mask   = drop_mask.unsqueeze(1)          # [B, 1]
            c_attr = torch.where(mask, zero, c_attr)
            c_obj  = torch.where(mask, zero, c_obj)

        # Query：vt + 时间嵌入
        t_emb = self.time_embed(t)                   # [B, D]
        x     = self.input_proj(vt) + t_emb          # [B, D]
        x     = x.unsqueeze(1)                       # [B, 1, D]

        # Key/Value：条件
        cond  = torch.cat([c_attr, c_obj], dim=-1)   # [B, 2D]
        cond  = self.cond_proj(cond).unsqueeze(1)    # [B, 1, D]

        # Transformer Decoder
        out   = self.transformer(x, cond)             # [B, 1, D]
        out   = self.out_norm(out.squeeze(1))         # [B, D]
        return self.out_proj(out)                     # [B, D]


# ──────────────────────────────────────────────
#  噪声计划与采样工具
# ──────────────────────────────────────────────

def cosine_schedule(T: int = 1000) -> torch.Tensor:
    """
    余弦噪声计划（Nichol & Dhariwal, 2021）。
    返回 alpha_bar: [T+1]，从 alpha_bar[0]=1.0 到 alpha_bar[T]≈0。
    """
    t   = torch.arange(T + 1, dtype=torch.float64)
    s   = 0.008
    ab  = torch.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
    ab  = ab / ab[0]
    return ab.float()


def q_sample(
    v0: torch.Tensor,
    t: torch.Tensor,
    alpha_bar: torch.Tensor,
    noise: torch.Tensor = None,
):
    """
    前向加噪：q(vt | v0) = N(sqrt(ᾱt)·v0, (1-ᾱt)·I)
    Args:
        v0        : [B, D]  干净特征
        t         : [B]     时间步（整数，1-based）
        alpha_bar : [T+1]   噪声计划（缓冲区）
    Returns:
        vt   : [B, D]  带噪特征
        noise: [B, D]  实际加入的噪声
    """
    if noise is None:
        noise = torch.randn_like(v0)
    ab_t      = alpha_bar[t].to(v0.device).float()  # [B]
    sqrt_ab   = ab_t.sqrt().view(-1, 1)
    sqrt_1mab = (1.0 - ab_t).sqrt().view(-1, 1)
    vt        = sqrt_ab * v0.float() + sqrt_1mab * noise
    return vt, noise


def diffusion_loss(
    model: FeatureDiffusionNet,
    v0: torch.Tensor,
    c_attr: torch.Tensor,
    c_obj: torch.Tensor,
    alpha_bar: torch.Tensor,
    cfg_dropout: float = 0.1,
) -> torch.Tensor:
    """
    单条去噪路径的 DDPM 训练 loss。
    随机采样时间步 t，加噪，让网络预测噪声，计算 MSE。
    """
    B = v0.shape[0]
    T = len(alpha_bar) - 1

    # 均匀采样时间步 t ∈ [1, T]
    t     = torch.randint(1, T + 1, (B,), device=v0.device)
    vt, eps = q_sample(v0, t, alpha_bar)

    # CFG dropout mask
    drop_mask = torch.rand(B, device=v0.device) < cfg_dropout

    eps_pred = model(vt, t, c_attr, c_obj, drop_mask=drop_mask)
    return F.mse_loss(eps_pred, eps.float())


# ──────────────────────────────────────────────
#  DDIM 推理采样（带 CFG）
# ──────────────────────────────────────────────

@torch.no_grad()
def ddim_sample(
    model: FeatureDiffusionNet,
    c_attr: torch.Tensor,
    c_obj: torch.Tensor,
    alpha_bar: torch.Tensor,
    T_sample: int = 50,
    guidance_scale: float = 2.0,
) -> torch.Tensor:
    """
    DDIM 确定性采样（带 Classifier-Free Guidance）。
    从纯高斯噪声 vT 出发，经 T_sample 步逆向去噪，生成合成视觉特征 v0_hat。

    Args:
        model          : 训练好的 FeatureDiffusionNet
        c_attr         : [B, D]  属性文本嵌入（作为条件）
        c_obj          : [B, D]  对象文本嵌入（作为条件）
        alpha_bar      : [T+1]   噪声计划
        T_sample       : DDIM 采样步数（默认 50，推理够用）
        guidance_scale : CFG 引导强度（建议 1.5~3.0）
    Returns:
        v0_hat : [B, D]  合成视觉特征（未做 L2 归一化）
    """
    model.eval()
    B, D   = c_attr.shape[0], model.feat_dim
    device = c_attr.device

    # 等间隔采样时间步 T → 1
    T_full    = len(alpha_bar) - 1
    timesteps = torch.linspace(T_full, 1, T_sample, dtype=torch.long, device=device)

    # 从纯噪声出发
    vt         = torch.randn(B, D, device=device, dtype=torch.float32)
    zeros_attr = torch.zeros_like(c_attr).float()
    zeros_obj  = torch.zeros_like(c_obj).float()

    for i, t_val in enumerate(timesteps):
        t_batch = t_val.expand(B)

        # 有条件和无条件预测
        eps_cond   = model(vt, t_batch, c_attr.float(), c_obj.float())
        eps_uncond = model(vt, t_batch, zeros_attr, zeros_obj)

        # CFG 线性组合
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # DDIM 更新
        ab_t    = alpha_bar[t_val].to(device).float()
        t_prev  = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
        ab_prev = alpha_bar[t_prev].to(device).float()

        # 预测干净特征 x0
        v0_pred = (vt - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp(min=1e-8)
        v0_pred = v0_pred.clamp(-10.0, 10.0)   # 防止数值爆炸

        # DDIM 方向
        dir_xt = (1.0 - ab_prev).sqrt() * eps
        vt     = ab_prev.sqrt() * v0_pred + dir_xt

    return vt   # v0_hat，[B, D]


# ──────────────────────────────────────────────
#  开放世界合成特征库构建
# ──────────────────────────────────────────────

@torch.no_grad()
def build_synth_feat_bank(
    diff_net: FeatureDiffusionNet,
    alpha_bar: torch.Tensor,
    pairs: torch.Tensor,
    soft_att_obj: torch.Tensor,
    offset: int,
    T_sample: int = 50,
    guidance_scale: float = 2.0,
    K: int = 5,
    batch_size: int = 256,
) -> torch.Tensor:
    """
    离线预计算合成特征库（Synthetic Feature Bank）。
    对候选集中的每个组合 (attr, obj)，采样 K 次取均值，得到一个稳定的合成特征向量。

    Args:
        diff_net       : 训练好的 FeatureDiffusionNet
        alpha_bar      : 噪声计划
        pairs          : [N, 2]  候选组合对 (attr_idx, obj_idx)
        soft_att_obj   : [N_attr+N_obj, D]  软属性/对象嵌入（Troika 参数）
        offset         : 属性数量，用于索引 soft_att_obj
        T_sample       : DDIM 采样步数
        guidance_scale : CFG 引导系数
        K              : 每个组合采样次数（取均值降方差）
        batch_size     : 每批处理的组合数（防止 OOM）
    Returns:
        synth_bank : [N, D]  L2 归一化后的合成特征库
    """
    diff_net.eval()
    N      = len(pairs)
    D      = soft_att_obj.shape[-1]
    device = soft_att_obj.device
    bank   = torch.zeros(N, D, device=device, dtype=torch.float32)

    for start in range(0, N, batch_size):
        end        = min(start + batch_size, N)
        batch_pair = pairs[start:end]              # [B', 2]
        attr_idx   = batch_pair[:, 0]
        obj_idx    = batch_pair[:, 1]
        c_attr     = soft_att_obj[attr_idx].float()
        c_obj      = soft_att_obj[obj_idx + offset].float()

        # 采样 K 次取均值，降低方差
        samples = []
        for _ in range(K):
            v = ddim_sample(
                diff_net, c_attr, c_obj, alpha_bar,
                T_sample=T_sample, guidance_scale=guidance_scale,
            )
            samples.append(v)
        bank[start:end] = torch.stack(samples, dim=0).mean(dim=0)

    # L2 归一化（对齐 Troika 的 logit 计算方式）
    bank = F.normalize(bank, dim=-1)
    return bank
