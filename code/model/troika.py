"""
DiffComp: Troika + Feature-space Diffusion
在 Troika 原有三分支架构基础上集成特征扩散网络（FDN）。

主要新增：
  - FeatureDiffusionNet 作为 self.diff_net
  - compute_diffusion_loss()  三路去噪联合 loss
  - build_synth_feat_bank()   开放世界合成特征库（离线）
  - forward_with_synth()      使用合成库的开放世界推理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import reduce
from operator import mul
from copy import deepcopy
from torch.nn.modules.utils import _pair
from torch.nn.modules.loss import CrossEntropyLoss

from clip_modules.clip_model import load_clip, QuickGELU
from clip_modules.tokenization_clip import SimpleTokenizer
from model.common import *
from model.diffusion import (
    FeatureDiffusionNet,
    cosine_schedule,
    diffusion_loss,
    ddim_sample,
    build_synth_feat_bank,
)


# ──────────────────────────────────────────────
#  以下三个类与原 Troika 完全相同，保留不变
# ──────────────────────────────────────────────

class Adapter(nn.Module):
    def __init__(self, d_model=None, bottleneck=None, dropout=0.0,
                 init_option="lora", adapter_scalar="0.1",
                 adapter_layernorm_option="none"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option in ("in", "out"):
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)
        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = dropout
        self.init_option = init_option
        self._reset_parameters()

    def _reset_parameters(self):
        if self.init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * self.scale
        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)
        if add_residual:
            output = up + residual
        else:
            output = up
        return output


class Disentangler(nn.Module):
    def __init__(self, emb_dim):
        super(Disentangler, self).__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.bn1_fc = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x


class MulitHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.cross_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, q, kv):
        q = q + self.cross_attn(q, kv, kv)
        q = q + self.dropout(self.mlp(self.norm(q)))
        return q


# ──────────────────────────────────────────────
#  DiffComp 主模型
# ──────────────────────────────────────────────

class Troika(nn.Module):
    """
    Troika + Feature-space Diffusion (DiffComp)。
    在原 Troika 基础上新增：
      1. self.diff_net   : FeatureDiffusionNet
      2. self.alpha_bar  : 余弦噪声计划（缓冲区，不参与梯度）
      3. compute_diffusion_loss()
      4. build_synth_feat_bank() / forward_with_synth()
    """

    def __init__(self, config, attributes, classes, offset):
        super().__init__()
        self.clip = load_clip(name=config.clip_arch, context_length=config.context_length)
        self.tokenizer = SimpleTokenizer()
        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.attr_dropout = nn.Dropout(config.attr_dropout)
        self.cross_attn_dropout = getattr(config, 'cross_attn_dropout', 0.1)
        self.prim_loss_weight   = getattr(config, 'prim_loss_weight', 1)

        self.token_ids, self.soft_att_obj, comp_ctx, attr_ctx, obj_ctx = \
            self.construct_soft_prompt()
        self.offset = offset
        self.enable_pos_emb = True

        dtype = self.clip.dtype
        self.dtype = dtype if dtype is not None else torch.float16

        self.text_encoder = CustomTextEncoder(self.clip, self.tokenizer, self.dtype)

        # 冻结 CLIP 参数
        for p in self.parameters():
            p.requires_grad = False

        assert 'ViT' in config.clip_model, "Troika only supports ViT-based CLIP."

        self.additional_visual_params = self.add_visual_tunable_params()

        output_dim = self.clip.visual.output_dim

        self.soft_att_obj    = nn.Parameter(self.soft_att_obj)
        self.comp_ctx_vectors = nn.Parameter(comp_ctx).cuda()
        self.attr_ctx_vectors = nn.Parameter(attr_ctx).cuda()
        self.obj_ctx_vectors  = nn.Parameter(obj_ctx).cuda()

        self.attr_disentangler = Disentangler(output_dim)
        self.obj_disentangler  = Disentangler(output_dim)

        self.cmt = nn.ModuleList([
            CrossAttentionLayer(output_dim, output_dim // 64, self.cross_attn_dropout)
            for _ in range(config.cmt_layers)
        ])
        self.lamda      = nn.Parameter(torch.ones(output_dim) * config.init_lamda)
        self.patch_norm = nn.LayerNorm(output_dim)

        # ──────────────────────────────────────
        #  DiffComp 新增：特征扩散网络
        # ──────────────────────────────────────
        diff_n_layers = getattr(config, 'diff_n_layers', 4)
        diff_n_heads  = getattr(config, 'diff_n_heads', 4)
        diff_cfg_drop = getattr(config, 'diff_cfg_dropout', 0.1)
        diff_T        = getattr(config, 'diff_T', 1000)

        self.diff_net = FeatureDiffusionNet(
            feat_dim    = output_dim,
            n_heads     = diff_n_heads,
            n_layers    = diff_n_layers,
            time_dim    = 256,
            cfg_dropout = diff_cfg_drop,
        )

        # 噪声计划注册为 buffer（不参与梯度，随 model.cuda() 自动移动）
        self.register_buffer('alpha_bar', cosine_schedule(T=diff_T))

        # 扩散相关超参
        self.diff_lambda        = getattr(config, 'diff_lambda', 0.5)
        self.diff_alpha         = getattr(config, 'diff_alpha', 0.3)
        self.diff_cfg_dropout   = diff_cfg_drop
        self.diff_sample_T      = getattr(config, 'diff_sample_T', 50)
        self.diff_guidance_scale = getattr(config, 'diff_guidance_scale', 2.0)
        self.diff_synth_k       = getattr(config, 'diff_synth_k', 5)
        self.synth_weight       = getattr(config, 'synth_weight', 0.3)

    # ──────────────────────────────────────────
    #  以下方法与原 Troika 完全相同
    # ──────────────────────────────────────────

    def add_visual_tunable_params(self):
        adapter_num = 2 * self.clip.visual.transformer.layers
        params = nn.ModuleList([
            Adapter(
                d_model    = self.clip.visual.transformer.width,
                bottleneck = self.config.adapter_dim,
                dropout    = self.config.adapter_dropout,
            )
            for _ in range(adapter_num)
        ])
        return params

    def encode_image(self, x: torch.Tensor):
        return self.encode_image_with_adapter(x)

    def encode_image_with_adapter(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([
            self.clip.visual.class_embedding.to(x.dtype)
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)
        x = x.permute(1, 0, 2)

        for i_block in range(self.clip.visual.transformer.layers):
            adapt_x  = self.additional_visual_params[i_block](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].attention(
                self.clip.visual.transformer.resblocks[i_block].ln_1(x)
            )
            x = x + adapt_x + residual

            i_adapter = i_block + self.clip.visual.transformer.layers
            adapt_x  = self.additional_visual_params[i_adapter](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].mlp(
                self.clip.visual.transformer.resblocks[i_block].ln_2(x)
            )
            x = x + adapt_x + residual

        img_feature = x.permute(1, 0, 2)
        img_feature = self.clip.visual.ln_post(img_feature)
        if self.clip.visual.proj is not None:
            img_feature = img_feature @ self.clip.visual.proj
        return img_feature[:, 0, :], img_feature

    def encode_text(self, token_ids, token_tensors=None, enable_pos_emb=False):
        return self.text_encoder(token_ids, token_tensors, enable_pos_emb)

    def construct_soft_prompt(self):
        token_ids = self.tokenizer(
            self.config.prompt_template,
            context_length=self.config.context_length
        ).cuda()

        tokenized = torch.cat([
            self.tokenizer(tok, context_length=self.config.context_length)
            for tok in self.attributes + self.classes
        ])
        orig_token_embedding = self.clip.token_embedding(tokenized.cuda())
        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1))
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = self.config.ctx_init
        assert isinstance(ctx_init, list)
        n_ctx  = [len(ctx.split()) for ctx in ctx_init]
        prompt = self.tokenizer(ctx_init, context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)

        comp_ctx = embedding[0, 1: 1 + n_ctx[0], :].to(self.clip.dtype)
        attr_ctx = embedding[1, 1: 1 + n_ctx[1], :].to(self.clip.dtype)
        obj_ctx  = embedding[2, 1: 1 + n_ctx[2], :].to(self.clip.dtype)
        return token_ids, soft_att_obj, comp_ctx, attr_ctx, obj_ctx

    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        token_tensor, num_elements = list(), [len(pair_idx), self.offset, len(self.classes)]
        for i_element in range(self.token_ids.shape[0]):
            class_token_ids = self.token_ids[i_element].repeat(num_elements[i_element], 1)
            token_tensor.append(
                self.clip.token_embedding(class_token_ids.cuda()).type(self.clip.dtype)
            )

        eos_idx     = [int(self.token_ids[i].argmax()) for i in range(self.token_ids.shape[0])]
        soft_att_obj = self.attr_dropout(self.soft_att_obj)

        token_tensor[0][:, eos_idx[0] - 2, :] = soft_att_obj[attr_idx].type(self.clip.dtype)
        token_tensor[0][:, eos_idx[0] - 1, :] = soft_att_obj[obj_idx + self.offset].type(self.clip.dtype)
        token_tensor[0][:, 1: len(self.comp_ctx_vectors) + 1, :] = \
            self.comp_ctx_vectors.type(self.clip.dtype)

        token_tensor[1][:, eos_idx[1] - 1, :] = soft_att_obj[:self.offset].type(self.clip.dtype)
        token_tensor[1][:, 1: len(self.attr_ctx_vectors) + 1, :] = \
            self.attr_ctx_vectors.type(self.clip.dtype)

        token_tensor[2][:, eos_idx[2] - 1, :] = soft_att_obj[self.offset:].type(self.clip.dtype)
        token_tensor[2][:, 1: len(self.obj_ctx_vectors) + 1, :] = \
            self.obj_ctx_vectors.type(self.clip.dtype)

        return token_tensor

    def loss_calu(self, predict, target):
        loss_fn = CrossEntropyLoss()
        _, batch_attr, batch_obj, batch_target = target
        comp_logits, attr_logits, obj_logits   = predict
        batch_attr   = batch_attr.cuda()
        batch_obj    = batch_obj.cuda()
        batch_target = batch_target.cuda()
        loss_comp = loss_fn(comp_logits, batch_target)
        loss_attr = loss_fn(attr_logits, batch_attr)
        loss_obj  = loss_fn(obj_logits,  batch_obj)
        loss = (loss_comp * self.config.pair_loss_weight
              + loss_attr * self.config.attr_loss_weight
              + loss_obj  * self.config.obj_loss_weight)
        return loss

    def logit_infer(self, predict, pairs):
        comp_logits, attr_logits, obj_logits = predict
        attr_pred = F.softmax(attr_logits, dim=-1)
        obj_pred  = F.softmax(obj_logits,  dim=-1)
        for i_comp in range(comp_logits.shape[-1]):
            w_attr = (1 if self.config.attr_inference_weight == 0
                      else attr_pred[:, pairs[i_comp][0]] * self.config.attr_inference_weight)
            w_obj  = (1 if self.config.obj_inference_weight == 0
                      else obj_pred[:,  pairs[i_comp][1]] * self.config.obj_inference_weight)
            comp_logits[:, i_comp] = (comp_logits[:, i_comp]
                                      * self.config.pair_inference_weight
                                      + w_attr * w_obj)
        return comp_logits

    def encode_text_for_open(self, idx):
        token_tensors = self.construct_token_tensors(idx)
        text_features = []
        for i_element in range(self.token_ids.shape[0]):
            _tf, _ = self.encode_text(
                self.token_ids[i_element],
                token_tensors[i_element],
                enable_pos_emb=self.enable_pos_emb,
            )
            text_features.append(_tf / _tf.norm(dim=-1, keepdim=True))
        return text_features

    def forward_for_open(self, batch, text_feats):
        batch_img = batch[0].cuda()
        b = batch_img.shape[0]
        batch_img, batch_patch = self.encode_image(batch_img.type(self.clip.dtype))
        batch_img_features     = [
            batch_img,
            self.attr_disentangler(batch_img),
            self.obj_disentangler(batch_img),
        ]
        norm_img = [f / f.norm(dim=-1, keepdim=True) for f in batch_img_features]

        logits = []
        for i_element in range(self.token_ids.shape[0]):
            idx_tf = text_feats[i_element]
            cmt_tf = idx_tf.unsqueeze(0).expand(b, -1, -1)
            batch_patch_n = self.patch_norm(batch_patch)
            for layer in self.cmt:
                cmt_tf = layer(cmt_tf, batch_patch_n)
            cmt_tf = idx_tf + self.lamda * cmt_tf.squeeze(1)
            cmt_tf = cmt_tf / cmt_tf.norm(dim=-1, keepdim=True)
            logits.append(torch.einsum(
                "bd, bkd->bk",
                norm_img[i_element],
                cmt_tf * self.clip.logit_scale.exp(),
            ))
        return logits

    def forward(self, batch, idx):
        """
        标准前向传播（与原 Troika 完全相同）。
        返回 [comp_logits, attr_logits, obj_logits]，供 loss_calu 使用。
        扩散 loss 通过 compute_diffusion_loss() 单独计算。
        """
        batch_img = batch[0].cuda()
        b = batch_img.shape[0]
        batch_img, batch_patch = self.encode_image(batch_img.type(self.clip.dtype))
        batch_img_features     = [
            batch_img,
            self.attr_disentangler(batch_img),
            self.obj_disentangler(batch_img),
        ]
        norm_img = [f / f.norm(dim=-1, keepdim=True) for f in batch_img_features]

        token_tensors = self.construct_token_tensors(idx)
        logits = []
        for i_element in range(self.token_ids.shape[0]):
            _tf, _ = self.encode_text(
                self.token_ids[i_element],
                token_tensors[i_element],
                enable_pos_emb=self.enable_pos_emb,
            )
            idx_tf = _tf / _tf.norm(dim=-1, keepdim=True)
            cmt_tf = idx_tf.unsqueeze(0).expand(b, -1, -1)
            batch_patch_n = self.patch_norm(batch_patch)
            for layer in self.cmt:
                cmt_tf = layer(cmt_tf, batch_patch_n)
            cmt_tf = idx_tf + self.lamda * cmt_tf.squeeze(1)
            cmt_tf = cmt_tf / cmt_tf.norm(dim=-1, keepdim=True)
            logits.append(torch.einsum(
                "bd, bkd->bk",
                norm_img[i_element],
                cmt_tf * self.clip.logit_scale.exp(),
            ))
        return logits

    # ──────────────────────────────────────────
    #  DiffComp 新增方法
    # ──────────────────────────────────────────

    def compute_diffusion_loss(self, batch, cached_img_feat=None) -> torch.Tensor:
        """
        三路去噪联合扩散 loss。

        路径 C（组合）：cond=(c_attr, c_obj)  target=v0（完整 CLIP 视觉特征）
        路径 A（属性）：cond=(c_attr, 0)       target=attr_disentangler(v0)  stopgrad
        路径 O（对象）：cond=(0, c_obj)         target=obj_disentangler(v0)   stopgrad

        L_diff = L_C + alpha * (L_A + L_O)

        Args:
            batch : 训练 DataLoader 返回的 batch
                    batch[0]: 图像  [B, 3, H, W]
                    batch[1]: attr 标签  [B]
                    batch[2]: obj  标签  [B]
        Returns:
            L_diff : 标量 tensor，可直接 .backward()
        """
        batch_img  = batch[0].cuda()
        batch_attr = batch[1].cuda()   # attr 类别索引，[B]
        batch_obj  = batch[2].cuda()   # obj  类别索引，[B]

        # 提取干净视觉特征（float32，截断梯度不传回 CLIP ViT）
        if cached_img_feat is not None:
            v0_clip = cached_img_feat.detach()
        else:
            batch_img = batch[0].cuda()
            with torch.no_grad():
                v0_clip, _ = self.encode_image(batch_img.type(self.clip.dtype))
            v0_clip = v0_clip.detach()
        v0 = v0_clip.float()

        # 属性/对象的软嵌入作为条件
        # soft_att_obj: [N_attr+N_obj, D]，attr 部分为 [0:offset]，obj 为 [offset:]
        c_attr = self.soft_att_obj[batch_attr].float()           # [B, D]
        c_obj  = self.soft_att_obj[batch_obj + self.offset].float()  # [B, D]

        # ── 路径 C：组合去噪（完整 v0 作目标）──
        L_C = diffusion_loss(
            self.diff_net, v0, c_attr, c_obj,
            self.alpha_bar, cfg_dropout=self.diff_cfg_dropout,
        )

        # ── 路径 A：属性去噪（attr_disentangler 输出作目标，stopgrad）──
        # disentangler 输出 float32 是因为 Disentangler 内有 BN
        with torch.no_grad():
            v0_attr = self.attr_disentangler(v0_clip.float()).detach()
        zeros_obj = torch.zeros_like(c_obj)
        L_A = diffusion_loss(
            self.diff_net, v0_attr, c_attr, zeros_obj,
            self.alpha_bar, cfg_dropout=self.diff_cfg_dropout,
        )

        # ── 路径 O：对象去噪（obj_disentangler 输出作目标，stopgrad）──
        with torch.no_grad():
            v0_obj = self.obj_disentangler(v0_clip.float()).detach()
        zeros_attr = torch.zeros_like(c_attr)
        L_O = diffusion_loss(
            self.diff_net, v0_obj, zeros_attr, c_obj,
            self.alpha_bar, cfg_dropout=self.diff_cfg_dropout,
        )

        return L_C + self.diff_alpha * (L_A + L_O)

    @torch.no_grad()
    def build_synth_feat_bank(
        self,
        pairs: torch.Tensor,
        batch_size: int = 256,
    ) -> torch.Tensor:
        """
        为开放世界推理离线构建合成特征库。
        调用 model.eval() 后使用，在测试开始前执行一次。

        Args:
            pairs      : [N, 2]  候选组合 (attr_idx, obj_idx)
            batch_size : 每批组合数
        Returns:
            synth_bank : [N, D]  L2 归一化合成视觉特征
        """
        return build_synth_feat_bank(
            diff_net        = self.diff_net,
            alpha_bar       = self.alpha_bar,
            pairs           = pairs,
            soft_att_obj    = self.soft_att_obj,
            offset          = self.offset,
            T_sample        = self.diff_sample_T,
            guidance_scale  = self.diff_guidance_scale,
            K               = self.diff_synth_k,
            batch_size      = batch_size,
        )

    def forward_with_synth(
        self,
        batch,
        text_feats,
        synth_bank: torch.Tensor,
    ):
        """
        开放世界推理：融合 Troika 文本 logit 与合成特征 logit。
        图像只编码一次，复用于 Troika 三分支 logit 和合成特征 logit。

        Args:
            batch      : 测试 batch，batch[0] 为图像
            text_feats : encode_text_for_open() 的输出，长度为 3 的 list
                         每个元素 [N_candidates, D]，分别对应 comp/attr/obj 分支
            synth_bank : build_synth_feat_bank() 的输出，[N_candidates, D]，已 L2 归一化
        Returns:
            logits : [comp_logit_fused, attr_logit, obj_logit]
                     comp_logit_fused 已融合合成特征，[B, N_candidates]
        """
        batch_img_raw = batch[0].cuda()
        b = batch_img_raw.shape[0]

        # ── 图像编码：只做一次 ──────────────────────────────────
        batch_img, batch_patch = self.encode_image(batch_img_raw.type(self.clip.dtype))

        # 三分支各自的视觉特征（comp/attr/obj）
        batch_img_features = [
            batch_img,
            self.attr_disentangler(batch_img),
            self.obj_disentangler(batch_img),
        ]
        norm_img = [f / f.norm(dim=-1, keepdim=True) for f in batch_img_features]

        # ── Troika 三分支 logit（与 forward_for_open 逻辑相同）──
        batch_patch_n = self.patch_norm(batch_patch)
        logits = []
        for i_element in range(self.token_ids.shape[0]):
            idx_tf = text_feats[i_element]                         # [N_candidates, D]
            cmt_tf = idx_tf.unsqueeze(0).expand(b, -1, -1)        # [B, N, D]
            for layer in self.cmt:
                cmt_tf = layer(cmt_tf, batch_patch_n)
            cmt_tf = idx_tf + self.lamda * cmt_tf.squeeze(1)      # NOTE: squeeze 适配 N=1 情况
            # 当 N_candidates > 1 时 cmt_tf shape 为 [B, N, D]，需要按 N 展开再归一化
            # 与 forward_for_open 保持完全一致的处理方式
            cmt_tf = cmt_tf / cmt_tf.norm(dim=-1, keepdim=True)
            logits.append(torch.einsum(
                "bd, bkd->bk",
                norm_img[i_element],
                cmt_tf * self.clip.logit_scale.exp(),
            ))

        # ── 合成特征 logit ──────────────────────────────────────
        # 使用 comp branch（i=0）的归一化图像特征与合成库做内积
        img_norm_f32  = norm_img[0].float()                         # [B, D]
        synth_logit   = img_norm_f32 @ synth_bank.float().T \
                        * self.clip.logit_scale.exp()               # [B, N_candidates]

        # ── 加权融合 comp logit ────────────────────────────────
        logits[0] = logits[0] + self.synth_weight * synth_logit

        return logits   # [comp_fused, attr, obj]
