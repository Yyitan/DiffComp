# DiffComp：面向组合零样本学习的 CLIP 特征空间潜扩散增强方案

> 基于 Troika（CVPR 2024）三分支架构的扩展方案
> 方向：生成式扩散 × 视觉-语言对齐

---

## 1. 研究动机

### 1.1 Troika 的三个核心局限

Troika 在 CZSL 上取得了当前最佳结果，但其判别式框架存在三个结构性弱点：

**弱点一：未见组合没有视觉监督信号。**
训练集中只有已见组合（seen compositions）的图像。三个分支的 Cross-Entropy loss 对未见组合（unseen compositions）没有直接的梯度来源，泛化完全依赖 CLIP 预训练特征的跨组合迁移能力，上限受 CLIP 的组合泛化边界制约。

**弱点二：属性/对象解耦信号弱。**
Troika 用两个浅层 FC（`attr_disentangler`, `obj_disentangler`）从 CLIP 视觉特征中分离属性和对象的视觉表示，但这两个模块的唯一监督来自分类 CE loss，属于间接、弱监督。没有任何约束强制 `attr_disentangler` 的输出真正对属性特征敏感而对对象特征不敏感，反之亦然。

**弱点三：CMT 跨模态牵引是单次、单向的。**
Cross-Modal Traction（CMT）模块用文本 prompt 作 query，图像 patch 作 key/value，只做一次交叉注意力。这限制了视觉内容对文本提示的反馈能力，特别是对于外观变化显著的未见组合。

### 1.2 核心洞察

CLIP 的视觉特征空间（ViT-L/14 的输出维度 D=768）本身是一个紧凑的语义流形。已见组合的视觉特征在这个空间里遵循一定的组合规律：属性 "wet" 对不同对象（dog, stone, paper）的视觉影响有一致的方向性（反光增强、纹理变化）；对象 "dog" 在不同属性下的视觉核心结构是稳定的（轮廓、毛发纹理）。

如果能在 CLIP 特征空间里学习这种**组合规律的生成模型**，就可以对任意未见的属性-对象配对，采样生成一个近似的组合视觉特征，作为该未见组合的伪视觉正样本，直接补充训练信号。

---

## 2. 方法：DiffComp

### 2.1 总体框架

DiffComp 在 Troika 的基础上增加一个**轻量特征扩散网络**（Feature Diffusion Network, FDN），以属性和对象的文本嵌入为条件，学习在 CLIP 视觉特征空间中去噪。整体框架保持 Troika 的三分支结构不变，FDN 作为独立模块接入，提供三路辅助训练信号，并在开放世界推理时提供合成特征库。

```
                        ┌────────────────────────────────────┐
                        │          DiffComp 新增部分          │
  训练图像  ──CLIP ViT──► v₀ (D维视觉特征)                    │
                        │         ↓ 加噪 q(vₜ|v₀)            │
  attr文本  ──文本编码──► c_attr ──►  ε_θ(vₜ, t, c_attr, c_obj) │
  obj文本   ──文本编码──► c_obj  ──►  → 预测噪声 ε              │
                        │         ↓ L_diff                   │
                        └────────────────────────────────────┘
                                      ↓ (合并到总loss)
                        ┌────────────────────────────────────┐
                        │          Troika 原有部分（不变）      │
                        │  三分支对齐 + CMT + Adapter          │
                        └────────────────────────────────────┘
```

### 2.2 特征扩散网络（FDN）

#### 2.2.1 前向加噪过程

标准 DDPM 前向过程，对象从像素图像换为 CLIP 视觉特征向量 v₀ ∈ ℝᴰ：

```
q(vₜ | v₀) = N(vₜ ; √ᾱₜ · v₀, (1 - ᾱₜ) · I)
```

使用余弦噪声计划（Nichol & Dhariwal 2021）：

```
ᾱₜ = cos²(((t/T + s) / (1 + s)) · π/2),   s = 0.008
```

#### 2.2.2 去噪网络结构

去噪网络 ε_θ 接受带噪特征 vₜ、时间步嵌入、以及属性/对象条件，输出预测噪声：

```
ε_θ : (vₜ ∈ ℝᴰ, t ∈ ℕ, c_attr ∈ ℝᴰ, c_obj ∈ ℝᴰ) → ε_pred ∈ ℝᴰ
```

网络结构为 4 层 Transformer Decoder（Pre-Norm）：
- vₜ + 时间嵌入 → 作为 Query（序列长度 = 1）
- concat(c_attr, c_obj) → 线性投影 → 作为 Key/Value
- 输出层零初始化，保证训练初始阶段不干扰 Troika 主网络

参数量约 12M，与 Troika 的 Adapter 模块（`additional_visual_params`）同量级。

#### 2.2.3 Classifier-Free Guidance（CFG）

训练时以概率 p=0.1 将条件置为零向量（无条件分支），使模型同时学习无条件分布。推理时使用 CFG 引导采样：

```
ε_guided = ε_uncond + γ · (ε_cond - ε_uncond)
```

其中 γ 为引导系数（建议 2.0），控制生成特征对文本条件的遵从程度。

### 2.3 三路去噪，强化解耦

设计三条并行的去噪路径，每路针对 Troika 三个分支各自的视觉特征作为目标：

| 路径 | 条件 | 去噪目标 | 对应 Troika 分支 |
|------|------|----------|-----------------|
| 路径 C（组合） | c_attr + c_obj | v₀（原始 CLIP 特征） | comp branch |
| 路径 A（属性） | c_attr + 零向量 | attr_disentangler(v₀) | attr branch |
| 路径 O（对象） | 零向量 + c_obj | obj_disentangler(v₀) | obj branch |

路径 A 和 O 的目标使用 `.detach()` 截断梯度，防止扩散 loss 反向传播进入 disentangler 模块引起训练不稳定。

**为什么这能强化解耦？** 路径 A 强制网络从 c_attr 条件（无 obj 信息）重建出属性视觉特征，是比分类 CE loss 强得多的生成式目标。反过来，Troika 的 `attr_disentangler` 输出作为去噪目标，其质量直接影响路径 A 的 loss，从而形成联合优化压力。

### 2.4 联合训练目标

#### 扩散 loss

```
L_diff_C = E_{t, ε} [‖ε_θ(vₜ, t, c_attr, c_obj) − ε‖²]       (路径 C)
L_diff_A = E_{t, ε} [‖ε_θ(vₜ_a, t, c_attr, 0) − ε‖²]          (路径 A)
L_diff_O = E_{t, ε} [‖ε_θ(vₜ_o, t, 0, c_obj) − ε‖²]           (路径 O)

L_diff = L_diff_C + α · (L_diff_A + L_diff_O)
```

其中 α=0.3 控制属性/对象解耦路径的权重。

#### 总 loss

```
L_total = L_troika + λ · L_diff
```

- `L_troika`：Troika 原有的三分支 Cross-Entropy 加权和（保持不变）
- `λ`：扩散 loss 权重，建议 0.3~0.5，从 0.3 开始调

#### 超参汇总

| 超参 | 默认值 | 说明 |
|------|--------|------|
| `diff_T` | 1000 | 训练时扩散总步数 |
| `diff_sample_T` | 50 | DDIM 采样步数（推理时） |
| `diff_lambda` | 0.5 | L_diff 在总 loss 中的权重 |
| `diff_alpha` | 0.3 | 属性/对象路径权重 |
| `diff_guidance_scale` | 2.0 | CFG 引导系数 |
| `diff_cfg_dropout` | 0.1 | CFG 训练 dropout 概率 |
| `diff_n_layers` | 4 | FDN Transformer 层数 |
| `diff_n_heads` | 4 | FDN 多头注意力头数 |
| `diff_lr` | 1e-4 | FDN 独立学习率 |
| `diff_synth_k` | 5 | 合成特征库每对组合采样次数 |
| `synth_weight` | 0.3 | 开放世界推理时合成 logit 权重 |

### 2.5 开放世界推理：合成特征库

开放世界 CZSL（OW-CZSL）的候选组合空间巨大（|A|×|O| 级别）。DiffComp 在测试前离线构建合成特征库（Synthetic Feature Bank）：

```
对每个候选组合 (aᵢ, oⱼ) ∈ C_all：
    v̂₀ᵢⱼ = mean_{k=1}^{K}[DDIM_sample(ε_θ, c_aᵢ, c_oⱼ, γ)]
    normalize(v̂₀ᵢⱼ)
```

采样 K=5 次取均值，用 DDIM 50 步（远快于 DDPM 1000 步）。推理时：

```
logit_final(xᵢ, (a,o)) = logit_troika(xᵢ, (a,o)) + w · (fᵢ · v̂₀(a,o))
```

其中 fᵢ 是查询图像的 L2 归一化 CLIP 视觉特征，w 为合成 logit 权重（建议 0.3）。

---

## 3. 与 Troika 代码的集成点

| 修改位置 | 修改内容 |
|----------|---------|
| `model/diffusion.py`（新增） | FDN、余弦噪声计划、q_sample、DDIM 采样、合成库构建 |
| `model/troika.py` | `__init__` 加 FDN；新增 `compute_diffusion_loss`、`build_synth_feat_bank`、`forward_with_synth` |
| `train.py` | 训练循环加 `compute_diffusion_loss`；FDN 使用独立优化器 |
| `parameters.py` | 新增扩散相关超参 |
| `config/troika/*.yml` | 新增扩散超参字段 |

---

## 4. 实验方案

### 4.1 数据集

与 Troika 保持一致：MIT-States、UT-Zappos50K、C-GQA，同时评估 Closed-World 和 Open-World 设置。

### 4.2 评估指标

- Closed-World：HM（Harmonic Mean of Seen/Unseen AUC）、AUC
- Open-World：HM、AUC、Best Seen、Best Unseen

### 4.3 消融实验设计

| 实验编号 | 配置 | 目的 |
|----------|------|------|
| A1 | Troika baseline | 基线 |
| A2 | + 路径 C（组合去噪） | 验证组合去噪的贡献 |
| A3 | + 路径 C + A + O（三路去噪） | 验证解耦强化的贡献 |
| A4 | A3 + 合成特征库（OW） | 验证 OW 合成特征的贡献 |
| A5 | A3 + CFG | 验证 CFG 的贡献 |
| A6 | 完整 DiffComp | 最终方案 |

### 4.4 可视化实验

- 用 UMAP 可视化 CLIP 特征空间中真实视觉特征 v₀ 和合成特征 v̂₀ 的分布，验证合成特征的质量
- 对未见组合，比较 Troika 的 logit 分布 vs DiffComp 的 logit 分布

---

## 5. 预期性能提升

基于方法分析，预计：
- Closed-World HM：+1.5~3.0%（相对 Troika）
- Open-World AUC：+1.0~2.5%（主要来自合成特征库）
- 属性分类准确率（attr_acc）：+2~4%（来自三路去噪解耦强化）

训练时间增加约 15~25%（FDN 计算开销），推理增加合成库构建（仅一次，离线完成）。

---

## 6. 文件结构

```
code/
├── model/
│   ├── diffusion.py          # 新增：FDN 及全套扩散工具
│   ├── troika.py             # 修改：集成 FDN
│   ├── common.py             # 不变
│   └── model_factory.py      # 不变
├── train.py                  # 修改：双优化器 + 扩散 loss
├── parameters.py             # 修改：新增超参
└── config/troika/
    └── mit-states.yml        # 修改：新增超参字段
```
