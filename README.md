# DiffComp —— CLIP 特征空间潜扩散增强的组合零样本学习

> 基于 [Troika](https://github.com/bighuang624/Troika)（CVPR 2024, arXiv:2303.15230）
> 三分支架构的扩展实现。本仓库包含**完整可运行代码**：Troika 原有文件原样保留，
> 新增特征扩散网络（FDN）及其训练/推理路径。

## 1. 研究动机

Troika 是 CZSL 的强基线，但其判别式框架存在三个结构性弱点（详见
[`code/DiffComp_proposal.md`](code/DiffComp_proposal.md)）：

1. **未见组合没有视觉监督信号**：三分支 CE loss 只对 seen 组合有梯度，泛化完全
   依赖 CLIP 预训练特征的跨组合迁移；
2. **属性/对象解耦信号弱**：两个浅层 disentangler 仅由分类 CE 间接监督；
3. **CMT 跨模态牵引是单次、单向的**。

核心洞察：CLIP 视觉特征空间本身是紧凑语义流形，已见组合的视觉特征在其中遵循
组合规律（属性带来一致方向的视觉变化、对象核心结构稳定）。若能学习该空间的
**生成模型**，即可为任意未见组合采样"伪视觉特征"，直接补充训练信号。

## 2. 方法概要

DiffComp 在 Troika 三分支结构不变的前提下，新增一个轻量**特征扩散网络
（Feature Diffusion Network, FDN）**：

- **特征空间 DDPM**：前向加噪对象从像素图像换成 CLIP 视觉特征 `v0`（D=768），
  余弦噪声计划；去噪网络为 4 层 Pre-Norm Transformer Decoder，以属性/对象文本
  嵌入为条件，输出层零初始化以保证训练初期不干扰主网络（约 12M 参数）；
- **Classifier-Free Guidance**：训练时以 p=0.1 置空条件，推理用 CFG（γ≈2.0）
  引导采样，DDIM 50 步；
- **三路去噪强化解耦**：路径 C 以原始视觉特征为目标；路径 A / 路径 O 分别以
  `attr_disentangler` / `obj_disentangler` 输出（`.detach()`）为目标，为解耦
  模块提供比分类 CE 强得多的生成式监督；
  `L_diff = L_diff_C + α·(L_diff_A + L_diff_O)`，
  总损失 `L_total = L_troika + λ·L_diff`，FDN 使用独立优化器；
- **开放世界合成特征库**：测试前对每个候选组合 DDIM 采样 K=5 次取均值、归一化，
  推理时 `logit_final = logit_troika + w·(f_img · v̂0(a,o))`，合成库离线构建一次。

## 3. 仓库结构

```
DiffComp/
├── README.md                 # 本文件
├── LICENSE                   # MIT（继承自 Troika，见文末说明）
└── code/                     # 代码主体
    ├── DiffComp_proposal.md  # 完整方案（动机/公式/超参/消融设计）
    ├── README.md             # 详细使用说明（差异清单、超参调参建议、FAQ）
    ├── model/
    │   ├── diffusion.py      # ★新增★ FDN、噪声计划、DDIM 采样、合成库构建
    │   ├── troika.py         # 修改：集成 compute_diffusion_loss /
    │   │                     #   build_synth_feat_bank / forward_with_synth
    │   ├── common.py / model_factory.py   # 不变
    ├── train.py              # 修改：双优化器 + 扩散 loss + OW 合成推理路径
    ├── test.py / dataset.py / parameters.py / utils.py
    ├── parameters.py         # 修改：新增 diff_* 系列超参
    ├── config/troika/        # 三个数据集 × closed/open-world 配置（含扩散超参）
    ├── clip_modules/         # CLIP 模型与 tokenizer（Troika 原样）
    ├── tools/                # 优化器 / mixup（Troika 原样）
    └── requirements.txt      # 与原 Troika 相同，无额外依赖
```

## 4. 复现步骤

### 4.1 环境

```bash
conda create -n diffcomp python=3.8   # 与 Troika 一致即可
pip install -r code/requirements.txt
```

另需 OpenAI CLIP 的 `ViT-L-14.pt` 权重，路径通过 `--clip_arch` 传入。

### 4.2 数据集

与 Troika 完全一致：**MIT-States、UT-Zappos、C-GQA**。下载与整理可参考上游
Troika 仓库的 `download_data` 脚本，整理为 `data/<dataset>/` 目录后通过
`--dataset_path` 传入。

### 4.3 训练与评估

```bash
cd code

# 闭世界训练（三数据集同理，替换 config 与数据路径）
python train.py --yml_path config/troika/mit-states.yml \
    --dataset_path data/mit-states \
    --clip_arch /path/to/ViT-L-14.pt \
    --save_path output/mit

# 开放世界训练
python train.py --yml_path config/troika/mit-states-ow.yml \
    --dataset_path data/mit-states \
    --clip_arch /path/to/ViT-L-14.pt \
    --save_path output/mit_ow --open_world True

# 开放世界推理 + 合成特征库（DiffComp 新增能力）
python train.py --yml_path config/troika/mit-states-ow.yml \
    --dataset_path data/mit-states \
    --save_path output/mit_ow --open_world True \
    --use_synth_feat True --synth_weight 0.3 \
    --load_model output/mit_ow/val_best.pt
```

关键超参（均在 `parameters.py` 与 yml 中，含注释）：

| 超参 | 默认值 | 说明 |
|---|---|---|
| `diff_lambda` | 0.5 | 扩散 loss 权重 λ（建议 0.3~0.5） |
| `diff_alpha` | 0.3 | 属性/对象去噪路径权重 α |
| `diff_lr` | 1e-4 | FDN 独立学习率（双优化器） |
| `diff_guidance_scale` | 2.0 | CFG 引导系数 |
| `diff_sample_T` | 50 | DDIM 采样步数 |
| `diff_synth_k` | 5 | 合成库每组合采样次数 |
| `use_synth_feat` | False | 是否启用 OW 合成特征库 |
| `synth_weight` | 0.3 | OW 推理合成 logit 融合权重 |

只想要最轻量的用法：保持 `use_synth_feat: False`、仅调 `diff_lambda`，此时只增加
训练 loss，推理与原 Troika 完全相同。更多调参建议与 FAQ 见
[`code/README.md`](code/README.md)。

### 4.4 结果

本仓库暂未附实验结果文件。完整方案、消融设计（A1~A6）与预期分析见
[`code/DiffComp_proposal.md`](code/DiffComp_proposal.md)；实测数字以实验记录为准，
此处不作引用。

## 5. 引用

本方法基于 Troika，请引用：

```bibtex
@inproceedings{Huang2024Troika,
  title={Troika: Multi-Path Cross-Modal Traction for Compositional Zero-Shot Learning},
  author={Siteng Huang and Biao Gong and Yutong Feng and Min Zhang and Yiliang Lv and Donglin Wang},
  booktitle={CVPR},
  year={2024}
}
```

## 6. License

本仓库基于 **MIT 协议的 Troika** 发布：随代码保留的原 MIT LICENSE 文本（见
[LICENSE](LICENSE)，版权行 `Copyright (c) 2022 Carman Lu` 保持原样）继续覆盖
Troika 原有文件；本仓库新增的模块（`model/diffusion.py` 及扩散训练/推理、合成
特征库路径）同样以 MIT 协议发布。
