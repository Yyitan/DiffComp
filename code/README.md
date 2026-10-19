# DiffComp：CLIP 特征空间潜扩散增强的 CZSL

> 基于 Troika（CVPR 2024）的扩展实现

---

## 环境配置

```bash
# 与原 Troika 完全相同，无额外依赖
pip install -r requirements.txt
```

## 数据集准备

与原 Troika 完全相同，参考 `download_data/download_data.sh`。

---

## 快速开始

### 1. 闭合世界训练（Closed-World）

```bash
cd code
CUDA_VISIBLE_DEVICES=1 python train.py --yml_path config/troika/mit-states.yml \
    --dataset_path data/mit-states \
    --clip_arch ./lyt/czsl/CLIP-main/ViT-L-14.pt \
    --save_path ./lyt/czsl/DiffComp/code/output/mit
    
CUDA_VISIBLE_DEVICES=1 python train.py --yml_path config/troika/ut-zappos.yml \
    --dataset_path data/ut-zap50k \
    --clip_arch ./lyt/czsl/CLIP-main/ViT-L-14.pt \
    --save_path ./lyt/czsl/DiffComp/code/output/ut

CUDA_VISIBLE_DEVICES=2 python train.py --yml_path config/troika/cgqa.yml \
    --dataset_path data/cgqa \
    --clip_arch ./lyt/czsl/CLIP-main/ViT-L-14.pt \
    --save_path ./lyt/czsl/DiffComp/code/output/cgqa
```



### 2. 开放世界训练（Open-World）

```bash
python train.py --yml_path config/troika/mit-states-ow.yml \
    --dataset_path /data/mit-states \
    --clip_arch /path/to/ViT-L-14.pt \
    --save_path /output/diffcomp_mit_ow \
    --open_world True
```

### 3. 开放世界推理 + 合成特征库（DiffComp 新增能力）

在配置文件中或命令行启用：

```bash
python train.py --yml_path config/troika/mit-states-ow.yml \
    --dataset_path /data/mit-states \
    --save_path /output/diffcomp_mit_ow \
    --open_world True \
    --use_synth_feat True \
    --synth_weight 0.3 \
    --load_model /output/diffcomp_mit_ow/val_best.pt
```

---

## 与原 Troika 的差异

| 文件 | 状态 | 说明 |
|------|------|------|
| `model/diffusion.py` | **新增** | FDN 全套扩散工具 |
| `model/troika.py` | **修改** | 新增 `compute_diffusion_loss`、`build_synth_feat_bank`、`forward_with_synth` |
| `train.py` | **修改** | 双优化器 + 扩散 loss + OW 合成推理路径 |
| `parameters.py` | **修改** | 新增 `diff_*` 系列超参 |
| `config/troika/mit-states.yml` | **修改** | 新增扩散超参字段（带注释） |
| 其余所有文件 | **不变** | 完整复制自原 Troika |

---

## 关键超参说明

| 超参 | 默认值 | 调参建议 |
|------|--------|---------|
| `diff_lambda` | 0.5 | 0.3~0.5；过大会压制 Troika 主损失 |
| `diff_alpha` | 0.3 | 控制属性/对象解耦路径权重 |
| `diff_lr` | 1e-4 | FDN 独立学习率，通常比主网络 lr 大 |
| `diff_guidance_scale` | 2.0 | 1.5~3.0 均可；越大生成特征越遵循文本条件 |
| `synth_weight` | 0.3 | OW 推理时合成 logit 的融合比例 |
| `use_synth_feat` | False | 设为 True 启用 OW 合成特征库（仅 OW 有效） |

---

## 常见问题

**Q: 训练速度慢了多少？**
A: FDN（12M 参数）的三路去噪 loss 约增加 15~25% 训练时间。

**Q: OW 推理合成库构建多久？**
A: MIT-States 全候选组合约 28k 对，DDIM 50 步，batch=256，约 3~5 分钟（单 A100）。

**Q: 能否不用合成库，只用扩散 loss 做训练增强？**
A: 可以。保持 `use_synth_feat: False`，只修改 `diff_lambda` 即可。
   这是最轻量的使用方式，只增加训练 loss，推理与原 Troika 完全相同。
