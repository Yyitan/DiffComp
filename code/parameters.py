"""
DiffComp 参数定义。
在原 Troika parameters.py 基础上新增扩散相关超参（以 diff_ 前缀标识）。
"""

import argparse

parser = argparse.ArgumentParser()

# ──────────────────────────────────────────────
#  原 Troika 参数（保持不变）
# ──────────────────────────────────────────────

# 基础训练配置
parser.add_argument("--model_name",           type=str,   help="model name")
parser.add_argument("--lr",                   type=float, default=5e-05,  help="learning rate")
parser.add_argument("--dataset",              type=str,   default='mit-states')
parser.add_argument("--weight_decay",         type=float, default=1e-05)
parser.add_argument("--clip_model",           type=str,   default="ViT-L/14")
parser.add_argument("--epochs",               type=int,   default=20)
parser.add_argument("--epoch_start",          type=int,   default=0)
parser.add_argument("--train_batch_size",     type=int,   default=48)
parser.add_argument("--eval_batch_size",      type=int,   default=16)
parser.add_argument("--num_workers",          type=int,   default=4)
parser.add_argument("--context_length",       type=int,   default=8)
parser.add_argument("--attr_dropout",         type=float, default=0.3)
parser.add_argument("--yml_path",             type=str,   help="yml config path")
parser.add_argument("--clip_arch",            type=str,   help="clip checkpoint path")
parser.add_argument("--dataset_path",         type=str)
parser.add_argument("--save_path",            type=str)
parser.add_argument("--save_every_n",         type=int,   default=5)
parser.add_argument("--save_final_model",     action="store_true")
parser.add_argument("--load_model",           type=str,   default=None)
parser.add_argument("--seed",                 type=int,   default=0)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--same_prim_sample",     action="store_true")

# 评估配置
parser.add_argument("--open_world",           type=bool,  default=False)
parser.add_argument("--bias",                 type=float, default=1e3)
parser.add_argument("--topk",                 type=int,   default=1)
parser.add_argument("--text_encoder_batch_size", type=int, default=16)
parser.add_argument("--threshold",            type=float, default=None)
parser.add_argument("--threshold_trials",     type=int,   default=50)

# Troika 架构参数
parser.add_argument("--adapter_dim",          type=int,   default=64,  help="Adapter bottleneck dim")
parser.add_argument("--init_lamda",           type=float, default=0.1, help="CMT lambda init")
parser.add_argument("--cmt_layers",           type=int,   default=2,   help="CMT layer count")

# ──────────────────────────────────────────────
#  DiffComp 新增：扩散网络超参
# ──────────────────────────────────────────────

# 扩散网络结构
parser.add_argument("--diff_n_layers",        type=int,   default=4,
                    help="FDN Transformer decoder 层数")
parser.add_argument("--diff_n_heads",         type=int,   default=4,
                    help="FDN 多头注意力头数")

# 扩散训练过程
parser.add_argument("--diff_T",               type=int,   default=1000,
                    help="扩散总步数（训练时的噪声计划步数）")
parser.add_argument("--diff_cfg_dropout",     type=float, default=0.1,
                    help="Classifier-Free Guidance 条件 dropout 概率")
parser.add_argument("--diff_lambda",          type=float, default=0.5,
                    help="扩散 loss 在总 loss 中的权重系数 λ")
parser.add_argument("--diff_alpha",           type=float, default=0.3,
                    help="属性/对象去噪路径在扩散 loss 中的权重 α")

# FDN 独立优化器
parser.add_argument("--diff_lr",             type=float, default=1e-4,
                    help="FDN 独立学习率（建议比主网络 lr 大）")
parser.add_argument("--diff_weight_decay",    type=float, default=1e-5,
                    help="FDN 权重衰减")

# 推理采样
parser.add_argument("--diff_sample_T",        type=int,   default=50,
                    help="DDIM 采样步数（推理时，远小于训练 diff_T）")
parser.add_argument("--diff_guidance_scale",  type=float, default=2.0,
                    help="CFG 引导系数 γ（1.5~3.0 稳定）")

# 合成特征库（开放世界推理）
parser.add_argument("--diff_synth_k",         type=int,   default=5,
                    help="每个候选组合采样次数，取均值降方差")
parser.add_argument("--use_synth_feat",       type=bool,  default=False,
                    help="开放世界推理时是否启用合成特征库")
parser.add_argument("--synth_weight",         type=float, default=0.3,
                    help="合成特征 logit 的融合权重")
parser.add_argument("--synth_build_batch",    type=int,   default=256,
                    help="构建合成库时每批处理的组合数（防止 OOM）")
