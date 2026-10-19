"""
DiffComp 训练脚本。
在原 Troika train.py 基础上的改动（标注 ── DiffComp ── 处）：
  1. FDN 使用独立优化器 diff_optimizer
  2. 每 step 额外调用 model.compute_diffusion_loss(batch)
  3. evaluate() 中，OW 模式下可选构建合成特征库并用 forward_with_synth 推理
  4. predict_logits_diffcomp 对齐原 predict_logits_text_first 的接口约定：
     - pairs 使用整数索引张量（而非字符串元组）
     - 分批次预计算文本特征
     - 正确调用 logit_infer / loss_calu
"""

import argparse
import os
import pickle

import numpy as np
import torch
import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

from model.model_factory import get_model
from parameters import parser

import test as test
from dataset import CompositionDataset
from utils import *


# ──────────────────────────────────────────────
#  DiffComp：FDN 独立优化器
# ──────────────────────────────────────────────

def get_diff_optimizer(model, config):
    """
    为 diff_net（FeatureDiffusionNet）单独建立 Adam 优化器。
    diff_lr 默认 1e-4，通常比 Troika 主网络 lr 大 2x，
    因为 FDN 从零开始训练，需要更快收敛。
    """
    diff_lr = getattr(config, 'diff_lr', 1e-4)
    diff_wd = getattr(config, 'diff_weight_decay', 1e-5)
    return torch.optim.Adam(
        list(model.diff_net.parameters()),
        lr=diff_lr,
        weight_decay=diff_wd,
    )


# ──────────────────────────────────────────────
#  主训练循环
# ──────────────────────────────────────────────

def train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    model.train()
    best_metric = 0
    best_loss   = 1e5
    best_epoch  = 0
    val_results = []

    scheduler = get_scheduler(optimizer, config, len(train_dataloader))

    # ── DiffComp：FDN 独立优化器及 epoch-level 调度器 ──────────
    diff_optimizer = get_diff_optimizer(model, config)
    diff_scheduler = torch.optim.lr_scheduler.StepLR(
        diff_optimizer,
        step_size=getattr(config, 'step_size', 5),
        gamma=getattr(config, 'gamma', 0.5),
    )
    diff_lambda = getattr(config, 'diff_lambda', 0.5)

    attr2idx    = train_dataset.attr2idx
    obj2idx     = train_dataset.obj2idx
    train_pairs = torch.tensor([
        (attr2idx[attr], obj2idx[obj])
        for attr, obj in train_dataset.train_pairs
    ]).cuda()

    train_losses = []

    for i in range(config.epoch_start, config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader),
            desc="epoch % 3d" % (i + 1),
        )
        epoch_troika_losses = []
        epoch_diff_losses   = []

        for bid, batch in enumerate(train_dataloader):

            # ── 1. Troika 主分支前向 + CE loss ────────────────
            # encode_image 在 forward() 内部执行，这里同时把视觉特征缓存下来
            # 做法：临时 hook 住 encode_image 的输出
            _cached_img_feat = [None]
            _orig_encode = model.encode_image

            def _encode_and_cache(x):
                feat, patch = _orig_encode(x)
                _cached_img_feat[0] = feat.detach()   # [B, D]，float16
                return feat, patch

            model.encode_image = _encode_and_cache
            predict     = model(batch, train_pairs)
            model.encode_image = _orig_encode          # 还原

            loss_troika = model.loss_calu(predict, batch)

            # ── 2. DiffComp：三路扩散 loss（复用已缓存的图像特征）──
            # 传入 cached_img_feat，彻底避免第二次 ViT forward
            loss_diff = model.compute_diffusion_loss(
                batch,
                cached_img_feat=_cached_img_feat[0],
            )

            # ── 3. 总 loss ────────────────────────────────────
            loss_total = (loss_troika + diff_lambda * loss_diff) \
                         / config.gradient_accumulation_steps
            loss_total.backward()

            if ((bid + 1) % config.gradient_accumulation_steps == 0) \
                    or (bid + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()
                diff_optimizer.step()
                diff_optimizer.zero_grad()

            scheduler = step_scheduler(scheduler, config, bid, len(train_dataloader))

            epoch_troika_losses.append(loss_troika.item())
            epoch_diff_losses.append(loss_diff.item())

            progress_bar.set_postfix({
                "troika": f"{np.mean(epoch_troika_losses[-50:]):.4f}",
                "diff":   f"{np.mean(epoch_diff_losses[-50:]):.4f}",
            })
            progress_bar.update()

        progress_bar.close()
        progress_bar.write(
            f"epoch {i+1}  "
            f"troika_loss={np.mean(epoch_troika_losses):.4f}  "
            f"diff_loss={np.mean(epoch_diff_losses):.4f}"
        )
        train_losses.append(np.mean(epoch_troika_losses))

        # FDN 调度器 epoch-level step
        diff_scheduler.step()

        if (i + 1) % config.save_every_n == 0:
            torch.save(model.state_dict(),
                       os.path.join(config.save_path, f"epoch_{i}.pt"))

        print("Evaluating val dataset:")
        val_result = evaluate(model, val_dataset, config)
        val_results.append(val_result)

        if config.val_metric == 'best_loss' and val_result['loss'] < best_loss:
            best_loss  = val_result['loss']
            best_epoch = i
            torch.save(model.state_dict(),
                       os.path.join(config.save_path, "val_best.pt"))
        elif config.val_metric != 'best_loss' and val_result[config.val_metric] > best_metric:
            best_metric = val_result[config.val_metric]
            best_epoch  = i
            torch.save(model.state_dict(),
                       os.path.join(config.save_path, "val_best.pt"))

        if i + 1 == config.epochs:
            print("--- Evaluating test dataset on Closed World ---")
            model.load_state_dict(
                torch.load(os.path.join(config.save_path, "val_best.pt"))
            )
            evaluate(model, test_dataset, config)

    if config.save_final_model:
        torch.save(model.state_dict(),
                   os.path.join(config.save_path, 'final_model.pt'))


# ──────────────────────────────────────────────
#  评估
# ──────────────────────────────────────────────

def evaluate(model, dataset, config):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)

    # ── DiffComp：OW 模式下可选构建合成特征库 ─────────────────
    use_synth  = getattr(config, 'use_synth_feat', False) and config.open_world
    synth_bank = None

    if use_synth:
        print("Building synthetic feature bank for open-world inference...")
        attr2idx  = dataset.attr2idx
        obj2idx   = dataset.obj2idx
        all_pairs = torch.tensor([
            (attr2idx[attr], obj2idx[obj])
            for attr, obj in dataset.pairs
        ]).cuda()
        synth_bank = model.build_synth_feat_bank(
            pairs      = all_pairs,
            batch_size = getattr(config, 'synth_build_batch', 256),
        )
        norms = synth_bank.norm(dim=-1)
        print(f"Synth bank built: {synth_bank.shape}  "
              f"norm=[{norms.min():.3f}, {norms.max():.3f}]")

    # ── 选择推理路径 ──────────────────────────────────────────
    if use_synth:
        # DiffComp OW 推理（合成特征融合）
        all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = \
            predict_logits_diffcomp(model, dataset, config, synth_bank=synth_bank)
    elif getattr(config, 'text_first', False):
        # 原 Troika text_first 路径（预计算全量文本特征）
        all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = \
            test.predict_logits_text_first(model, dataset, config)
    else:
        # 原 Troika 标准路径
        all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = \
            test.predict_logits(model, dataset, config)

    test_stats = test.test(
        dataset, evaluator,
        all_logits, all_attr_gt, all_obj_gt, all_pair_gt,
        config,
    )

    test_saved_results = {}
    result = ""
    key_set = ["best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"]
    for key in key_set:
        result += key + "  " + str(round(test_stats[key], 4)) + "| "
        test_saved_results[key] = round(test_stats[key], 4)
    print(result)
    test_saved_results['loss'] = loss_avg
    return test_saved_results


# ──────────────────────────────────────────────
#  DiffComp OW 推理：带合成特征库的 predict_logits
# ──────────────────────────────────────────────

def predict_logits_diffcomp(model, dataset, config, synth_bank):
    """
    使用合成特征库进行开放世界推理，接口完全对齐 predict_logits_text_first：
      - pairs 用整数索引张量（不是字符串元组），供 logit_infer 正确索引
      - 分批次预计算全量候选组合文本特征（防 OOM）
      - 每 batch 调用 model.forward_with_synth + model.logit_infer
      - loss 通过 model.loss_calu 正确计算（传入完整 3-branch logits）
    """
    model.eval()
    attr2idx = dataset.attr2idx
    obj2idx  = dataset.obj2idx

    # 候选组合：整数索引张量 [N_candidates, 2]（logit_infer 需要整数索引）
    pairs_str = dataset.pairs
    pairs = torch.tensor([
        (attr2idx[attr], obj2idx[obj])
        for attr, obj in pairs_str
    ]).cuda()                                       # [N, 2]

    # ── 分批次预计算全量文本特征（与 predict_logits_text_first 完全一致）──
    text_feats = [[], [], []]
    tbsz       = config.text_encoder_batch_size
    N          = pairs.shape[0]
    n_batches  = N // tbsz

    with torch.no_grad():
        for i in range(n_batches):
            cur_pairs = pairs[i * tbsz: (i + 1) * tbsz]
            cur_tf    = model.encode_text_for_open(cur_pairs)
            for k in range(3):
                text_feats[k].append(cur_tf[k])

        if N % tbsz != 0:
            cur_pairs = pairs[n_batches * tbsz:]
            cur_tf    = model.encode_text_for_open(cur_pairs)
            for k in range(3):
                text_feats[k].append(cur_tf[k])

    for k in range(3):
        text_feats[k] = torch.cat(text_feats[k], dim=0)   # [N, D]

    # ── 逐 batch 推理 ──────────────────────────────────────────
    dataloader = DataLoader(
        dataset,
        batch_size  = config.eval_batch_size,
        shuffle     = False,
        num_workers = config.num_workers,
    )

    all_logits  = torch.Tensor()
    all_attr_gt = []
    all_obj_gt  = []
    all_pair_gt = []
    loss_sum    = 0.0

    with torch.no_grad():
        for data in tqdm.tqdm(dataloader, total=len(dataloader),
                              desc="OW eval (DiffComp synth)"):
            # forward_with_synth 返回 [comp_fused, attr, obj]，图像只编码一次
            predict = model.forward_with_synth(data, text_feats, synth_bank)

            # logit_infer 需要整数索引的 pairs
            logits   = model.logit_infer(predict, pairs)

            # loss 使用完整 3-branch predict（与原 Troika evaluate 一致）
            loss_sum += model.loss_calu(predict, data).item()

            all_logits  = torch.cat([all_logits, logits.cpu()], dim=0)
            all_attr_gt.append(data[1])
            all_obj_gt.append(data[2])
            all_pair_gt.append(data[3])

    all_attr_gt = torch.cat(all_attr_gt).cpu()
    all_obj_gt  = torch.cat(all_obj_gt).cpu()
    all_pair_gt = torch.cat(all_pair_gt).cpu()
    loss_avg    = loss_sum / len(dataloader)

    return all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg


# ──────────────────────────────────────────────
#  入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    config = parser.parse_args()
    if config.yml_path:
        load_args(config.yml_path, config)
    print(config)
    set_seed(config.seed)

    dataset_path = config.dataset_path

    train_dataset = CompositionDataset(
        dataset_path, phase='train',
        split='compositional-split-natural',
        same_prim_sample=config.same_prim_sample,
    )
    val_dataset = CompositionDataset(
        dataset_path, phase='val',
        split='compositional-split-natural',
    )
    test_dataset = CompositionDataset(
        dataset_path, phase='test',
        split='compositional-split-natural',
    )

    allattrs   = train_dataset.attrs
    allobj     = train_dataset.objs
    classes    = [c.replace(".", " ").lower() for c in allobj]
    attributes = [a.replace(".", " ").lower() for a in allattrs]
    offset     = len(attributes)

    model     = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()
    optimizer = get_optimizer(model, config)

    os.makedirs(config.save_path, exist_ok=True)

    train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset)

    with open(os.path.join(config.save_path, "config.pkl"), "wb") as fp:
        pickle.dump(config, fp)
    write_json(os.path.join(config.save_path, "config.json"), vars(config))
    print("done!")
