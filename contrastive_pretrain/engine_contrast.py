"""
engine_contrast.py
单轮训练与验证：
  - train_one_epoch : 前向、在线 S_GT、soft InfoNCE、反向传播
  - validate        : 无梯度验证集 loss + 基础指标
"""

import math
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(
    __import__('os').path.abspath(__file__))))

from util import misc
from util import lr_sched
from contrastive_pretrain.loss_contrast import soft_infonce_loss_with_sgt


def train_one_epoch(
    fundus_model,
    cmr_encoder,
    loader,
    cmr_bank,
    optimizer,
    loss_scaler,
    epoch: int,
    args,
    log_writer=None,
):
    """
    单轮训练。

    Args:
        fundus_model : FundusContrastModel（可能已 DDP 包装）
        cmr_encoder  : CMREncoder（可能已 DDP 包装）
        loader       : FundusContrastDataset 的 DataLoader
        cmr_bank     : CMRBank（全量 CMR PC scores，在 GPU 上）
        optimizer    : AdamW
        loss_scaler  : NativeScalerWithGradNormCount（混合精度）
        epoch        : 当前轮次
        args         : argparse 命名空间，需含 sigma, temperature, cmr_sample_k,
                       lr, min_lr, warmup_epochs, epochs, clip_grad
        log_writer   : TensorBoard SummaryWriter（仅 rank-0 非 None）
    """
    fundus_model.train()
    cmr_encoder.train()

    metric_logger = misc.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr_backbone', misc.SmoothedValue(window_size=1, fmt='{value:.2e}'))
    metric_logger.add_meter('lr_proj',     misc.SmoothedValue(window_size=1, fmt='{value:.2e}'))
    metric_logger.add_meter('lr_cmr',      misc.SmoothedValue(window_size=1, fmt='{value:.2e}'))
    header = f'Epoch [{epoch}]'
    print_freq = 50

    n_steps_per_epoch = len(loader)

    for step, (images, eids, pc_fundus) in enumerate(
            metric_logger.log_every(loader, print_freq, header)):

        # ── 按步更新学习率（cosine schedule + warmup）──
        global_step = epoch * n_steps_per_epoch + step
        # adjust_learning_rate 使用 epoch 粒度，这里传入浮点 epoch
        lr_sched.adjust_learning_rate(
            optimizer,
            global_step / n_steps_per_epoch,
            args,
        )

        images = images.cuda(non_blocking=True)       # (B, 3, 224, 224)
        pc_fundus = pc_fundus.cuda(non_blocking=True) # (B, n_pc)

        # ── 采样 K 个 CMR PC scores ──
        _, pc_cmr = cmr_bank.sample(args.cmr_sample_k)  # (K, n_pc)

        # ── 混合精度前向 ──
        with torch.cuda.amp.autocast():
            z_fundus = fundus_model(images)                # (B, d)
            z_cmr = cmr_encoder(pc_cmr)                   # (K, d)（含梯度）
            loss, s_gt = soft_infonce_loss_with_sgt(
                z_fundus, z_cmr, pc_fundus, pc_cmr,
                sigma=args.sigma, temperature=args.temperature,
            )

        if not math.isfinite(loss.item()):
            print(f'Loss is {loss.item()}, stopping training')
            sys.exit(1)

        # ── 反向传播 ──
        optimizer.zero_grad()
        loss_scaler(
            loss, optimizer,
            clip_grad=getattr(args, 'clip_grad', None),
            parameters=list(fundus_model.parameters()) + list(cmr_encoder.parameters()),
            create_graph=False,
        )

        torch.cuda.synchronize()

        # ── 记录指标 ──
        loss_val = loss.item()
        metric_logger.update(loss=loss_val)

        # 记录三个组件的当前 lr
        lrs = [pg['lr'] for pg in optimizer.param_groups]
        # backbone 组：取最后一个非 proj/cmr 组（top layer）
        backbone_lrs = lrs[:-2]
        metric_logger.update(lr_backbone=max(backbone_lrs) if backbone_lrs else 0.0)
        metric_logger.update(lr_proj=lrs[-2])
        metric_logger.update(lr_cmr=lrs[-1])

        if log_writer is not None and (global_step % 50 == 0):
            log_writer.add_scalar('train/loss', loss_val, global_step)
            log_writer.add_scalar('train/lr_backbone', max(backbone_lrs) if backbone_lrs else 0., global_step)
            log_writer.add_scalar('train/lr_proj', lrs[-2], global_step)
            log_writer.add_scalar('train/lr_cmr', lrs[-1], global_step)

    metric_logger.synchronize_between_processes()
    print(f'[Train] Averaged stats: {metric_logger}')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(
    fundus_model,
    cmr_encoder,
    loader,
    cmr_bank,
    args,
    log_writer=None,
    epoch: int = 0,
):
    """
    验证集 loss + 在线 alignment & uniformity 监控。

    Returns:
        dict: {'val_loss': float, 'alignment': float, 'uniformity_f': float, 'uniformity_c': float}
    """
    fundus_model.eval()
    cmr_encoder.eval()

    metric_logger = misc.MetricLogger(delimiter='  ')
    header = 'Val:'

    all_z_fundus, all_z_cmr_matched = [], []

    for images, eids, pc_fundus in metric_logger.log_every(loader, 50, header):
        images = images.cuda(non_blocking=True)
        pc_fundus = pc_fundus.cuda(non_blocking=True)

        _, pc_cmr = cmr_bank.sample(args.cmr_sample_k)

        with torch.cuda.amp.autocast():
            z_fundus = fundus_model(images)
            z_cmr = cmr_encoder(pc_cmr)
            loss, _ = soft_infonce_loss_with_sgt(
                z_fundus, z_cmr, pc_fundus, pc_cmr,
                sigma=args.sigma, temperature=args.temperature,
            )

        metric_logger.update(loss=loss.item())

        # 收集嵌入用于 alignment 计算（用同 batch 中 pc_cmr 中与 pc_fundus 最近的 CMR 近似配对）
        all_z_fundus.append(z_fundus.cpu())

    metric_logger.synchronize_between_processes()
    val_loss = metric_logger.loss.global_avg

    # ── Uniformity（用本 epoch 所有验证集 fundus embedding）──
    z_f_all = F.normalize(torch.cat(all_z_fundus, dim=0), dim=-1)
    unif_f = _uniformity(z_f_all).item()

    stats = {
        'val_loss': val_loss,
        'uniformity_fundus': unif_f,
    }
    print(f'[Val] {stats}')

    if log_writer is not None:
        for k, v in stats.items():
            log_writer.add_scalar(f'val/{k}', v, epoch)

    return stats


def _uniformity(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """Wang & Isola uniformity，纯 torch 版本（避免 import 循环）。"""
    sq_dists = torch.cdist(z, z, p=2).pow(2)
    return sq_dists.mul(-t).exp().mean().log()
