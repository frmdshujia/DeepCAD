"""
engine_contrast.py
单轮训练与验证：
  - train_one_epoch : 前向、在线 S_GT、soft InfoNCE、反向传播
  - validate        : 无梯度验证集 loss + 基础指标
"""

import math
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(
    __import__('os').path.abspath(__file__))))

from util import misc
from util import lr_sched
from contrastive_pretrain.loss_contrast import (
    soft_infonce_loss_with_sgt,
    hard_infonce_loss,
    compute_sgt,
)


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

        # ── 采样 K 个 CMR：前 B 行为本 batch 各 EID 的正样本，其余为随机负样本 ──
        if isinstance(eids, torch.Tensor):
            eids_list = eids.detach().cpu().tolist()
        else:
            eids_list = list(eids)
        if getattr(args, 'cmr_sample_mode', 'random') == 'stratified':
            _, pc_cmr = cmr_bank.sample_with_batch_positives_stratified(
                eids_list, args.cmr_sample_k)
        else:
            _, pc_cmr = cmr_bank.sample_with_batch_positives(
                eids_list, args.cmr_sample_k)  # (K, n_pc)

        # ── 混合精度前向 ──
        with torch.cuda.amp.autocast():
            z_fundus = fundus_model(images)                # (B, d)
            z_cmr = cmr_encoder(pc_cmr)                   # (K, d)（含梯度）

            loss_type = getattr(args, 'loss_type', 'soft')
            if loss_type == 'hard':
                loss = hard_infonce_loss(
                    z_fundus, z_cmr, temperature=args.temperature)
                s_gt = compute_sgt(pc_fundus, pc_cmr, sigma=args.sigma)
            else:
                loss, s_gt = soft_infonce_loss_with_sgt(
                    z_fundus, z_cmr, pc_fundus, pc_cmr,
                    sigma=args.sigma, temperature=args.temperature,
                    sgt_temp=getattr(args, 'sgt_temp', 1.0),
                )

            # ── CMR 均匀性正则化：防止 CMR encoder 全局坍塌（modality gap）──
            unif_weight = getattr(args, 'unif_weight', 0.0)
            if unif_weight > 0 and z_cmr.shape[0] >= 2:
                B_size = z_fundus.shape[0]
                # 对正样本 CMR（前 B 行）计算均匀性损失
                z_pos_cmr = z_cmr[:B_size].float()
                sq_pdist = torch.cdist(z_pos_cmr, z_pos_cmr, p=2).pow(2)
                mask = ~torch.eye(B_size, dtype=torch.bool, device=sq_pdist.device)
                unif_loss = sq_pdist[mask].mul(-2).exp().mean().log()
                # 最大化均匀性 = 最小化 unif_loss（注意 log(exp(-2||...||^2)) 越大越均匀）
                loss = loss - unif_weight * unif_loss

        # ── Part B 诊断：batch 内 S_GT（剔除同人位置 (i,i)）的 mean / std ──
        if getattr(args, "log_sgt_batch_stats", False) and misc.get_rank() == 0:
            every = max(1, int(getattr(args, "sgt_batch_stats_every", 1)))
            if step % every == 0:
                with torch.no_grad():
                    Bcur, _Kcur = s_gt.shape
                    idx = torch.arange(Bcur, device=s_gt.device)
                    mask = torch.ones_like(s_gt, dtype=torch.bool)
                    mask[idx, idx] = False
                    sel = s_gt[mask]
                    sm = float(sel.mean().item())
                    sstd = float(sel.std(unbiased=False).item())
                    smin = float(sel.min().item())
                    smax = float(sel.max().item())
                csv_path = getattr(args, "sgt_batch_stats_csv", None) or os.path.join(
                    getattr(args, "output_dir", "."), "sgt_batch_stats.csv"
                )
                csv_path = os.path.abspath(csv_path)
                os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
                write_header = not os.path.isfile(csv_path)
                with open(csv_path, "a", newline="") as cf:
                    if write_header:
                        cf.write(
                            "global_step,epoch,step,sgt_mean_masked,sgt_std_masked,"
                            "sgt_min_masked,sgt_max_masked,B,K\n"
                        )
                    cf.write(
                        f"{global_step},{epoch},{step},{sm},{sstd},{smin},{smax},{Bcur},{_Kcur}\n"
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
            # 目标熵监控
            sgt_t = max(float(getattr(args, 'sgt_temp', 1.0)), 1e-6)
            with torch.no_grad():
                pi = F.softmax(s_gt / sgt_t, dim=-1)
                tgt_ent = (-pi * (pi.clamp_min(1e-12).log())).sum(dim=-1).mean()
                # 配对余弦相似度：第 i 个 fundus 与第 i 个 CMR（正样本）
                B_cur = z_fundus.size(0)
                paired_cos = (z_fundus.detach() * z_cmr[:B_cur].detach()).sum(dim=-1).mean()
            log_writer.add_scalar('train/target_entropy', tgt_ent.item(), global_step)
            log_writer.add_scalar('train/paired_cosine', paired_cos.item(), global_step)

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

    all_z_fundus, all_z_cmr_paired = [], []

    for images, eids, pc_fundus in metric_logger.log_every(loader, 50, header):
        images = images.cuda(non_blocking=True)
        pc_fundus = pc_fundus.cuda(non_blocking=True)

        if isinstance(eids, torch.Tensor):
            eids_list = eids.detach().cpu().tolist()
        else:
            eids_list = list(eids)
        _, pc_cmr = cmr_bank.sample_with_batch_positives(eids_list, args.cmr_sample_k)

        with torch.cuda.amp.autocast():
            z_fundus = fundus_model(images)
            z_cmr = cmr_encoder(pc_cmr)

            loss_type = getattr(args, 'loss_type', 'soft')
            if loss_type == 'hard':
                loss = hard_infonce_loss(
                    z_fundus, z_cmr, temperature=args.temperature)
            else:
                loss, _ = soft_infonce_loss_with_sgt(
                    z_fundus, z_cmr, pc_fundus, pc_cmr,
                    sigma=args.sigma, temperature=args.temperature,
                    sgt_temp=getattr(args, 'sgt_temp', 1.0),
                )

        metric_logger.update(loss=loss.item())

        B_cur = z_fundus.size(0)
        # 记录配对余弦相似度（正样本在前 B 位）
        paired_cos = (z_fundus * z_cmr[:B_cur]).sum(dim=-1).mean().item()
        metric_logger.update(paired_cosine=paired_cos)

        all_z_fundus.append(z_fundus.cpu())
        all_z_cmr_paired.append(z_cmr[:B_cur].cpu())

    metric_logger.synchronize_between_processes()
    val_loss = metric_logger.loss.global_avg
    val_paired_cos = getattr(metric_logger, 'paired_cosine', None)
    val_paired_cos = val_paired_cos.global_avg if val_paired_cos is not None else float('nan')

    # ── Uniformity + paired alignment ──
    z_f_all = F.normalize(torch.cat(all_z_fundus, dim=0), dim=-1)
    z_c_all = F.normalize(torch.cat(all_z_cmr_paired, dim=0), dim=-1)
    unif_f = _uniformity(z_f_all).item()
    # 配对 alignment：||z_f - z_c||² 均值（越小越好）
    alignment = (z_f_all - z_c_all).pow(2).sum(dim=-1).mean().item()

    stats = {
        'val_loss': val_loss,
        'val_paired_cosine': val_paired_cos,
        'val_alignment': alignment,
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
