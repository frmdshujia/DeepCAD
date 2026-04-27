"""
main_contrast.py
对比学习预训练主入口。

启动方式（多卡）：
    torchrun --nproc_per_node=4 contrastive_pretrain/main_contrast.py [args]

或单卡调试：
    python contrastive_pretrain/main_contrast.py --gpu 0 [args]
"""

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

# ── 项目根目录加入 sys.path，使 models_vit / util 等可被导入 ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 必须先 import torch，再注入 torch._six 占位（与 main_finetune.py 一致），否则 PyTorch 1.8 初始化失败
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import collections.abc
if 'torch._six' not in sys.modules:
    _inf = float('inf')
    class _TorchSix:
        container_abcs = collections.abc
        inf = _inf
    sys.modules['torch._six'] = _TorchSix()

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from contrastive_pretrain.datasets_contrast import (
    FundusContrastDataset,
    UniqueEIDSampler,
    DistributedUniqueEIDSampler,
    CMRBank,
)
from contrastive_pretrain.models_contrast import (
    FundusContrastModel,
    CMREncoder,
    build_param_groups,
)
from contrastive_pretrain.engine_contrast import train_one_epoch, validate
from contrastive_pretrain.eval_contrast import run_full_eval


# ─────────────────────────────────────────────
#  参数解析
# ─────────────────────────────────────────────
def get_args_parser():
    parser = argparse.ArgumentParser('Soft-label Contrastive Pretraining', add_help=False)

    # ── 数据 ──
    parser.add_argument('--fundus_csv', required=True, type=str,
                        help='Fundus 表 CSV 路径（数据 Agent 交付）')
    parser.add_argument('--cmr_csv', required=True, type=str,
                        help='CMR 表 CSV 路径（数据 Agent 交付）')
    parser.add_argument('--pc_cols', required=True, type=str,
                        help='14 维 PC score 列名，逗号分隔，如 M1_PC1,M1_PC2,...')
    parser.add_argument('--sigma', required=True, type=float,
                        help='高斯核带宽 σ（数据 Agent 在训练集 CMR 上用 median heuristic 估计）')

    # ── 子采样（保守探索 / 烟雾测，仅影响 train fundus 与可选 CMR train bank）──
    parser.add_argument('--fundus_train_subset_ratio', type=float, default=1.0,
                        help='仅 train：随机保留比例，1.0=全量')
    parser.add_argument('--fundus_max_train_samples', type=int, default=0,
                        help='仅 train：最多保留多少行（0=不限制；>0 时优先于 subset_ratio）')
    parser.add_argument('--subset_seed', type=int, default=0,
                        help='train fundus / CMR 子采样随机种子')
    parser.add_argument('--cmr_train_max_rows', type=int, default=0,
                        help='train CMR bank 最多行数（0=全量；烟雾测可设 8000–20000 减负）')

    # ── 模型 ──
    parser.add_argument('--finetune', required=True, type=str,
                        help='RETFound 预训练权重路径（.pth）')
    parser.add_argument('--proj_dim', default=256, type=int,
                        help='对比空间维度 d（128 或 256）')
    parser.add_argument('--drop_path', default=0.1, type=float,
                        help='ViT drop path rate')

    # ── 损失 ──
    parser.add_argument('--loss_type', default='soft', choices=['soft', 'hard'],
                        help='soft=Soft-label InfoNCE（用softmax(S_GT/τ_g)做目标）；'
                             'hard=标准InfoNCE（位置i为正样本，其余为负样本）')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='InfoNCE 温度参数（嵌入 logits）')
    parser.add_argument('--sgt_temp', default=1.0, type=float,
                        help='目标分布锐化：target=softmax(S_GT/τ_g)。1.0=旧版；'
                             '(0,1) 更尖，如 0.5；>1 更平。须为正数。')
    parser.add_argument('--cmr_sample_k', default=4096, type=int,
                        help='每 batch 从 CMR bank 随机采样的 CMR 数量')
    parser.add_argument(
        '--cmr_sample_mode', default='random', choices=['random', 'stratified'],
        help='random=均匀随机负样本；stratified=按 S_GT 高/中/低分层采样（需 --stratified_buckets_pkl）',
    )
    parser.add_argument(
        '--stratified_buckets_pkl', type=str, default='',
        help='precompute_stratified_buckets.py 输出的 .pkl；stratified 模式必填',
    )
    parser.add_argument(
        '--strat_neg_frac_high', type=float, default=1.0 / 3.0,
        help='负样本中 high 桶 (S_GT>thresh_high) 目标占比',
    )
    parser.add_argument(
        '--strat_neg_frac_low', type=float, default=1.0 / 3.0,
        help='负样本中 low 桶 (S_GT<thresh_low) 目标占比；余量为 mid',
    )

    # ── 训练 ──
    parser.add_argument('--batch_size', default=64, type=int,
                        help='每 GPU batch size（每 EID 一张图）')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup_epochs', default=10, type=int)

    # ── 优化器 ──
    parser.add_argument('--blr', default=1e-5, type=float,
                        help='backbone top layer 基础 LR（实际 lr = blr × total_bs/256）')
    parser.add_argument('--min_lr', default=1e-7, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--layer_decay', default=0.75, type=float,
                        help='ViT 层间 LR 衰减系数')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='冻结 ViT backbone（只训练 proj_head + CMR encoder），'
                             '数据量少时用于防止过拟合')
    parser.add_argument('--proj_lr_scale', default=10.0, type=float,
                        help='Projection head LR = blr × proj_lr_scale（默认 1e-4）')
    parser.add_argument('--cmr_lr_scale', default=100.0, type=float,
                        help='CMR MLP LR = blr × cmr_lr_scale（默认 1e-3）')
    parser.add_argument('--clip_grad', default=1.0, type=float,
                        help='梯度裁剪范数（None 表示不裁剪）')
    parser.add_argument('--unif_weight', default=0.0, type=float,
                        help='CMR 均匀性正则化权重（>0 时防止 CMR encoder 全局坍塌）')

    # ── S_GT 分布诊断（Step 1 Part B：训练时写 CSV，配合 diagnostics_sgt_distribution.py）──
    parser.add_argument('--log_sgt_batch_stats', action='store_true',
                        help='每步记录 batch 内 S_GT（剔除同人 (i,i)）的 mean/std 到 CSV')
    parser.add_argument('--sgt_batch_stats_csv', type=str, default=None,
                        help='Part B 统计输出路径（默认：<output_dir>/sgt_batch_stats.csv）')
    parser.add_argument('--sgt_batch_stats_every', type=int, default=1,
                        help='每多少个 train step 写一行（减小文件）')

    # ── 早停 ──
    parser.add_argument('--patience', default=12, type=int,
                        help='最佳指标无提升时的早停 patience（见 --metric_for_best）')
    parser.add_argument(
        '--metric_for_best', default='R@5',
        type=str,
        choices=['R@5', 'gt_pred_spearman', 'gt_pred_pearson', 'val_loss'],
        help='保存 best checkpoint / 早停所依据的验证集指标。'
             'gt_pred_* 需完整 eval（勿与 --skip_full_eval 同用）。'
             '端到端微调相关系数时常用 gt_pred_spearman 或 gt_pred_pearson。',
    )
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='每隔多少 epoch 计算一次完整 retrieval + gt_pred')
    parser.add_argument('--skip_full_eval', action='store_true',
                        help='跳过跨模态 retrieval 全量评估，仅用 val_loss 早停（烟雾测推荐）')

    # ── 输出 ──
    parser.add_argument('--output_dir', default='./output_dir/contrast',
                        help='Checkpoint 与日志保存目录')
    parser.add_argument('--log_dir', default=None,
                        help='TensorBoard 日志目录（默认与 output_dir 相同）')
    parser.add_argument('--save_freq', default=10, type=int,
                        help='每隔多少 epoch 保存一次 checkpoint')

    # ── 运行环境 ──
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True)
    parser.add_argument('--resume', default='', help='从 checkpoint 恢复训练')

    # ── 分布式 ──
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--gpu', default='0,1,2,3', type=str,
                        help='可见 GPU 列表（单卡调试时设为 "0"）')

    # ── 说明 ──
    parser.add_argument('--desc', default='', type=str, help='本次训练描述')

    return parser


def _init_best_metric_value(metric_name: str) -> float:
    return float('inf') if metric_name == 'val_loss' else float('-inf')


def _get_metric_value(args, eval_metrics, val_stats, recall5_val: float):
    """当前 epoch 用于 early-stopping / best-ckpt 的标量。"""
    m = args.metric_for_best
    if m == 'val_loss':
        return float(val_stats['val_loss'])
    if eval_metrics is None:
        return float('nan')
    if m == 'R@5':
        return float(eval_metrics.get('R@5', float('nan')))
    return float(eval_metrics.get(m, float('nan')))


def _is_improvement(args, cur: float, best: float) -> bool:
    if math.isnan(cur):
        return False
    if args.metric_for_best == 'val_loss':
        return cur < best - 1e-7
    return cur > best + 1e-7


# ─────────────────────────────────────────────
#  主函数
# ─────────────────────────────────────────────
def main(args):
    # 在 init_distributed_mode 之前设置 CUDA_VISIBLE_DEVICES
    # （init_distributed_mode 会把 args.gpu 从字符串改写为整型 local_rank）
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    misc.init_distributed_mode(args)
    # 此后 args.gpu 为整型 local rank，可直接用于 DDP device_ids

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # 解析 PC 列名
    pc_cols = [c.strip() for c in args.pc_cols.split(',')]
    args.n_pc = len(pc_cols)
    print(f'[main] PC columns ({args.n_pc}): {pc_cols}')

    # ── 计算实际 LR ──
    # effective_batch_size = batch_size × world_size（UniqueEIDSampler 每卡一份）
    eff_bs = args.batch_size * misc.get_world_size()
    args.lr = args.blr * eff_bs / 256
    print(f'[main] blr={args.blr}, eff_bs={eff_bs}, lr={args.lr:.2e}')
    print(f'[main] loss_type={args.loss_type}, temperature={args.temperature}, '
          f'sgt_temp={args.sgt_temp} sigma={args.sigma}')
    print(f'[main] cmr_sample_mode={getattr(args, "cmr_sample_mode", "random")} '
          f'strat_pkl={getattr(args, "stratified_buckets_pkl", "") or "(none)"}')
    print(f'[main] metric_for_best={args.metric_for_best}')
    if args.metric_for_best != 'val_loss' and args.skip_full_eval:
        raise ValueError(
            '--metric_for_best 为 R@5 或 gt_pred_* 时必须做完整 eval，请去掉 --skip_full_eval'
        )

    # ── 数据集 ──
    dataset_train = FundusContrastDataset(
        args.fundus_csv, pc_cols, split='train',
        train_subset_ratio=args.fundus_train_subset_ratio,
        train_max_samples=args.fundus_max_train_samples,
        subset_seed=args.subset_seed,
    )
    dataset_val = FundusContrastDataset(args.fundus_csv, pc_cols, split='val')

    # ── Sampler ──
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if args.distributed:
        sampler_train = DistributedUniqueEIDSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, seed=args.seed)
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_train = UniqueEIDSampler(dataset_train, seed=args.seed)
        sampler_val   = torch.utils.data.SequentialSampler(dataset_val)

    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True,
    )
    loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )

    # 用于 run_full_eval 的独立 loader：顺序采样全量 val 数据（仅 rank-0 使用）
    eval_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=torch.utils.data.SequentialSampler(dataset_val),
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=False, drop_last=False,
    )

    # ── CMR Bank（全量或子集 PC scores 加载到 GPU）──
    strat_path = (args.stratified_buckets_pkl or '').strip() or None
    if getattr(args, 'cmr_sample_mode', 'random') == 'stratified':
        if not strat_path:
            raise ValueError('cmr_sample_mode=stratified 时必须提供 --stratified_buckets_pkl')
        if args.cmr_train_max_rows and args.cmr_train_max_rows > 0:
            raise ValueError(
                '分层采样与 cmr_train_max_rows>0 不兼容，请设 cmr_train_max_rows=0 并使用全量 CMR bank'
            )

    cmr_bank = CMRBank(
        args.cmr_csv, pc_cols, split='train', device=str(device),
        max_rows=args.cmr_train_max_rows, subset_seed=args.subset_seed,
        stratified_buckets_path=strat_path,
        strat_neg_frac_high=args.strat_neg_frac_high,
        strat_neg_frac_low=args.strat_neg_frac_low,
        stratified_rng_rank=int(global_rank),
    )
    cmr_bank_val = CMRBank(args.cmr_csv, pc_cols, split='val', device=str(device))

    # ── 模型 ──
    fundus_model = FundusContrastModel(proj_dim=args.proj_dim, drop_path_rate=args.drop_path)
    fundus_model.load_pretrained(args.finetune)
    fundus_model.to(device)

    cmr_encoder = CMREncoder(in_dim=args.n_pc, hidden_dim=128, out_dim=args.proj_dim)
    cmr_encoder.to(device)

    # ── 冻结 backbone（数据量小时防过拟合）──
    if getattr(args, 'freeze_backbone', False):
        for param in fundus_model.backbone.parameters():
            param.requires_grad_(False)
        print('[main] ViT backbone 已冻结，只训练 proj_head + CMR encoder')
    else:
        print('[main] ViT backbone 参与训练：RETFound 全参微调 + proj_head + CMR encoder（双塔可训练）')

    _n_tr = sum(p.numel() for p in fundus_model.parameters() if p.requires_grad)
    _n_tr += sum(p.numel() for p in cmr_encoder.parameters() if p.requires_grad)
    print(f'[main] Trainable parameters (fundus + CMR): {_n_tr:,}')

    # ── DDP 包装 ──
    # init_distributed_mode 后 args.gpu 已是整型 local rank
    if args.distributed:
        fundus_model = torch.nn.parallel.DistributedDataParallel(
            fundus_model, device_ids=[args.gpu])
        cmr_encoder = torch.nn.parallel.DistributedDataParallel(
            cmr_encoder, device_ids=[args.gpu])

    fundus_model_without_ddp = fundus_model.module if args.distributed else fundus_model
    cmr_encoder_without_ddp  = cmr_encoder.module  if args.distributed else cmr_encoder

    # ── 优化器（分层 LR）──
    param_groups = build_param_groups(
        fundus_model_without_ddp,
        cmr_encoder_without_ddp,
        weight_decay=args.weight_decay,
        layer_decay=args.layer_decay,
        proj_lr_scale=args.proj_lr_scale,
        cmr_lr_scale=args.cmr_lr_scale,
    )
    optimizer = torch.optim.AdamW(param_groups)
    loss_scaler = NativeScaler()

    # ── TensorBoard ──
    log_dir = args.log_dir or args.output_dir
    log_writer = None
    if global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        tag = f'contrast_{args.desc}' if args.desc else 'contrast'
        log_writer = SummaryWriter(log_dir=os.path.join(log_dir, tag))

    # ── 恢复训练 ──
    start_epoch = 0
    best_recall5 = 0.0
    best_val_loss = float('inf')
    best_metric_value = _init_best_metric_value(args.metric_for_best)
    no_improve = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        fundus_model_without_ddp.load_state_dict(ckpt['fundus_model'])
        cmr_encoder_without_ddp.load_state_dict(ckpt['cmr_encoder'])
        optimizer.load_state_dict(ckpt['optimizer'])
        loss_scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_recall5 = ckpt.get('best_recall5', 0.0)
        if 'best_val_loss' in ckpt:
            best_val_loss = ckpt['best_val_loss']
        no_improve = ckpt.get('no_improve', 0)
        if 'best_metric_value' in ckpt:
            best_metric_value = ckpt['best_metric_value']
        print(f'[main] Resumed from epoch {start_epoch}, best_recall5={best_recall5:.4f}, '
              f'best_val_loss={best_val_loss}, best_{args.metric_for_best}={best_metric_value}')

    # ── 保存 args ──
    if global_rank == 0:
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

    print(f'\n[main] Start training for {args.epochs} epochs '
          f'(start_epoch={start_epoch})\n')
    start_time = time.time()

    # ─────────────────────── Epoch 循环 ───────────────────────
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        # ── 训练 ──
        train_stats = train_one_epoch(
            fundus_model, cmr_encoder, loader_train, cmr_bank,
            optimizer, loss_scaler, epoch, args, log_writer,
        )

        # ── 验证 ──
        val_stats = validate(
            fundus_model, cmr_encoder, loader_val, cmr_bank_val,
            args, log_writer, epoch,
        )

        # ── 完整 retrieval + gt_pred（可选；烟雾测可 --skip_full_eval）──
        recall5 = best_recall5
        eval_metrics = None
        do_full_eval = (
            (not args.skip_full_eval)
            and ((epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1)
        )
        if do_full_eval:
            if global_rank == 0:
                eval_metrics = run_full_eval(
                    fundus_model_without_ddp, cmr_encoder_without_ddp,
                    eval_loader_val, cmr_bank_val, device=str(device),
                    sigma=float(args.sigma),
                )
                recall5 = eval_metrics.get('R@5', 0.0)
                print(f'[Eval epoch {epoch}] {eval_metrics}')

                if log_writer is not None:
                    for k, v in eval_metrics.items():
                        log_writer.add_scalar(f'eval/{k}', v, epoch)

            if args.distributed:
                pack = torch.zeros(3, dtype=torch.float64, device=device)
                if global_rank == 0:
                    pack[0] = float(eval_metrics.get('R@5', 0.0))
                    pack[1] = float(eval_metrics.get('gt_pred_spearman', float('nan')))
                    pack[2] = float(eval_metrics.get('gt_pred_pearson', float('nan')))
                torch.distributed.broadcast(pack, src=0)
                recall5 = float(pack[0].item())
                if global_rank != 0:
                    eval_metrics = {
                        'R@5': pack[0].item(),
                        'gt_pred_spearman': pack[1].item(),
                        'gt_pred_pearson': pack[2].item(),
                    }

        val_loss_curr = val_stats['val_loss']

        # ── 早停 & 保存最佳 checkpoint（由 --metric_for_best 决定）──
        if args.skip_full_eval:
            cur_m = val_loss_curr
            is_best = cur_m < best_val_loss - 1e-6
            if is_best:
                best_val_loss = cur_m
                no_improve = 0
            else:
                no_improve += 1
        elif args.metric_for_best == 'val_loss':
            cur_m = val_loss_curr
            is_best = _is_improvement(args, cur_m, best_metric_value)
            if is_best:
                best_metric_value = cur_m
                best_val_loss = cur_m
                no_improve = 0
            else:
                no_improve += 1
        else:
            if do_full_eval:
                cur_m = _get_metric_value(args, eval_metrics, val_stats, recall5)
                is_best = _is_improvement(args, cur_m, best_metric_value)
                if is_best:
                    best_metric_value = cur_m
                    best_recall5 = max(best_recall5, recall5)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                is_best = False

        if global_rank == 0:
            _save_checkpoint(
                args, epoch, fundus_model_without_ddp, cmr_encoder_without_ddp,
                optimizer, loss_scaler, best_recall5, no_improve,
                is_best=is_best, best_val_loss=best_val_loss,
                best_metric_value=best_metric_value,
            )

        # ── 日志 ──
        log_stats = {
            'epoch': epoch,
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
            'recall5': recall5,
            'best_recall5': best_recall5,
            'best_val_loss': best_val_loss,
            'best_metric_value': best_metric_value,
            'metric_for_best': args.metric_for_best,
            'no_improve': no_improve,
        }
        if eval_metrics is not None and global_rank == 0:
            for k in ('gt_pred_spearman', 'gt_pred_pearson', 'R@5', 'paired_cosine'):
                if k in eval_metrics:
                    log_stats[f'eval_{k}'] = eval_metrics[k]
        if global_rank == 0:
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

        if no_improve >= args.patience:
            print(f'[main] Early stopping at epoch {epoch} '
                  f'(no improvement for {args.patience} epochs)')
            break

    # ─────────────────────── 训练结束 ───────────────────────
    total_time = time.time() - start_time
    print(f'\n[main] Training complete in '
          f'{datetime.timedelta(seconds=int(total_time))}')
    if args.skip_full_eval:
        print(f'[main] Best val_loss = {best_val_loss:.6f} (skip_full_eval 模式)')
    else:
        print(f'[main] Best {args.metric_for_best} = {best_metric_value} | '
              f'Recall@5 = {best_recall5:.4f}')

    # 保存下游微调兼容的 encoder checkpoint
    if global_rank == 0:
        compat_path = os.path.join(args.output_dir, 'contrast_pretrain_encoder.pth')
        fundus_model_without_ddp.save_encoder_ckpt(compat_path)
        print(f'[main] Compatible encoder saved → {compat_path}')
        print(f'[main] 下游微调命令：')
        print(f'  python main_finetune.py --finetune {compat_path} ...')


# ─────────────────────────────────────────────
#  Checkpoint 保存工具
# ─────────────────────────────────────────────
def _save_checkpoint(args, epoch, fundus_model, cmr_encoder,
                     optimizer, scaler, best_recall5, no_improve, is_best=False,
                     best_val_loss=float('inf'), best_metric_value=None):
    ckpt = {
        'epoch': epoch,
        'fundus_model': fundus_model.state_dict(),
        'cmr_encoder': cmr_encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'best_recall5': best_recall5,
        'best_val_loss': best_val_loss,
        'best_metric_value': best_metric_value,
        'metric_for_best': getattr(args, 'metric_for_best', 'R@5'),
        'no_improve': no_improve,
        'args': vars(args),
    }
    # 最新 checkpoint（用于断点续训）
    latest_path = os.path.join(args.output_dir, 'checkpoint_latest.pth')
    torch.save(ckpt, latest_path)

    # 最佳 checkpoint
    if is_best:
        best_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
        torch.save(ckpt, best_path)
        # 同时保存兼容格式
        fundus_model.save_encoder_ckpt(
            os.path.join(args.output_dir, 'contrast_pretrain_encoder_best.pth'))
        if getattr(args, 'skip_full_eval', False):
            print(f'[main] New best checkpoint saved (val_loss={best_val_loss:.6f})')
        else:
            mname = getattr(args, 'metric_for_best', 'R@5')
            bmv = best_metric_value
            print(f'[main] New best checkpoint saved ({mname}={bmv}, R@5={best_recall5:.4f})')

    # 定期存档
    if (epoch + 1) % args.save_freq == 0:
        archive_path = os.path.join(args.output_dir, f'checkpoint_ep{epoch:03d}.pth')
        torch.save(ckpt, archive_path)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.sgt_batch_stats_csv is None:
        args.sgt_batch_stats_csv = os.path.join(args.output_dir, 'sgt_batch_stats.csv')
    main(args)
