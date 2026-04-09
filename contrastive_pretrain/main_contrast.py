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

# timm 0.3.2 兼容补丁
import collections.abc
if 'torch._six' not in sys.modules:
    class _TorchSix:
        container_abcs = collections.abc
        inf = float('inf')
    sys.modules['torch._six'] = _TorchSix()

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

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

    # ── 模型 ──
    parser.add_argument('--finetune', required=True, type=str,
                        help='RETFound 预训练权重路径（.pth）')
    parser.add_argument('--proj_dim', default=256, type=int,
                        help='对比空间维度 d（128 或 256）')
    parser.add_argument('--drop_path', default=0.1, type=float,
                        help='ViT drop path rate')

    # ── 损失 ──
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='InfoNCE 温度参数')
    parser.add_argument('--cmr_sample_k', default=4096, type=int,
                        help='每 batch 从 CMR bank 随机采样的 CMR 数量')

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
    parser.add_argument('--proj_lr_scale', default=10.0, type=float,
                        help='Projection head LR = blr × proj_lr_scale（默认 1e-4）')
    parser.add_argument('--cmr_lr_scale', default=100.0, type=float,
                        help='CMR MLP LR = blr × cmr_lr_scale（默认 1e-3）')
    parser.add_argument('--clip_grad', default=1.0, type=float,
                        help='梯度裁剪范数（None 表示不裁剪）')

    # ── 早停 ──
    parser.add_argument('--patience', default=12, type=int,
                        help='val Recall@5 无提升时的早停 patience（epochs）')
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='每隔多少 epoch 计算一次完整 retrieval recall')

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

    # ── 数据集 ──
    dataset_train = FundusContrastDataset(args.fundus_csv, pc_cols, split='train')
    dataset_val   = FundusContrastDataset(args.fundus_csv, pc_cols, split='val')

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

    # ── CMR Bank（全量 PC scores 加载到 GPU）──
    cmr_bank = CMRBank(args.cmr_csv, pc_cols, split='train', device=str(device))
    cmr_bank_val = CMRBank(args.cmr_csv, pc_cols, split='val', device=str(device))

    # ── 模型 ──
    fundus_model = FundusContrastModel(proj_dim=args.proj_dim, drop_path_rate=args.drop_path)
    fundus_model.load_pretrained(args.finetune)
    fundus_model.to(device)

    cmr_encoder = CMREncoder(in_dim=args.n_pc, hidden_dim=128, out_dim=args.proj_dim)
    cmr_encoder.to(device)

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
    no_improve = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        fundus_model_without_ddp.load_state_dict(ckpt['fundus_model'])
        cmr_encoder_without_ddp.load_state_dict(ckpt['cmr_encoder'])
        optimizer.load_state_dict(ckpt['optimizer'])
        loss_scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_recall5 = ckpt.get('best_recall5', 0.0)
        no_improve = ckpt.get('no_improve', 0)
        print(f'[main] Resumed from epoch {start_epoch}, best_recall5={best_recall5:.4f}')

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

        # ── 完整 retrieval 评估（每 eval_freq 轮）──
        # broadcast 必须所有 rank 同时执行（避免死锁），故先算出 recall5 再广播
        recall5 = best_recall5  # 默认：本轮未评估时沿用上轮最优
        do_eval = (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1
        if do_eval:
            if global_rank == 0:
                # eval_loader_val 使用 SequentialSampler，覆盖全量 val 数据
                eval_metrics = run_full_eval(
                    fundus_model_without_ddp, cmr_encoder_without_ddp,
                    eval_loader_val, cmr_bank_val, device=str(device),
                )
                recall5 = eval_metrics.get('R@5', 0.0)
                print(f'[Eval epoch {epoch}] {eval_metrics}')

                if log_writer is not None:
                    for k, v in eval_metrics.items():
                        log_writer.add_scalar(f'eval/{k}', v, epoch)

            # 多卡时广播 recall5（所有 rank 必须同时执行此操作）
            if args.distributed:
                recall5_t = torch.tensor(recall5, device=device)
                torch.distributed.broadcast(recall5_t, src=0)
                recall5 = recall5_t.item()

        # ── 早停 & 保存最佳 checkpoint ──
        is_best = recall5 > best_recall5
        if is_best:
            best_recall5 = recall5
            no_improve = 0
        else:
            no_improve += 1

        if global_rank == 0:
            _save_checkpoint(
                args, epoch, fundus_model_without_ddp, cmr_encoder_without_ddp,
                optimizer, loss_scaler, best_recall5, no_improve,
                is_best=is_best,
            )

        # ── 日志 ──
        log_stats = {
            'epoch': epoch,
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
            'recall5': recall5,
            'best_recall5': best_recall5,
            'no_improve': no_improve,
        }
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
    print(f'[main] Best val Recall@5 = {best_recall5:.4f}')

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
                     optimizer, scaler, best_recall5, no_improve, is_best=False):
    ckpt = {
        'epoch': epoch,
        'fundus_model': fundus_model.state_dict(),
        'cmr_encoder': cmr_encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'best_recall5': best_recall5,
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
        print(f'[main] New best checkpoint saved (Recall@5={best_recall5:.4f})')

    # 定期存档
    if (epoch + 1) % args.save_freq == 0:
        archive_path = os.path.join(args.output_dir, f'checkpoint_ep{epoch:03d}.pth')
        torch.save(ckpt, archive_path)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
