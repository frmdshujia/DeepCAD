"""
sanity_check.py  —  双塔框架 Pipeline 验证（第一层）

使用随机向量替代真实图像 embedding，验证：
  [1] forward pass 不报错
  [2] loss 数值合理（非 nan/inf）
  [3] 三种 mode（cmr_only / fundus_only / paired）都能运行
  [4] 训练集 loss 持续下降
  [5] 梯度无 nan/inf

运行：
  /data/home/shujia/miniconda3/bin/python sanity_check.py --n_epochs 20
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── §5.1  LoRA 层 ─────────────────────────────────────────────────────────────
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, r=16, alpha=32):
        super().__init__()
        self.lora_A = nn.Linear(in_dim, r, bias=False)
        self.lora_B = nn.Linear(r, out_dim, bias=False)
        self.scale = alpha / r
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scale


# ── §5.2  跨模态交互 ──────────────────────────────────────────────────────────
class CrossModalGating(nn.Module):
    def __init__(self, dim_f=1024, dim_c=768, hidden=512, num_heads=8, r=16):
        super().__init__()
        self.proj_f = nn.Linear(dim_f, hidden)
        self.proj_c = nn.Linear(dim_c, hidden)
        self.attn_f2c = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.attn_c2f = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.norm_f = nn.LayerNorm(hidden)
        self.norm_c = nn.LayerNorm(hidden)
        self.gate_f = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.gate_c = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.back_proj_f = nn.Linear(hidden, dim_f)
        self.back_proj_c = nn.Linear(hidden, dim_c)
        self.lora_f = LoRALayer(dim_f, dim_f, r=r)
        self.lora_c = LoRALayer(dim_c, dim_c, r=r)

    def forward(self, z_f, z_c):
        zf = self.proj_f(z_f).unsqueeze(1)
        zc = self.proj_c(z_c).unsqueeze(1)
        f_from_c, _ = self.attn_f2c(zf, zc, zc)
        f_from_c = self.norm_f(f_from_c.squeeze(1))
        delta_f = self.gate_f(f_from_c) * f_from_c
        c_from_f, _ = self.attn_c2f(zc, zf, zf)
        c_from_f = self.norm_c(c_from_f.squeeze(1))
        delta_c = self.gate_c(c_from_f) * c_from_f
        delta_f = self.back_proj_f(delta_f)
        delta_c = self.back_proj_c(delta_c)
        return z_f + self.lora_f(delta_f), z_c + self.lora_c(delta_c)


# ── §5.3  多任务头 ────────────────────────────────────────────────────────────
class TaskHead(nn.Module):
    def __init__(self, in_dim, n_cls=4, n_reg=3, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.cls_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 1))
            for _ in range(n_cls)
        ])
        self.reg_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 1))
            for _ in range(n_reg)
        ])

    def forward(self, z):
        z = self.dropout(z)
        cls_out = [h(z).squeeze(-1) for h in self.cls_heads]
        reg_out = [h(z).squeeze(-1) for h in self.reg_heads]
        return cls_out, reg_out


# ── §6  多任务 Loss ───────────────────────────────────────────────────────────
class MultiTaskLoss(nn.Module):
    def __init__(self, pos_weights=None, reg_norm=None):
        super().__init__()
        if pos_weights is None:
            pos_weights = [1.0, 1.0, 1.0, 1.0]
        if reg_norm is None:
            reg_norm = [
                {'mean': 59.6,  'std': 6.4},
                {'mean': 145.9, 'std': 34.1},
                {'mean': 59.6,  'std': 19.9},
            ]
        self.cls_criteria = nn.ModuleList([
            nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w]))
            for w in pos_weights
        ])
        self.reg_norm = reg_norm

    def cls_loss(self, preds, labels):
        loss = torch.tensor(0.0, device=preds[0].device)
        for i, (pred, crit) in enumerate(zip(preds, self.cls_criteria)):
            valid = labels[:, i] >= 0
            if valid.sum() == 0:
                continue
            crit_device = nn.BCEWithLogitsLoss(
                pos_weight=crit.pos_weight.to(preds[0].device)
            )
            loss = loss + crit_device(pred[valid], labels[valid, i])
        return loss

    def reg_loss(self, preds, targets, reg_mask=None):
        loss = torch.tensor(0.0, device=preds[0].device)
        for i, pred in enumerate(preds):
            mu    = self.reg_norm[i]['mean']
            sigma = self.reg_norm[i]['std']
            target_norm = (targets[:, i] - mu) / sigma
            valid = ~torch.isnan(targets[:, i])
            if reg_mask is not None:
                valid = valid & reg_mask[:, i]
            if valid.sum() == 0:
                continue
            loss = loss + F.mse_loss(pred[valid], target_norm[valid])
        return loss

    def align_loss(self, z_f, z_c):
        return 1.0 - (z_f * z_c).sum(dim=-1).mean()

    def compute(self, results, batch, lambdas):
        total = torch.tensor(0.0, device=list(results.values())[0][0].device
                             if isinstance(list(results.values())[0], list)
                             else list(results.values())[0].device)
        mode = batch['mode']
        tw   = batch.get('time_weight', 1.0)

        if mode == 'fundus_only':
            total = total + self.cls_loss(results['fundus_cls'], batch['cls_labels'])
            total = total + self.reg_loss(results['fundus_reg'], batch['reg_labels'],
                                          batch.get('reg_mask'))
        elif mode == 'cmr_only':
            total = total + self.cls_loss(results['cmr_cls'],  batch['cls_labels'])
            total = total + self.reg_loss(results['cmr_reg'],  batch['reg_labels'])
        elif mode == 'paired':
            total = total + self.cls_loss(results['fundus_cls_base'], batch['cls_labels'])
            total = total + self.cls_loss(results['cmr_cls_base'],    batch['cls_labels'])
            total = total + self.reg_loss(results['cmr_reg_base'],    batch['reg_labels'])
            total = total + self.reg_loss(results['fundus_reg_base'], batch['reg_labels'],
                                          batch.get('reg_mask'))
            w = 0.5 * tw
            total = total + w * self.cls_loss(results['fundus_cls_enriched'], batch['cls_labels'])
            total = total + w * self.cls_loss(results['cmr_cls_enriched'],    batch['cls_labels'])
            total = total + w * self.reg_loss(results['cmr_reg_enriched'],    batch['reg_labels'])
            total = total + w * self.reg_loss(results['fundus_reg_enriched'], batch['reg_labels'],
                                              batch.get('reg_mask'))
            total = total + lambdas['align'] * tw * \
                    self.align_loss(results['align_f'], results['align_c'])
        return total


# ── §12.2  简化模型（直接接受 embedding） ─────────────────────────────────────
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_modal  = CrossModalGating()
        self.head_fundus  = TaskHead(1024)
        self.head_cmr     = TaskHead(768)
        self.align_proj_f = nn.Linear(1024, 256)
        self.align_proj_c = nn.Linear(768, 256)

    def forward(self, batch):
        results = {}
        mode = batch['mode']
        if mode == 'fundus_only':
            z_f = batch['fundus']
            cls, reg = self.head_fundus(z_f)
            results.update({'fundus_cls': cls, 'fundus_reg': reg})
        elif mode == 'cmr_only':
            z_c = batch['cmr']
            cls, reg = self.head_cmr(z_c)
            results.update({'cmr_cls': cls, 'cmr_reg': reg})
        elif mode == 'paired':
            z_f, z_c = batch['fundus'], batch['cmr']
            cls_f0, reg_f0 = self.head_fundus(z_f)
            cls_c0, reg_c0 = self.head_cmr(z_c)
            z_f_en, z_c_en = self.cross_modal(z_f, z_c)
            cls_f1, reg_f1 = self.head_fundus(z_f_en)
            cls_c1, reg_c1 = self.head_cmr(z_c_en)
            results.update({
                'fundus_cls_base': cls_f0, 'fundus_reg_base': reg_f0,
                'cmr_cls_base':    cls_c0, 'cmr_reg_base':    reg_c0,
                'fundus_cls_enriched': cls_f1, 'fundus_reg_enriched': reg_f1,
                'cmr_cls_enriched':    cls_c1, 'cmr_reg_enriched':    reg_c1,
                'align_f': F.normalize(self.align_proj_f(z_f), dim=-1),
                'align_c': F.normalize(self.align_proj_c(z_c), dim=-1),
            })
        return results


def make_fake_batch(n, mode, device, dim_f=1024, dim_c=768):
    cls_labels = torch.randint(0, 2, (n, 4)).float()
    reg_labels = torch.randn(n, 3)
    reg_mask   = torch.ones(n, 3).bool()
    batch = {'mode': mode}
    if mode == 'fundus_only':
        batch.update({'fundus': torch.randn(n, dim_f), 'cls_labels': cls_labels,
                      'reg_labels': reg_labels, 'reg_mask': reg_mask})
    elif mode == 'cmr_only':
        batch.update({'cmr': torch.randn(n, dim_c), 'cls_labels': cls_labels,
                      'reg_labels': reg_labels})
    elif mode == 'paired':
        batch.update({'fundus': torch.randn(n, dim_f), 'cmr': torch.randn(n, dim_c),
                      'cls_labels': cls_labels, 'reg_labels': reg_labels,
                      'reg_mask': reg_mask, 'time_weight': 1.0})
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def run_sanity_check(n_epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print('=' * 60)
    print('PIPELINE SANITY CHECK 开始')
    print('=' * 60)

    model     = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = MultiTaskLoss()
    lambdas   = {'align': 0.1}

    all_passed   = True
    loss_history = []
    max_grad     = 0.0

    for epoch in range(n_epochs):
        epoch_losses = []
        for mode in ['fundus_only', 'cmr_only', 'paired']:
            n     = 16 if mode == 'paired' else 32
            batch = make_fake_batch(n, mode, device)

            optimizer.zero_grad()
            results = model(batch)
            loss    = criterion.compute(results, batch, lambdas)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f'[FAIL] Epoch {epoch} mode={mode}: loss={loss.item()} (nan/inf)')
                all_passed = False
                break

            loss.backward()

            max_grad = max(
                p.grad.abs().max().item()
                for p in model.parameters() if p.grad is not None
            )
            if np.isnan(max_grad) or np.isinf(max_grad):
                print(f'[FAIL] Epoch {epoch} mode={mode}: 梯度爆炸 max_grad={max_grad}')
                all_passed = False
                break

            optimizer.step()
            epoch_losses.append(loss.item())

        if not all_passed:
            break

        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f'Epoch {epoch:3d} | Loss: {avg_loss:.4f} | MaxGrad: {max_grad:.6f}')

    if all_passed and len(loss_history) >= 5:
        early_loss = np.mean(loss_history[:3])
        late_loss  = np.mean(loss_history[-3:])
        if late_loss >= early_loss:
            print(f'[WARN] Loss 未下降: early={early_loss:.4f}, late={late_loss:.4f}')
        else:
            print(f'[OK]  Loss 持续下降: {early_loss:.4f} -> {late_loss:.4f}')

    print('\n' + '=' * 60)
    if all_passed:
        print('[PASS] Pipeline 验证通过！')
        print('[PASS] 无 nan/inf/梯度爆炸')
        print('[NEXT] 可进入第二层：快速消融实验')
    else:
        print('[FAIL] Pipeline 验证失败，请修复后再继续')
    print('=' * 60)
    return all_passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=20)
    args = parser.parse_args()
    run_sanity_check(args.n_epochs)
