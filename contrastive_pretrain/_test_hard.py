"""单步 hard InfoNCE 测试（CUDA_LAUNCH_BLOCKING=1 下定位 index 错误）"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # 必须先 import torch
import collections.abc
if 'torch._six' not in sys.modules:
    class _TorchSix:
        container_abcs = collections.abc
        inf = float('inf')
    sys.modules['torch._six'] = _TorchSix()

import torch.nn.functional as F

from contrastive_pretrain.datasets_contrast import FundusContrastDataset, CMRBank
from contrastive_pretrain.models_contrast import FundusContrastModel, CMREncoder
from contrastive_pretrain.loss_contrast import hard_infonce_loss

pc_cols = ['M1_PC1','M1_PC2','M2_PC1','M2_PC2','M2_PC3',
           'M3_PC1','M3_PC2','M4_PC1','M4_PC2','M5_PC1',
           'M5_PC2','M6_PC1','M6_PC2','M6_PC3']

ds = FundusContrastDataset(
    f'{ROOT}/contrastive_pretrain/preprocessed_data/fundus_table.csv',
    pc_cols, split='train', train_max_samples=64, subset_seed=42)
loader = torch.utils.data.DataLoader(
    ds, batch_size=16, shuffle=True, num_workers=0, drop_last=True)

cmr_bank = CMRBank(
    f'{ROOT}/contrastive_pretrain/preprocessed_data/cmr_table.csv',
    pc_cols, split='train', device='cuda', max_rows=5000)

images, eids, pc_fundus = next(iter(loader))
eids_list = list(eids)
_, pc_cmr = cmr_bank.sample_with_batch_positives(eids_list, 1024)
print('pc_cmr shape:', pc_cmr.shape)
print('B:', len(eids_list), 'K:', pc_cmr.shape[0])

fundus_model = FundusContrastModel(proj_dim=256, drop_path_rate=0.1)
fundus_model.load_pretrained(f'{ROOT}/RETFound_cfp_weights.pth')
fundus_model.cuda().train()

cmr_enc = CMREncoder(in_dim=14, hidden_dim=128, out_dim=256)
cmr_enc.cuda().train()

images = images.cuda()
pc_cmr_dev = pc_cmr.cuda()

with torch.cuda.amp.autocast():
    z_f = fundus_model(images)
    z_c = cmr_enc(pc_cmr_dev)
    print('z_f shape:', z_f.shape, 'z_c shape:', z_c.shape)
    B = z_f.size(0)
    K = z_c.size(0)
    logits = z_f @ z_c.T / 0.07
    labels = torch.arange(B, device=z_f.device)
    print(f'logits: {logits.shape}, labels max={labels.max().item()}, K={K}')
    loss = F.cross_entropy(logits, labels)
    print('loss:', loss.item())

print('Forward OK')

loss.backward()
print('Backward OK')
