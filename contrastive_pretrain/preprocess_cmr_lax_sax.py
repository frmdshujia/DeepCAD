"""
preprocess_cmr_lax_sax.py

Stage 2 image-level v2 CMR 预处理：每人一个 (4, 224, 224) float16 npy，
通道顺序 [LAX_4Ch_ED, LAX_4Ch_ES, SAX_mid_ED, SAX_mid_ES]。

关键帧定义（每 cine series 50 帧）:
  - ED (End-Diastole)  = InstanceNumber==1 或 TriggerTime 最小的帧（心室最大容积）
  - ES (End-Systole)   ≈ 中间帧（第 N/2 帧，TriggerTime 居中，近似心室最小容积）

处理逻辑（每 EID）:
  1. 解压 {eid}_20208_2_0.zip (LAX) 到 tempdir
     - 找 SeriesDescription == 'CINE_segmented_LAX_4Ch' 的 DICOM（单切片 50 帧）
     - 选 ED (frame 1) 和 ES (frame N/2)
  2. 解压 {eid}_20209_2_0.zip (SAX) 到 tempdir
     - 找所有 CINE_segmented_SAX_b* series（多切片，每 series 50 帧）
     - 按 SliceLocation 排序，选中位 slice series (mid-ventricular)
     - 选 ED (frame 1) 和 ES (frame N/2)
  3. 所有帧 resize 224×224 + 2-98 percentile clip + minmax [0,1]
  4. Stack → (4, 224, 224) float16 npy

CLI:
  python preprocess_cmr_lax_sax.py \
    --eid_file contrastive_pretrain/stage2_tier0_eids.txt \
    --cmr_dir /data/home/shujia/UKB/CMRI/downloaded \
    --out_dir /data/home/shujia/UKB/CMRI/preprocessed_lax_sax \
    --num_workers 4
"""

import argparse
import os
import shutil
import sys
import tempfile
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


LAX_SERIES = 'CINE_segmented_LAX_4Ch'
SAX_SERIES_PREFIX = 'CINE_segmented_SAX_b'  # e.g. b1, b2, ..., b10


def _select_ed_es_frames(dcm_files):
    """从一个 cine series 的 DICOM 文件列表里选 ED 和 ES 两帧。
    策略：
      - 按 InstanceNumber 排序（或 TriggerTime 排序，作为 fallback）
      - ED = 第 1 帧（TriggerTime 最小 / 心室最大容积）
      - ES ≈ 第 N/2 帧（TriggerTime 居中 / 近似心室最小容积）
    返回 (ed_pixel_array, es_pixel_array)。
    """
    metas = []
    for f in dcm_files:
        try:
            d = pydicom.dcmread(f, stop_before_pixels=True)
            metas.append({
                'file': f,
                'InstanceNumber': getattr(d, 'InstanceNumber', 9999),
                'TriggerTime': float(getattr(d, 'TriggerTime', 9e9) or 9e9),
            })
        except Exception:
            continue
    if not metas:
        return None, None
    # 优先按 InstanceNumber 排序，fallback TriggerTime
    if all(m['InstanceNumber'] != 9999 for m in metas):
        metas.sort(key=lambda m: m['InstanceNumber'])
    else:
        metas.sort(key=lambda m: m['TriggerTime'])
    n = len(metas)
    ed_meta = metas[0]
    es_meta = metas[n // 2] if n >= 2 else metas[0]
    ed_arr = pydicom.dcmread(ed_meta['file']).pixel_array
    es_arr = pydicom.dcmread(es_meta['file']).pixel_array
    return ed_arr, es_arr


def _normalize_to_224(img: np.ndarray) -> np.ndarray:
    """2-98 percentile clip + minmax [0,1] + resize 224x224 via PIL bilinear."""
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo)  # [0,1]
    img8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(img8, mode='L')
    pil = pil.resize((224, 224), Image.BILINEAR)
    out = np.array(pil, dtype=np.float32) / 255.0  # [0,1] 224x224
    return out


def _process_lax(zip_path: str, workdir: str):
    """Return (lax_ed_224, lax_es_224)."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(workdir)
    dcm_files = []
    for f in os.listdir(workdir):
        p = os.path.join(workdir, f)
        if not f.endswith('.dcm'):
            continue
        try:
            d = pydicom.dcmread(p, stop_before_pixels=True)
            if getattr(d, 'SeriesDescription', '') == LAX_SERIES:
                dcm_files.append(p)
        except Exception:
            continue
    if not dcm_files:
        raise RuntimeError(f'no {LAX_SERIES} DICOMs in {zip_path}')
    ed_arr, es_arr = _select_ed_es_frames(dcm_files)
    if ed_arr is None:
        raise RuntimeError(f'could not read any {LAX_SERIES} pixel data')
    return _normalize_to_224(ed_arr), _normalize_to_224(es_arr)


def _process_sax(zip_path: str, workdir: str):
    """Return (sax_ed_224, sax_es_224, mid_series_desc)."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(workdir)
    series_map = {}
    for f in os.listdir(workdir):
        p = os.path.join(workdir, f)
        if not f.endswith('.dcm'):
            continue
        try:
            d = pydicom.dcmread(p, stop_before_pixels=True)
            desc = getattr(d, 'SeriesDescription', '')
            if desc.startswith(SAX_SERIES_PREFIX) and desc != 'CINE_segmented_SAX_InlineVF':
                series_map.setdefault(desc, []).append(p)
        except Exception:
            continue
    if not series_map:
        raise RuntimeError(f'no SAX cine series in {zip_path}')
    series_locs = {}
    for sd, fs in series_map.items():
        try:
            d0 = pydicom.dcmread(fs[0], stop_before_pixels=True)
            series_locs[sd] = float(getattr(d0, 'SliceLocation', 0.0) or 0.0)
        except Exception:
            series_locs[sd] = 0.0
    sorted_series = sorted(series_locs.keys(), key=lambda s: series_locs[s])
    mid_idx = len(sorted_series) // 2
    mid_sd = sorted_series[mid_idx]
    ed_arr, es_arr = _select_ed_es_frames(series_map[mid_sd])
    if ed_arr is None:
        raise RuntimeError(f'could not read any SAX mid-slice pixel data')
    return _normalize_to_224(ed_arr), _normalize_to_224(es_arr), mid_sd


def process_one(eid: int, cmr_dir: str, out_dir: str) -> dict:
    """返回 dict: {eid, status, msg, frame_shapes, sax_series_used, out_path}
    Output npy shape = (4, 224, 224) float16:
      channel 0 = LAX_4Ch ED, 1 = LAX_4Ch ES, 2 = SAX_mid ED, 3 = SAX_mid ES
    """
    result = {'eid': eid, 'status': 'unknown', 'msg': '',
              'frame_shape': None, 'sax_series_used': None, 'out_path': None}
    out_path = os.path.join(out_dir, f'{eid}.npy')
    if os.path.exists(out_path):
        result['status'] = 'skip_exists'
        result['out_path'] = out_path
        return result

    lax_zip = os.path.join(cmr_dir, f'{eid}_20208_2_0.zip')
    sax_zip = os.path.join(cmr_dir, f'{eid}_20209_2_0.zip')
    if not (os.path.exists(lax_zip) and os.path.exists(sax_zip)):
        result['status'] = 'missing_zip'
        result['msg'] = f'lax={os.path.exists(lax_zip)} sax={os.path.exists(sax_zip)}'
        return result

    tmp = tempfile.mkdtemp(prefix=f'cmr_{eid}_')
    try:
        lax_dir = os.path.join(tmp, 'lax')
        sax_dir = os.path.join(tmp, 'sax')
        os.makedirs(lax_dir); os.makedirs(sax_dir)
        lax_ed, lax_es = _process_lax(lax_zip, lax_dir)
        sax_ed, sax_es, sax_sd = _process_sax(sax_zip, sax_dir)
        stacked = np.stack([lax_ed, lax_es, sax_ed, sax_es], axis=0).astype(np.float16)
        np.save(out_path, stacked)
        result['status'] = 'ok'
        result['frame_shape'] = stacked.shape
        result['sax_series_used'] = sax_sd
        result['out_path'] = out_path
    except Exception as e:
        result['status'] = 'err'
        result['msg'] = f'{type(e).__name__}: {e}'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eid_file', required=True, type=str,
                    help='Text file of EIDs, one per line.')
    ap.add_argument('--cmr_dir', required=True, type=str,
                    help='Directory with {eid}_20208/20209_2_0.zip files.')
    ap.add_argument('--out_dir', required=True, type=str,
                    help='Output directory for {eid}.npy files.')
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--max_eids', type=int, default=0,
                    help='For quick debug: limit to N EIDs (0=all).')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.eid_file) as f:
        eids = [int(l.strip()) for l in f if l.strip()]
    if args.max_eids > 0:
        eids = eids[:args.max_eids]
    print(f'[preprocess] processing {len(eids)} EIDs with {args.num_workers} workers')

    stats = {'ok': 0, 'skip_exists': 0, 'missing_zip': 0, 'err': 0}
    err_log = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(process_one, e, args.cmr_dir, args.out_dir): e
                   for e in eids}
        for i, fut in enumerate(as_completed(futures)):
            r = fut.result()
            stats[r['status']] = stats.get(r['status'], 0) + 1
            if r['status'] == 'err':
                err_log.append(f'EID {r["eid"]}: {r["msg"]}')
            if (i + 1) % 10 == 0 or i + 1 == len(eids):
                print(f'  [{i+1}/{len(eids)}] stats={stats}')
    print(f'\n=== done ===')
    print(f'stats: {stats}')
    if err_log:
        print(f'first errors:\n  ' + '\n  '.join(err_log[:10]))


if __name__ == '__main__':
    main()
