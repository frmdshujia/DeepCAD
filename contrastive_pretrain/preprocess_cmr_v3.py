"""
preprocess_cmr_v3.py  —  生成 (16, 224, 224) float16 npy

帧布局 (16 帧):
  [0]     LAX 2Ch,  ED
  [1]     LAX 2Ch,  mid-sys (N//4)
  [2]     LAX 2Ch,  ES      (N//2)
  [3]     LAX 4Ch,  ED
  [4]     LAX 4Ch,  mid-sys (N//4)
  [5]     LAX 4Ch,  ES      (N//2)
  [6-8]   SAX base (~20% 位置): ED, mid-sys, ES
  [9-11]  SAX mid  (~50% 位置): ED, mid-sys, ES
  [12-14] SAX apex (~80% 位置): ED, mid-sys, ES
  [15]    T1 map (field 20214, ShMOLLI T1 Map; 缺失时填零)

元数据 JSON (同名 .json):
  {t1_valid: bool, sax_n_slices: int, sax_used: [b_idx, m_idx, a_idx]}

CLI:
  python preprocess_cmr_v3.py \
      --eid_file contrastive_pretrain/task_reports/task1_cmr_train.csv \
      --cmr_dir  /data/home/shujia/UKB/CMRI/downloaded \
      --out_dir  /data/home/shujia/UKB/CMRI/preprocessed_cmr_v3 \
      --num_workers 8
"""
from __future__ import annotations
import argparse, io, json, os, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


# ── 常量 ──────────────────────────────────────────────────────────────────────
LAX_2CH = "CINE_segmented_LAX_2Ch"
LAX_4CH = "CINE_segmented_LAX_4Ch"
SAX_PREFIX = "CINE_segmented_SAX_b"
SAX_INLINE = "CINE_segmented_SAX_InlineVF"
T1_SERIES  = "ShMOLLI_192i_SAX_b2s_SAX_b2s_SAX_b2s_T1MAP"
T1_COMMENT = "T1 Map"   # ImageComments 含此字符串的帧

NUM_FRAMES = 16
T1_CLIP    = (400.0, 2000.0)   # ms，正常心肌 800-1200，安全裕量
ROI_SIZE   = 160               # crop size before resize to 224 (~75% of 208px FOV)


# ── 工具函数 ───────────────────────────────────────────────────────────────────
def _hdr(data: bytes):
    return pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True, force=True)

def _full(data: bytes):
    return pydicom.dcmread(io.BytesIO(data), force=True)

def _heart_roi(img: np.ndarray, crop: int = ROI_SIZE) -> tuple[int, int, int, int]:
    """
    Detect cardiac ROI from a single SSFP frame using bright blood pool centroid.
    Returns (y0, y1, x0, x1) crop coordinates, consistent crop size = crop×crop.
    Falls back to image centre if detection is uncertain.
    Blood is the brightest structure in balanced-SSFP (cine MRI).
    """
    H, W = img.shape
    # restrict to central 80% to avoid scanner labels / phase-wrap ghosts
    margin_y, margin_x = H // 10, W // 10
    roi_img = img[margin_y:H - margin_y, margin_x:W - margin_x].astype(np.float32)
    thr = np.percentile(roi_img, 82)
    mask = roi_img > thr
    ys, xs = np.where(mask)
    if len(ys) < 50:            # fallback: image too dark or uniform
        cy, cx = H // 2, W // 2
    else:
        cy = int(ys.mean()) + margin_y
        cx = int(xs.mean()) + margin_x
    half = crop // 2
    y0 = max(0, min(cy - half, H - crop))
    x0 = max(0, min(cx - half, W - crop))
    return y0, y0 + crop, x0, x0 + crop

def _to224(img: np.ndarray, roi: tuple | None = None,
           t1_mode: bool = False) -> np.ndarray:
    """
    Crop to cardiac ROI (if given), normalise, resize to 224×224 float32 [0,1].
    roi: (y0, y1, x0, x1) — consistent across all frames of the same series.
    """
    img = img.astype(np.float32)
    if roi is not None:
        y0, y1, x0, x1 = roi
        img = img[y0:y1, x0:x1]
    if t1_mode:
        img = np.clip(img, *T1_CLIP)
        img = (img - T1_CLIP[0]) / (T1_CLIP[1] - T1_CLIP[0])
    else:
        lo, hi = np.percentile(img, [2, 98])
        if hi <= lo: hi = lo + 1.0
        img = np.clip(img, lo, hi)
        img = (img - lo) / (hi - lo)
    pil = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8), mode='L')
    return np.array(pil.resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0

def _sort_frames(metas: list[dict]) -> list[dict]:
    """按 InstanceNumber 或 TriggerTime 排序。"""
    if all(m['inst'] != 9999 for m in metas):
        return sorted(metas, key=lambda m: m['inst'])
    return sorted(metas, key=lambda m: m['tt'])

def _pick_timepoints(metas: list[dict], fracs=(0.0, 0.25, 0.5)) -> list[str]:
    """从已排序帧列表里按比例取帧，返回 zip 内文件名列表。"""
    metas = _sort_frames(metas)
    n = len(metas)
    return [metas[min(round(f * n), n - 1)]['name'] for f in fracs]


# ── LAX 处理 ───────────────────────────────────────────────────────────────────
def _process_lax(zip_path: str) -> dict[str, np.ndarray]:
    """返回 {LAX_2Ch: [ed, ms, es], LAX_4Ch: [ed, ms, es]}，帧为 224×224 float32。"""
    series: dict[str, list[dict]] = {}
    raw: dict[str, bytes] = {}

    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith('.dcm')]
        for name in names:
            data = z.read(name)
            raw[name] = data
            try:
                d = _hdr(data)
                desc = str(getattr(d, 'SeriesDescription', ''))
                if desc not in (LAX_2CH, LAX_4CH):
                    continue
                series.setdefault(desc, []).append({
                    'name': name,
                    'inst': int(getattr(d, 'InstanceNumber', 9999) or 9999),
                    'tt':   float(getattr(d, 'TriggerTime', 9e9) or 9e9),
                })
            except Exception:
                pass

    result = {}
    for desc, metas in series.items():
        names_sel = _pick_timepoints(metas, fracs=(0.0, 0.25, 0.5))
        # decode all 3 frames first (ED, mid-sys, ES)
        arrs = [_full(raw[nm]).pixel_array for nm in names_sel]
        # ROI derived from ED frame → applied consistently to all 3 timepoints
        roi = _heart_roi(arrs[0])
        result[desc] = [_to224(a, roi=roi) for a in arrs]
    return result


# ── SAX 处理 ───────────────────────────────────────────────────────────────────
def _process_sax(zip_path: str) -> tuple[list[list[np.ndarray]], dict]:
    """返回 (slices_frames, meta)
    slices_frames: list[3] of list[3] → [base, mid, apex] × [ed, ms, es]
    meta: {n_slices, used_indices}
    """
    series_metas: dict[str, list[dict]] = {}
    series_locs:  dict[str, float]      = {}
    raw: dict[str, bytes] = {}

    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith('.dcm')]
        for name in names:
            data = z.read(name)
            raw[name] = data
            try:
                d = _hdr(data)
                desc = str(getattr(d, 'SeriesDescription', ''))
                if not desc.startswith(SAX_PREFIX) or desc == SAX_INLINE:
                    continue
                series_metas.setdefault(desc, []).append({
                    'name': name,
                    'inst': int(getattr(d, 'InstanceNumber', 9999) or 9999),
                    'tt':   float(getattr(d, 'TriggerTime', 9e9) or 9e9),
                })
                if desc not in series_locs:
                    series_locs[desc] = float(getattr(d, 'SliceLocation', 0.0) or 0.0)
            except Exception:
                pass

    # 按 SliceLocation 从基底到心尖排序（升序 = 从底部切，降序情况少见，取绝对值保持相对顺序）
    sorted_descs = sorted(series_locs, key=lambda s: series_locs[s])
    n = len(sorted_descs)
    if n == 0:
        raise RuntimeError("no SAX series found")

    # 取 20%, 50%, 80% 位置（基底/中段/心尖）
    idxs = [min(round(n * f), n - 1) for f in (0.2, 0.5, 0.8)]
    # 避免重复
    if idxs[0] == idxs[1]: idxs[1] = min(idxs[0] + 1, n - 1)
    if idxs[1] == idxs[2]: idxs[2] = min(idxs[1] + 1, n - 1)

    slices_frames = []
    for idx in idxs:
        desc = sorted_descs[idx]
        metas = series_metas[desc]
        names_sel = _pick_timepoints(metas, fracs=(0.0, 0.25, 0.5))
        arrs = [_full(raw[nm]).pixel_array for nm in names_sel]
        # ROI from ED frame, applied consistently to mid-sys and ES
        roi = _heart_roi(arrs[0])
        frames = [_to224(a, roi=roi) for a in arrs]
        slices_frames.append(frames)

    meta = {'n_slices': n, 'used_indices': idxs,
            'used_descs': [sorted_descs[i] for i in idxs]}
    return slices_frames, meta


# ── T1 map 处理 ────────────────────────────────────────────────────────────────
def _process_t1(zip_path: str) -> np.ndarray | None:
    """返回 224×224 float32 T1 归一化图；失败返回 None。"""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith('.dcm')]
        for name in names:
            data = z.read(name)
            try:
                dh = _hdr(data)
                if str(getattr(dh, 'SeriesDescription', '')) != T1_SERIES:
                    continue
                comments = str(getattr(dh, 'ImageComments', ''))
                if T1_COMMENT not in comments:
                    continue
                d = _full(data)
                arr = d.pixel_array.astype(np.float32)
                slope = float(getattr(d, 'RescaleSlope', 1) or 1)
                intercept = float(getattr(d, 'RescaleIntercept', 0) or 0)
                arr = arr * slope + intercept
                # T1 map: use clip-normalised image to find heart ROI
                # (blood pool has intermediate T1 ~1500ms → not the brightest,
                #  but myocardium 1000ms and background form a clear structure)
                arr_vis = np.clip(arr, *T1_CLIP)
                roi = _heart_roi(arr_vis)
                return _to224(arr, roi=roi, t1_mode=True)
            except Exception:
                pass
    return None


# ── 主流程（每 EID）────────────────────────────────────────────────────────────
def process_one(eid: int, cmr_dir: str, out_dir: str) -> dict:
    out_npy  = os.path.join(out_dir, f'{eid}.npy')
    out_json = os.path.join(out_dir, f'{eid}.json')
    if os.path.exists(out_npy) and os.path.exists(out_json):
        return {'eid': eid, 'status': 'skip'}

    lax_zip = os.path.join(cmr_dir, f'{eid}_20208_2_0.zip')
    sax_zip = os.path.join(cmr_dir, f'{eid}_20209_2_0.zip')
    t1_zip  = os.path.join(cmr_dir, f'{eid}_20214_2_0.zip')

    if not (os.path.exists(lax_zip) and os.path.exists(sax_zip)):
        return {'eid': eid, 'status': 'missing_lax_sax'}

    try:
        lax_data  = _process_lax(lax_zip)
        sax_data, sax_meta = _process_sax(sax_zip)
        t1_arr = _process_t1(t1_zip) if os.path.exists(t1_zip) else None

        # LAX frames: 2Ch×3 + 4Ch×3 = 6 frames
        # _process_lax() already extracts [ED, mid-sys, ES] for each series
        lax2ch = lax_data.get(LAX_2CH)
        lax4ch = lax_data.get(LAX_4CH)
        if lax4ch is None:
            return {'eid': eid, 'status': 'missing_lax4ch'}
        # LAX 2Ch: fallback to 4Ch if 2Ch absent (all 3 timepoints)
        lax2ch_frames = lax2ch if lax2ch else lax4ch   # [ED, mid-sys, ES]
        lax4ch_frames = lax4ch                          # [ED, mid-sys, ES]

        # SAX: [base, mid, apex] × [ed, ms, es] = 9 frames
        sax_frames = [f for level in sax_data for f in level]

        # T1 map (static, 1 frame)
        t1_frame = t1_arr if t1_arr is not None else np.zeros((224, 224), np.float32)
        t1_valid = t1_arr is not None

        # Stack: (16, 224, 224)
        # [0-2] LAX 2Ch ED/mid/ES, [3-5] LAX 4Ch ED/mid/ES,
        # [6-14] SAX base/mid/apex × ED/mid/ES, [15] T1 map
        stack = np.stack(
            lax2ch_frames + lax4ch_frames + sax_frames + [t1_frame],
            axis=0
        ).astype(np.float16)

        np.save(out_npy, stack)
        with open(out_json, 'w') as f:
            json.dump({'t1_valid': t1_valid, **sax_meta}, f)

        return {'eid': eid, 'status': 'ok', 't1_valid': t1_valid}

    except Exception as e:
        return {'eid': eid, 'status': 'err', 'msg': f'{type(e).__name__}: {e}'}


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eid_file',    required=True)
    ap.add_argument('--cmr_dir',     required=True)
    ap.add_argument('--out_dir',     required=True)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--max_eids',    type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 支持 CSV（含 eid 列）或纯文本 EID 列表
    src = Path(args.eid_file)
    if src.suffix == '.csv':
        import pandas as pd
        eids = pd.read_csv(src)['eid'].astype(int).unique().tolist()
    else:
        eids = [int(l.strip()) for l in src.read_text().splitlines() if l.strip()]

    if args.max_eids > 0:
        eids = eids[:args.max_eids]
    print(f'[preprocess_v3] {len(eids)} EIDs  workers={args.num_workers}')

    stats: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {ex.submit(process_one, e, args.cmr_dir, args.out_dir): e for e in eids}
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            stats[r['status']] = stats.get(r['status'], 0) + 1
            if r['status'] == 'err':
                print(f"  ERR eid={r['eid']}: {r.get('msg','')}")
            if (i + 1) % 500 == 0 or i + 1 == len(eids):
                print(f'  [{i+1}/{len(eids)}] {stats}')

    print(f'\n=== done === {stats}')


if __name__ == '__main__':
    main()
