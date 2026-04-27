"""preprocess_cmr_lax_sax_fast.py -- same output as preprocess_cmr_lax_sax.py
but ~5-8x faster via in-memory zip reads (no tempdir extraction) and reading
pixel data only for the 4 frames we actually keep.

Same output schema: {out_dir}/{eid}.npy, shape (4, 224, 224) float16
  ch 0 = LAX_4Ch ED, ch 1 = LAX_4Ch ES, ch 2 = SAX_mid ED, ch 3 = SAX_mid ES

CLI identical to the original script:
  python preprocess_cmr_lax_sax_fast.py \
      --eid_file stage2_tier2_preprocess_todo.txt \
      --cmr_dir /data/home/shujia/UKB/CMRI/downloaded \
      --out_dir /data/home/shujia/UKB/CMRI/preprocessed_lax_sax \
      --num_workers 24
"""

from __future__ import annotations

import argparse
import io
import os
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pydicom
from PIL import Image


LAX_SERIES = "CINE_segmented_LAX_4Ch"
SAX_SERIES_PREFIX = "CINE_segmented_SAX_b"


# ---------- helpers -----------------------------------------------------------


def _read_dcm_header_from_bytes(data: bytes):
    """Parse DICOM header (no pixels) from an in-memory byte buffer.
    pydicom requires a seekable file-like, BytesIO gives us that.
    """
    return pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True, force=True)


def _read_dcm_full_from_bytes(data: bytes):
    return pydicom.dcmread(io.BytesIO(data), force=True)


def _pick_ed_es(metas):
    """metas is a list of dicts with InstanceNumber / TriggerTime / name.
    Returns (ed_name, es_name) chosen identically to the original script.
    """
    if not metas:
        return None, None
    if all(m["InstanceNumber"] != 9999 for m in metas):
        metas = sorted(metas, key=lambda m: m["InstanceNumber"])
    else:
        metas = sorted(metas, key=lambda m: m["TriggerTime"])
    n = len(metas)
    ed = metas[0]
    es = metas[n // 2] if n >= 2 else metas[0]
    return ed["name"], es["name"]


def _normalize_to_224(img: np.ndarray) -> np.ndarray:
    """2-98 percentile clip + minmax [0,1] + resize 224x224 via PIL bilinear."""
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo)
    img8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(img8, mode="L").resize((224, 224), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


# ---------- per-zip processors ------------------------------------------------


def _process_lax_zip_fast(zip_path: str):
    """Scan headers in memory, pick ED/ES of LAX_4Ch series, load those 2 pixels."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".dcm")]
        lax_metas = []
        raw_cache: dict[str, bytes] = {}
        for name in names:
            try:
                raw = zf.read(name)
            except Exception:
                continue
            try:
                d = _read_dcm_header_from_bytes(raw)
            except Exception:
                continue
            if getattr(d, "SeriesDescription", "") != LAX_SERIES:
                continue
            raw_cache[name] = raw
            lax_metas.append({
                "name": name,
                "InstanceNumber": getattr(d, "InstanceNumber", 9999),
                "TriggerTime": float(getattr(d, "TriggerTime", 9e9) or 9e9),
            })
        if not lax_metas:
            raise RuntimeError(f"no {LAX_SERIES} DICOMs in {zip_path}")
        ed_name, es_name = _pick_ed_es(lax_metas)
        ed = _read_dcm_full_from_bytes(raw_cache[ed_name]).pixel_array
        es = _read_dcm_full_from_bytes(raw_cache[es_name]).pixel_array
        return _normalize_to_224(ed), _normalize_to_224(es)


def _process_sax_zip_fast(zip_path: str):
    """Two-pass through SAX zip in memory:
      pass-1: read every .dcm header, group by SeriesDescription -> metas + SliceLocation
      pick mid series by SliceLocation (same rule as original)
      pass-2: for that series' files we've already cached bytes; read pixels for ED/ES
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".dcm")]
        series_metas: dict[str, list[dict]] = {}
        series_loc: dict[str, float] = {}
        raw_cache: dict[str, bytes] = {}
        for name in names:
            try:
                raw = zf.read(name)
            except Exception:
                continue
            try:
                d = _read_dcm_header_from_bytes(raw)
            except Exception:
                continue
            desc = getattr(d, "SeriesDescription", "")
            if not desc.startswith(SAX_SERIES_PREFIX):
                continue
            if desc == "CINE_segmented_SAX_InlineVF":
                continue
            # cache bytes of candidate series only (memory save)
            raw_cache[name] = raw
            series_metas.setdefault(desc, []).append({
                "name": name,
                "InstanceNumber": getattr(d, "InstanceNumber", 9999),
                "TriggerTime": float(getattr(d, "TriggerTime", 9e9) or 9e9),
            })
            if desc not in series_loc:
                series_loc[desc] = float(getattr(d, "SliceLocation", 0.0) or 0.0)

        if not series_metas:
            raise RuntimeError(f"no SAX cine series in {zip_path}")

        sorted_series = sorted(series_loc.keys(), key=lambda s: series_loc[s])
        mid_sd = sorted_series[len(sorted_series) // 2]
        ed_name, es_name = _pick_ed_es(series_metas[mid_sd])
        if ed_name is None:
            raise RuntimeError("could not pick ED/ES for SAX mid series")

        ed = _read_dcm_full_from_bytes(raw_cache[ed_name]).pixel_array
        es = _read_dcm_full_from_bytes(raw_cache[es_name]).pixel_array
        return _normalize_to_224(ed), _normalize_to_224(es), mid_sd


# ---------- per-EID driver ----------------------------------------------------


def process_one(eid: int, cmr_dir: str, out_dir: str) -> dict:
    result = {
        "eid": eid, "status": "unknown", "msg": "",
        "frame_shape": None, "sax_series_used": None, "out_path": None,
    }
    out_path = os.path.join(out_dir, f"{eid}.npy")
    if os.path.exists(out_path):
        result["status"] = "skip_exists"
        result["out_path"] = out_path
        return result

    lax_zip = os.path.join(cmr_dir, f"{eid}_20208_2_0.zip")
    sax_zip = os.path.join(cmr_dir, f"{eid}_20209_2_0.zip")
    if not (os.path.exists(lax_zip) and os.path.exists(sax_zip)):
        result["status"] = "missing_zip"
        result["msg"] = f"lax={os.path.exists(lax_zip)} sax={os.path.exists(sax_zip)}"
        return result

    try:
        lax_ed, lax_es = _process_lax_zip_fast(lax_zip)
        sax_ed, sax_es, sax_sd = _process_sax_zip_fast(sax_zip)
        stacked = np.stack([lax_ed, lax_es, sax_ed, sax_es], axis=0).astype(np.float16)
        np.save(out_path, stacked)
        result["status"] = "ok"
        result["frame_shape"] = stacked.shape
        result["sax_series_used"] = sax_sd
        result["out_path"] = out_path
    except Exception as e:
        result["status"] = "err"
        result["msg"] = f"{type(e).__name__}: {e}"
    return result


# ---------- main --------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eid_file", required=True, type=str)
    ap.add_argument("--cmr_dir", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--max_eids", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.eid_file) as f:
        eids = [int(l.strip()) for l in f if l.strip()]
    if args.max_eids > 0:
        eids = eids[: args.max_eids]
    print(f"[preprocess-fast] processing {len(eids)} EIDs with {args.num_workers} workers", flush=True)

    stats = {"ok": 0, "skip_exists": 0, "missing_zip": 0, "err": 0}
    err_log = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(process_one, e, args.cmr_dir, args.out_dir): e for e in eids}
        for i, fut in enumerate(as_completed(futures)):
            r = fut.result()
            stats[r["status"]] = stats.get(r["status"], 0) + 1
            if r["status"] == "err":
                err_log.append(f'EID {r["eid"]}: {r["msg"]}')
            if (i + 1) % 50 == 0 or i + 1 == len(eids):
                dt = time.time() - t0
                rate = (i + 1) / max(dt, 1e-6)
                eta = (len(eids) - i - 1) / max(rate, 1e-6)
                print(
                    f"  [{i+1}/{len(eids)}] stats={stats}  rate={rate:.2f} eids/s  eta={eta/60:.1f} min",
                    flush=True,
                )

    print("\n=== done ===")
    print(f"stats: {stats}")
    print(f"elapsed: {(time.time()-t0)/60:.2f} min")
    if err_log:
        print("first errors:\n  " + "\n  ".join(err_log[:10]))


if __name__ == "__main__":
    main()
