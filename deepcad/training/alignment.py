from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from deepcad.losses import symmetric_info_nce
from deepcad.training.common import autocast_context


@torch.no_grad()
def collect_participant_embeddings(
    fundus_encoder,
    cmr_projector,
    loader,
    device: torch.device,
    amp_dtype: str = "bfloat16",
):
    """Encode every eye and average retinal projections within participant."""
    fundus_sums = defaultdict(lambda: None)
    counts = defaultdict(int)
    cmr_by_eid = {}
    t1_by_eid = {}
    fundus_encoder.eval()
    cmr_projector.eval()
    for batch in loader:
        with autocast_context(device, amp_dtype):
            fundus_projection, _ = fundus_encoder(batch["image"].to(device))
            cmr_projection = F.normalize(
                cmr_projector(batch["cmr_embedding"].to(device)), dim=-1)
        fundus_projection = fundus_projection.float().cpu()
        cmr_projection = cmr_projection.float().cpu()
        for index, eid_value in enumerate(batch["eid"].tolist()):
            eid = int(eid_value)
            current = fundus_projection[index]
            fundus_sums[eid] = (
                current.clone() if fundus_sums[eid] is None
                else fundus_sums[eid] + current)
            counts[eid] += 1
            cmr_by_eid.setdefault(eid, cmr_projection[index].clone())
            t1_by_eid[eid] = bool(batch["t1_available"][index])
    eids = np.asarray(sorted(fundus_sums), dtype=np.int64)
    fundus = torch.stack([
        fundus_sums[int(eid)] / counts[int(eid)] for eid in eids])
    fundus = F.normalize(fundus, dim=-1)
    cmr = F.normalize(torch.stack([cmr_by_eid[int(eid)] for eid in eids]), dim=-1)
    t1_available = np.asarray([t1_by_eid[int(eid)] for eid in eids], dtype=bool)
    eye_counts = np.asarray([counts[int(eid)] for eid in eids], dtype=np.int64)
    return eids, fundus, cmr, t1_available, eye_counts


def alignment_metrics(
    fundus: torch.Tensor,
    cmr: torch.Tensor,
    logit_scale: torch.Tensor,
) -> dict[str, float]:
    if len(fundus) == 0:
        return {"n": 0, "infonce": float("nan")}
    scale = logit_scale.detach().float().exp().clamp(max=100.0).cpu()
    temperature = scale.reciprocal()
    loss = symmetric_info_nce(fundus, cmr, temperature)
    similarity = fundus @ cmr.T
    labels = torch.arange(len(fundus))
    k = min(5, len(fundus))
    f2c = similarity.topk(k, dim=1).indices
    c2f = similarity.T.topk(k, dim=1).indices
    return {
        "n": int(len(fundus)),
        "infonce": float(loss),
        "fundus_to_cmr_top1": float((f2c[:, 0] == labels).float().mean()),
        "fundus_to_cmr_top5": float(
            (f2c == labels[:, None]).any(dim=1).float().mean()),
        "cmr_to_fundus_top1": float((c2f[:, 0] == labels).float().mean()),
        "cmr_to_fundus_top5": float(
            (c2f == labels[:, None]).any(dim=1).float().mean()),
    }
