from __future__ import annotations

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
    """Encode one prespecified retinal photograph per participant."""
    eids, fundus, cmr, t1_available = [], [], [], []
    fundus_encoder.eval()
    cmr_projector.eval()
    for batch in loader:
        with autocast_context(device, amp_dtype):
            fundus_projection, _ = fundus_encoder(batch["image"].to(device))
            cmr_projection = F.normalize(
                cmr_projector(batch["cmr_embedding"].to(device)), dim=-1)
        eids.extend(int(value) for value in batch["eid"].tolist())
        fundus.append(fundus_projection.float().cpu())
        cmr.append(cmr_projection.float().cpu())
        t1_available.extend(bool(value) for value in batch["t1_available"])
    if len(set(eids)) != len(eids):
        raise RuntimeError(
            "Evaluation requires exactly one retinal photograph per participant.")
    return (
        np.asarray(eids, dtype=np.int64),
        F.normalize(torch.cat(fundus), dim=-1),
        F.normalize(torch.cat(cmr), dim=-1),
        np.asarray(t1_available, dtype=bool),
    )


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
