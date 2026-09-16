import torch
import torch.nn as nn

from deepcad.training.alignment import (
    alignment_metrics, collect_participant_embeddings,
)


class FakeFundusEncoder(nn.Module):
    def forward(self, images):
        projection = torch.nn.functional.normalize(images, dim=-1)
        return projection, images


def test_eye_embeddings_are_averaged_within_eid():
    batch = {
        "eid": torch.tensor([10, 10, 20]),
        "image": torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        "cmr_embedding": torch.tensor(
            [[1.0, 1.0], [1.0, 1.0], [0.0, 1.0]]),
        "t1_available": torch.tensor([True, True, False]),
    }
    eids, fundus, cmr, t1, eye_counts = collect_participant_embeddings(
        FakeFundusEncoder(), nn.Identity(), [batch], torch.device("cpu"), "none")
    assert eids.tolist() == [10, 20]
    assert eye_counts.tolist() == [2, 1]
    assert t1.tolist() == [True, False]
    assert alignment_metrics(
        fundus, cmr, torch.tensor(1 / 0.07).log())["n"] == 2
