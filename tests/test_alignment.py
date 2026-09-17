import torch
import torch.nn as nn

from deepcad.training.alignment import (
    alignment_metrics, collect_participant_embeddings,
)


class FakeFundusEncoder(nn.Module):
    def forward(self, images):
        projection = torch.nn.functional.normalize(images, dim=-1)
        return projection, images


def test_single_image_embeddings_preserve_participants():
    batch = {
        "eid": torch.tensor([10, 20]),
        "image": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "cmr_embedding": torch.tensor(
            [[1.0, 1.0], [0.0, 1.0]]),
        "t1_available": torch.tensor([True, False]),
    }
    eids, fundus, cmr, t1 = collect_participant_embeddings(
        FakeFundusEncoder(), nn.Identity(), [batch], torch.device("cpu"), "none")
    assert eids.tolist() == [10, 20]
    assert t1.tolist() == [True, False]
    assert alignment_metrics(
        fundus, cmr, torch.tensor(1 / 0.1).log())["n"] == 2


def test_duplicate_participant_images_are_rejected_at_evaluation():
    batch = {
        "eid": torch.tensor([10, 10]),
        "image": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "cmr_embedding": torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        "t1_available": torch.tensor([True, True]),
    }
    try:
        collect_participant_embeddings(
            FakeFundusEncoder(), nn.Identity(), [batch], torch.device("cpu"),
            "none")
    except RuntimeError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("Duplicate participant images were not rejected.")
