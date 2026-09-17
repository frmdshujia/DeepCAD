import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _yaml(name):
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def test_cmr_teacher_configuration_is_manuscript_locked():
    config = _yaml("cmr_teacher.yaml")
    classifications = config["classification_columns"].split(",")
    regressions = config["regression_columns"].split(",")
    assert classifications == [
        "prevalent_I21", "prevalent_I25", "history_revasc",
        "composite_cardiomyopathy_hf", "prevalent_I48",
    ]
    assert len(regressions) == 22
    assert config["backbone"] == "medsam2_heart"
    assert config["fusion_mode"] == "hierarchical"
    assert (config["epochs"], config["freeze_epochs"]) == (80, 3)
    assert (config["batch_size"], config["unfreeze_batch_size"]) == (32, 16)
    assert config["backbone_learning_rate"] == 1e-5
    assert config["new_module_learning_rate"] == 1e-4
    assert config["weight_decay"] == 0.1
    assert config["warmup_epochs"] == 3
    assert config["min_learning_rate_ratio"] == 0.01
    assert config["gradient_clip"] == 3.0
    assert config["frame_dropout"] == 0.15
    assert config["attention_dropout"] == 0.1
    assert config["focal_gamma"] == 2.0
    assert config["patience"] == 30


def test_stage1_configuration_is_manuscript_locked():
    config = _yaml("stage1_alignment.yaml")
    assert config["temperature"] == 0.1
    assert config["unfreeze_last_blocks"] == 12
    assert config["projection_dim"] == 128
    assert config["backbone_learning_rate"] == 1e-5
    assert config["new_module_learning_rate"] == 1e-4
    assert config["batch_size"] == 64
    assert config["epochs"] == 100
    assert config["warmup_epochs"] == 10


def test_stage2_sequential_configurations_are_locked():
    sdpp = _yaml("stage2_sdpp.yaml")
    shcc = _yaml("stage2_shcc.yaml")
    assert (sdpp["epochs"], sdpp["learning_rate"]) == (50, 1e-4)
    assert (shcc["epochs"], shcc["learning_rate"]) == (30, 5e-5)
    for config in (sdpp, shcc):
        assert config["batch_size"] == 64
        assert config["weight_decay"] == 1e-4
        assert config["loss_type"] == "focal"
        assert (config["focal_alpha"], config["focal_gamma"]) == (0.25, 2.0)
        assert config["unfreeze_last_blocks"] == 12
    assert sdpp["calibrate"] is False
    assert shcc["calibrate"] is True


def test_task_registry_matches_primary_cmr_yaml():
    tasks = json.loads((ROOT / "configs" / "v2_tasks.json").read_text())
    cmr = _yaml("cmr_teacher.yaml")
    assert tasks["classification_5"] == cmr["classification_columns"].split(",")
    assert tasks["regression_22"] == cmr["regression_columns"].split(",")
