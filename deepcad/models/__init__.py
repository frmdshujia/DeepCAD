from .cmr_encoder import CMREncoderV4
from .heads import ClinicalRiskMLP, MultiTaskHead

__all__ = [
    "CMREncoderV4",
    "RETFoundFundusEncoder",
    "FundusBinaryClassifier",
    "MultiTaskHead",
    "ClinicalRiskMLP",
]


def __getattr__(name):
    """Load the timm-dependent retinal models only when they are requested."""
    if name in {"RETFoundFundusEncoder", "FundusBinaryClassifier"}:
        from .fundus_encoder import FundusBinaryClassifier, RETFoundFundusEncoder
        return {
            "RETFoundFundusEncoder": RETFoundFundusEncoder,
            "FundusBinaryClassifier": FundusBinaryClassifier,
        }[name]
    raise AttributeError(name)
