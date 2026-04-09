"""
眼底图像门控模型 - 推理模块
用法：
  from fundus_gate.inference import FundusGateChecker
  checker = FundusGateChecker()
  is_fundus, prob = checker.check(image_path_or_pil)
"""

import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

# 默认模型路径（相对于本文件）
_DEFAULT_CKPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints", "best_fundus_gate.pth"
)


class FundusGateChecker:
    """
    判断输入图像是否为眼底图像。

    Args:
        ckpt_path: 模型权重路径，默认使用 checkpoints/best_fundus_gate.pth
        threshold: 判定为眼底图像的概率阈值（默认0.5）
        device: 推理设备
    """

    def __init__(self, ckpt_path: str = None, threshold: float = 0.5, device: str = None):
        self.threshold = threshold

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        ckpt_path = ckpt_path or _DEFAULT_CKPT

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"眼底门控模型权重不存在: {ckpt_path}\n"
                "请先运行 prepare_data.py 和 train.py 完成训练。"
            )

        self.model, self.img_size = self._load_model(ckpt_path)
        self.transform = self._build_transform(self.img_size)
        print(f"[FundusGate] 模型加载完成，设备: {self.device}, 阈值: {threshold}")

    def _load_model(self, ckpt_path: str):
        import torchvision.models as tvm

        ckpt = torch.load(ckpt_path, map_location=self.device)
        arch = ckpt.get("arch", "resnet18")
        img_size = ckpt.get("img_size", 224)

        tv_models = {
            "resnet18": (tvm.resnet18, lambda m: setattr(m, "fc", nn.Linear(m.fc.in_features, 1))),
            "resnet50": (tvm.resnet50, lambda m: setattr(m, "fc", nn.Linear(m.fc.in_features, 1))),
        }

        arch_lower = arch.lower()
        if arch_lower in tv_models:
            model_fn, head_replacer = tv_models[arch_lower]
            model = model_fn(pretrained=False)
            head_replacer(model)
        elif TIMM_AVAILABLE:
            model = timm.create_model(arch, pretrained=False, num_classes=1)
        else:
            raise RuntimeError(f"不支持的模型架构: {arch}，请安装 timm 或使用 resnet18/resnet50")

        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        model.to(self.device)
        return model, img_size

    @staticmethod
    def _build_transform(img_size: int):
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def check(self, image) -> tuple:
        """
        检查图像是否为眼底图像。

        Args:
            image: PIL.Image 或 文件路径字符串

        Returns:
            (is_fundus: bool, prob: float)
              - is_fundus: True 表示是眼底图像
              - prob: 是眼底图像的概率 [0, 1]
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logit = self.model(tensor).squeeze()
        prob = float(torch.sigmoid(logit).cpu())
        is_fundus = prob >= self.threshold
        return is_fundus, prob

    def check_batch(self, images: list) -> list:
        """批量检查图像列表"""
        results = []
        for img in images:
            results.append(self.check(img))
        return results


# ─── 命令行测试 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="眼底门控推理测试")
    parser.add_argument("image_paths", nargs="+", help="待测图像路径")
    parser.add_argument("--ckpt", default=None, help="模型权重路径")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    checker = FundusGateChecker(ckpt_path=args.ckpt, threshold=args.threshold)

    for path in args.image_paths:
        is_fundus, prob = checker.check(path)
        status = "✓ 眼底图像" if is_fundus else "✗ 非眼底图像"
        print(f"{path}: {status}  (概率={prob:.4f})")
