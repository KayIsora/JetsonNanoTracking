from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
LTR_ROOT = REPO_ROOT / "MyECOTracker" / "pytracking"
for root in (REPO_ROOT, LTR_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from ltr.models.backbone.resnet18_vggm import resnet18_vggmconv1


class EcoBackboneOnnxWrapper(torch.nn.Module):
    def __init__(self, weights_path: Path):
        super().__init__()
        self.backbone = resnet18_vggmconv1(["vggconv1", "layer3"], path=str(weights_path))
        self.backbone.eval()

    def forward(self, x):
        outputs = self.backbone(x)
        return outputs["vggconv1"], outputs["layer3"]


def parse_args():
    parser = argparse.ArgumentParser(description="Export the ECO backbone to ONNX.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=LTR_ROOT / "pretrained_network" / "resnet18_vggmconv1" / "resnet18_vggmconv1.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LTR_ROOT / "pretrained_network" / "resnet18_vggmconv1" / "resnet18_vggmconv1_otb_dyn.onnx",
    )
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--opt-batch", type=int, default=5)
    parser.add_argument("--opt-size", type=int, default=224)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = EcoBackboneOnnxWrapper(args.weights)
    model.eval()

    dummy = torch.randn(args.opt_batch, 3, args.opt_size, args.opt_size, dtype=torch.float32)

    dynamic_axes = {
        "input": {0: "batch", 2: "height", 3: "width"},
        "vggconv1": {0: "batch", 2: "vggconv1_height", 3: "vggconv1_width"},
        "layer3": {0: "batch", 2: "layer3_height", 3: "layer3_width"},
    }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(args.output),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["vggconv1", "layer3"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

    print("Exported ONNX to {}".format(args.output))


if __name__ == "__main__":
    main()
