import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_pytracking_root():
    script_path = Path(__file__).resolve()
    for p in [script_path.parent] + list(script_path.parents):
        if (p / "ltr").is_dir() and (p / "pytracking").is_dir():
            return p
        if (p / "pytracking" / "ltr").is_dir():
            return p / "pytracking"
    raise RuntimeError("Could not resolve pytracking root containing ltr/")


PYTRACKING_ROOT = resolve_pytracking_root()
root_str = str(PYTRACKING_ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from ltr.models.backbone.resnet18_vggm import resnet18_vggmconv1


class _ExportLRNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return F.local_response_norm(x, size=5, alpha=0.0005, beta=0.75, k=2.0)

    @staticmethod
    def symbolic(g, x):
        return g.op(
            "LRN",
            x,
            size_i=5,
            alpha_f=0.0005,
            beta_f=0.75,
            bias_f=2.0,
        )


class ExportableLRN(nn.Module):
    def forward(self, x):
        return _ExportLRNFunction.apply(x)


class EcoBackboneOnnxWrapper(nn.Module):
    def __init__(self, weights_path):
        super(EcoBackboneOnnxWrapper, self).__init__()
        self.backbone = resnet18_vggmconv1(["vggconv1", "layer3"], path=str(weights_path))

        if hasattr(self.backbone, "vgglrn"):
            self.backbone.vgglrn = ExportableLRN()

        self.backbone.eval()

    def forward(self, x):
        outputs = self.backbone(x)
        return outputs["vggconv1"], outputs["layer3"]


def parse_args():
    parser = argparse.ArgumentParser(description="Export the ECO backbone to ONNX.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
        )

    print("Exported ONNX to {}".format(args.output))


if __name__ == "__main__":
    main()
