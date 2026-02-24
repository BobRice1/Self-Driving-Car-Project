from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from src.model import SelfDrivingRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the self-driving model architecture.")
    parser.add_argument("--backbone", type=str, default="tf_efficientnetv2_s", help="timm backbone name.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Shared head hidden size.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout in shared head.")
    parser.add_argument("--height", type=int, default=240, help="Input image height.")
    parser.add_argument("--width", type=int, default=320, help="Input image width.")
    parser.add_argument("--batch_size", type=int, default=1, help="Dummy batch size for visualization.")
    parser.add_argument("--depth", type=int, default=3, help="Depth for torchinfo summary.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device used for dummy forward pass.",
    )
    parser.add_argument(
        "--summary_out",
        type=Path,
        default=None,
        help="Optional path to save torchinfo summary text.",
    )
    parser.add_argument(
        "--mermaid_out",
        type=Path,
        default=None,
        help="Optional path to save a Mermaid architecture diagram (.mmd).",
    )
    parser.add_argument(
        "--dot_out",
        type=Path,
        default=None,
        help="Optional path to save a Graphviz DOT architecture diagram (.dot).",
    )
    return parser.parse_args()


def resolve_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

q
def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_mermaid(
    backbone: str,
    hidden_dim: int,
    batch_size: int,
    height: int,
    width: int,
) -> str:
    return f"""flowchart LR
    A[Input\\n{batch_size} x 3 x {height} x {width}] --> B[Backbone\\n{backbone}\\n(global avg pool)]
    B --> C[Feature vector\\nB x F]
    C --> D[Shared head\\nLinear(F->{hidden_dim}) + GELU + Dropout]
    D --> E[Angle head\\nLinear({hidden_dim}->17)]
    D --> F[Speed head\\nLinear({hidden_dim}->1)]
    E --> G[Softmax over 17 bins]
    G --> H[Expected value with bins\\n0/16 ... 16/16]
    H --> I[angle_pred in [0,1]]
    F --> J[Sigmoid]
    J --> K[speed_pred in [0,1]]
"""


def build_dot(
    backbone: str,
    hidden_dim: int,
    batch_size: int,
    height: int,
    width: int,
) -> str:
    return f"""digraph SelfDrivingRegressor {{
    rankdir=LR;
    graph [fontname="Helvetica"];
    node [shape=box, style="rounded,filled", fillcolor="#f5f8ff", color="#4a5568", fontname="Helvetica"];
    edge [color="#2d3748", arrowsize=0.8];

    input [label="Input\\n{batch_size} x 3 x {height} x {width}"];
    backbone [label="Backbone\\n{backbone}\\nnum_classes=0, global_pool=avg"];
    features [label="Features\\nB x F"];
    shared [label="Shared head\\nLinear(F->{hidden_dim})\\nGELU\\nDropout"];
    angle_head [label="Angle head\\nLinear({hidden_dim}->17)"];
    speed_head [label="Speed head\\nLinear({hidden_dim}->1)"];
    softmax [label="Softmax(dim=1)"];
    expected [label="Expected value\\nSum(p_i * bin_i), i=0..16"];
    angle_out [label="angle_pred\\nB x 1, [0,1]", fillcolor="#e6fffa"];
    sigmoid [label="Sigmoid"];
    speed_out [label="speed_pred\\nB x 1, [0,1]", fillcolor="#e6fffa"];

    input -> backbone;
    backbone -> features;
    features -> shared;
    shared -> angle_head;
    shared -> speed_head;
    angle_head -> softmax;
    softmax -> expected;
    expected -> angle_out;
    speed_head -> sigmoid;
    sigmoid -> speed_out;
}}
"""


def run() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    model = SelfDrivingRegressor(
        backbone=args.backbone,
        pretrained=False,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    model.eval()

    print("\nModel definition:\n")
    print(model)
    print(f"\nTrainable parameters: {count_trainable_params(model):,}")
    print(f"Device: {device}")

    x = torch.randn(args.batch_size, 3, args.height, args.width, device=device)
    with torch.no_grad():
        outputs: Dict[str, Any] = model(x)

    print("\nOutput tensor shapes:")
    for key, value in outputs.items():
        print(f"- {key}: {tuple(value.shape)}")

    summary_text = None
    try:
        from torchinfo import summary

        stats = summary(
            model,
            input_size=(args.batch_size, 3, args.height, args.width),
            depth=args.depth,
            col_names=("input_size", "output_size", "num_params"),
            verbose=0,
            device=str(device),
        )
        summary_text = str(stats)
        print("\nTorchinfo summary:\n")
        print(summary_text)
    except ModuleNotFoundError:
        print("\nTorchinfo not installed. Install with: pip install torchinfo")

    if args.summary_out is not None:
        if summary_text is None:
            raise RuntimeError("Cannot write summary file because torchinfo is not installed.")
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(summary_text, encoding="utf-8")
        print(f"\nSummary written: {args.summary_out}")

    if args.mermaid_out is not None:
        mermaid = build_mermaid(
            backbone=args.backbone,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
        )
        args.mermaid_out.parent.mkdir(parents=True, exist_ok=True)
        args.mermaid_out.write_text(mermaid, encoding="utf-8")
        print(f"Mermaid diagram written: {args.mermaid_out}")

    if args.dot_out is not None:
        dot = build_dot(
            backbone=args.backbone,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
        )
        args.dot_out.parent.mkdir(parents=True, exist_ok=True)
        args.dot_out.write_text(dot, encoding="utf-8")
        print(f"DOT diagram written: {args.dot_out}")


if __name__ == "__main__":
    run()
