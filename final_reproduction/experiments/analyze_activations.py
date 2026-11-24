"""Figure 1(c): how the input distribution of a layer changes during
training, with and without BatchNorm.

No training happens here. For every saved snapshot (step 0, 500, ...,
15000) we load the weights, run one fixed batch of training images
through the network and record the activations that enter a chosen
conv layer. We then save percentiles of those activations to a CSV and
plot them over training steps, as in Santurkar et al. (2018), Fig. 1(c).

Run from the repository root:
    python -m final_reproduction.experiments.analyze_activations
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms

from final_reproduction.models.vgg import VGGA

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / "data"
RESULTS_ROOT = ROOT / "results"
OUTPUT_DIR = RESULTS_ROOT / "figure1c"

SEED = 42
NUM_IMAGES = 1000
PERCENTILES = [5, 25, 50, 75, 95]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fixed_batch():
    """The same NUM_IMAGES training images for every snapshot."""
    data = datasets.CIFAR10(
        root=str(DATA_ROOT), train=True, download=True,
        transform=transforms.ToTensor(),
    )
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(data), generator=generator)[:NUM_IMAGES]
    images = torch.stack([data[i][0] for i in indices.tolist()])
    return images.to(DEVICE)


def initial_model(use_bn):
    """Step 0. train_figure1.py calls set_seed() and then builds the
    model, so the same seed gives exactly the same starting weights."""
    torch.manual_seed(SEED)
    return VGGA(use_batchnorm=use_bn).to(DEVICE)


def load_snapshot(path, use_bn):
    model = VGGA(use_batchnorm=use_bn).to(DEVICE)
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model"])
    return model


@torch.no_grad()
def layer_input(model, images, layer):
    """Activations entering conv layer number `layer` (1-based).

    The input of conv layer k is the output of the ReLU of conv layer
    k-1 (after max pooling, if there is one in between). We capture the
    tensor that the k-th Conv2d receives with a forward hook.
    """
    convs = [m for m in model.features if isinstance(m, nn.Conv2d)]
    captured = {}

    def hook(module, inputs):
        captured["x"] = inputs[0].detach().flatten().cpu()

    handle = convs[layer - 1].register_forward_pre_hook(hook)

    # Train mode: BatchNorm uses the statistics of this batch, as it
    # does during training. The model is thrown away afterwards, so the
    # running_mean / running_var updates do not matter.
    model.train()
    model(images)
    handle.remove()
    return captured["x"].numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[3, 6])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = fixed_batch()
    rows = []

    for name, use_bn in [("with_bn", True), ("without_bn", False)]:
        ckpt_dir = RESULTS_ROOT / name / "checkpoints"
        snapshots = sorted(ckpt_dir.glob("step_*.pt"))
        if not snapshots:
            raise FileNotFoundError(f"No snapshots in {ckpt_dir}")

        entries = [(0, None)] + [
            (int(p.stem.split("_")[1]), p) for p in snapshots
        ]

        for step, path in entries:
            model = (initial_model(use_bn) if path is None
                     else load_snapshot(path, use_bn))
            for layer in args.layers:
                acts = layer_input(model, images, layer)
                values = np.percentile(acts, PERCENTILES)
                rows.append({
                    "model": name, "layer": layer, "step": step,
                    "mean": float(acts.mean()), "std": float(acts.std()),
                    **{f"p{p}": float(v) for p, v in zip(PERCENTILES, values)},
                })
            print(f"{name} step {step} done", flush=True)
            del model

    csv_path = OUTPUT_DIR / "activation_percentiles.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", csv_path)

    # One figure per layer: left = with BN, right = without BN.
    for layer in args.layers:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for ax, name, title in [
            (axes[0], "with_bn", "Standard + BatchNorm"),
            (axes[1], "without_bn", "Standard"),
        ]:
            r = [x for x in rows if x["model"] == name and x["layer"] == layer]
            steps = [x["step"] for x in r]
            ax.fill_between(steps, [x["p5"] for x in r],
                            [x["p95"] for x in r], alpha=0.2,
                            color="tab:blue", label="5–95 percentile")
            ax.fill_between(steps, [x["p25"] for x in r],
                            [x["p75"] for x in r], alpha=0.4,
                            color="tab:blue", label="25–75 percentile")
            ax.plot(steps, [x["p50"] for x in r], color="tab:blue",
                    label="median")
            ax.plot(steps, [x["mean"] for x in r], color="black",
                    linestyle="--", label="mean")
            ax.set_title(f"{title}: input of conv layer {layer}")
            ax.set_xlabel("Training steps")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Activation value")
        axes[1].legend()
        fig.tight_layout()
        path = OUTPUT_DIR / f"layer{layer}_input_distribution.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print("Saved:", path)


if __name__ == "__main__":
    main()
