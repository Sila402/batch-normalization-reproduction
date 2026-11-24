"""Figure 3: internal covariate shift (ICS) as defined in
Santurkar et al. (2018), Definition 2.1.

For layer i at training step t, on one batch (x, y):

    G_i  = gradient of the loss w.r.t. the parameters of layer i,
           with all layers at their current values W^t.
    G'_i = the same gradient, on the same batch, after all *previous*
           layers 1..i-1 have been updated with one SGD step
           (W_j <- W_j - lr * G_j), layers i..k unchanged.

ICS is the difference between them: ||G_i - G'_i||_2 (ideal: 0) and
cos(G_i, G'_i) (ideal: 1). If BatchNorm worked by reducing ICS, the BN
network should have a smaller l2 difference and a cosine closer to 1.

No training happens here. We evaluate this at every saved snapshot
(step 0, 500, ..., 15000) on a few random training batches and average.

"Layer" here: each Conv2d together with its BatchNorm (if any), and each
Linear layer. That gives 10 layers for our VGG (8 conv + 2 linear). The
paper reports VGG layers #5 and #10 (Fig. 3a).

Run from the repository root:
    python -m final_reproduction.experiments.analyze_ics
"""

import argparse
import copy
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets, transforms

from final_reproduction.models.vgg import VGGA

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / "data"
RESULTS_ROOT = ROOT / "results"
OUTPUT_DIR = RESULTS_ROOT / "figure3"

SEED = 42
BATCH_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def layer_groups(model):
    """List of parameter lists, one per layer (conv+BN, or linear)."""
    groups = []
    for module in list(model.features) + list(model.classifier):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            groups.append(list(module.parameters()))
        elif isinstance(module, nn.BatchNorm2d):
            # gamma and beta belong to the conv layer just before.
            groups[-1].extend(module.parameters())
    return groups


def gradients(model, images, labels):
    """Gradient of every layer group, as one flat vector per group."""
    model.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(images), labels)
    loss.backward()
    return [
        torch.cat([p.grad.flatten() for p in group])
        for group in layer_groups(model)
    ]


def ics_for_batch(model, images, labels, lr):
    """l2 difference and cosine between G_i and G'_i for every layer."""
    # Work on copies: train-mode forward passes change BN running stats.
    base = copy.deepcopy(model)
    g = gradients(base, images, labels)

    # Gradient of each group at W^t, kept per parameter for the update.
    per_param_grads = [
        [p.grad.detach().clone() for p in group]
        for group in layer_groups(base)
    ]

    results = []
    for i in range(len(g)):
        shifted = copy.deepcopy(model)
        groups = layer_groups(shifted)
        with torch.no_grad():
            for j in range(i):  # update previous layers only
                for p, grad in zip(groups[j], per_param_grads[j]):
                    p -= lr * grad
        g_prime = gradients(shifted, images, labels)[i]

        diff = (g[i] - g_prime).norm().item()
        cos = nn.functional.cosine_similarity(
            g[i], g_prime, dim=0).item()
        results.append((diff, cos))
        del shifted
    return results


def initial_model(use_bn):
    """Step 0: same seed, same initial weights as train_figure1.py."""
    torch.manual_seed(SEED)
    return VGGA(use_batchnorm=use_bn).to(DEVICE)


def load_snapshot(path, use_bn):
    model = VGGA(use_batchnorm=use_bn).to(DEVICE)
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Step size of the update (training LR).")
    parser.add_argument("--batches", type=int, default=5,
                        help="Random batches averaged per snapshot.")
    parser.add_argument("--plot-layers", type=int, nargs="+",
                        default=[5, 10])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = datasets.CIFAR10(
        root=str(DATA_ROOT), train=True, download=True,
        transform=transforms.ToTensor(),
    )
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
            model.train()

            # Same batches for both models at the same step.
            generator = torch.Generator().manual_seed(SEED + step)
            sums = None
            for _ in range(args.batches):
                idx = torch.randint(len(data), (BATCH_SIZE,),
                                    generator=generator).tolist()
                images = torch.stack([data[i][0] for i in idx]).to(DEVICE)
                labels = torch.tensor([data[i][1] for i in idx],
                                      device=DEVICE)
                res = ics_for_batch(model, images, labels, args.lr)
                if sums is None:
                    sums = [[0.0, 0.0] for _ in res]
                for s, (d, c) in zip(sums, res):
                    s[0] += d
                    s[1] += c

            for layer, (d, c) in enumerate(sums, start=1):
                rows.append({
                    "model": name, "step": step, "layer": layer,
                    "l2_difference": d / args.batches,
                    "cosine": c / args.batches,
                })
            print(f"{name} step {step} done", flush=True)
            del model

    csv_path = OUTPUT_DIR / "ics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", csv_path)

    # Like Fig. 3a: rows = layers, columns = l2 difference / cosine.
    n = len(args.plot_layers)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.6 * n), squeeze=False)
    styles = {"with_bn": ("Standard + BatchNorm", "tab:blue"),
              "without_bn": ("Standard", "tab:orange")}
    for r, layer in enumerate(args.plot_layers):
        for name, (label, color) in styles.items():
            sel = [x for x in rows
                   if x["model"] == name and x["layer"] == layer]
            steps = [x["step"] for x in sel]
            axes[r, 0].plot(steps, [x["l2_difference"] for x in sel],
                            label=label, color=color)
            axes[r, 1].plot(steps, [x["cosine"] for x in sel],
                            label=label, color=color)
        axes[r, 0].set_title(f"Layer {layer}: l2 difference (ideal 0)")
        axes[r, 1].set_title(f"Layer {layer}: cosine angle (ideal 1)")
        for ax in axes[r]:
            ax.set_xlabel("Training steps")
            ax.grid(alpha=0.3)
    axes[0, 1].legend()
    fig.tight_layout()
    path = OUTPUT_DIR / "ics.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("Saved:", path)


if __name__ == "__main__":
    main()
