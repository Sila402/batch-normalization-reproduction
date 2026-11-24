"""Why is the test accuracy of the noisy-BN model low?

During training, NoisyBatchNorm2d multiplies every BN output by a random
scale s (mean (1 + n_sigma) / 2 = 1.125) and adds a random shift m
(mean 0). At test time we switch the noise off. Two things then differ
from training:

1. The layers after a noisy BN never saw "clean" inputs during training.
2. The running mean / variance of the later BN layers were collected on
   noisy inputs, but at test time their inputs are clean.

The paper does not say how the noisy model is evaluated, so this is our
choice. This script evaluates one saved noisy-BN checkpoint on the test
set in four ways, without any training:

  clean        noise off, stored running stats (what train_figure1.py
               reports)
  noise        noise on (fresh sample per batch), stored running stats;
               softmax averaged over --passes forward passes
  expected     noise replaced by its expected value: s = 1.125, m = 0
  recalibrated noise off, BN running stats recomputed on clean training
               batches (no weights change)

If "noise", "expected" or "recalibrated" is clearly better than "clean",
the low test accuracy comes from the train/test mismatch above, not
from the model failing to learn.

Run from the repository root, e.g. on Colab:
    python -m final_reproduction.experiments.evaluate_noisy_bn \\
        --checkpoint /content/drive/MyDrive/BatchNorm_Reproduction/runs/noisy_bn_lr0.1_seed0/checkpoints/latest.pt
"""

import argparse
import copy
import csv
from pathlib import Path

import torch
from torch import nn
from torchvision import datasets, transforms

from final_reproduction.models.vgg import NoisyBatchNorm2d, VGGA

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_noise_mode(model, mode):
    """mode: "off", "sample" or "expected". BN itself stays in eval mode
    (stored running statistics); only the noise changes."""
    for module in model.modules():
        if not isinstance(module, NoisyBatchNorm2d):
            continue

        def forward(x, m=module):
            out = nn.BatchNorm2d.forward(m, x)
            if mode == "sample":
                return m.add_noise(out)
            if mode == "expected":
                return out * (1.0 + m.n_sigma) / 2.0
            return out

        module.forward = forward


@torch.no_grad()
def accuracy(model, loader, passes=1):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        probs = sum(model(images).softmax(dim=1) for _ in range(passes))
        correct += (probs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def recalibrate_bn(model, loader, batches):
    """Recompute running mean / var of every BN layer on clean inputs."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None  # cumulative average over batches
    set_noise_mode(model, "off")
    model.train()  # BN collects statistics in train mode
    # In train mode our patched forward still calls BatchNorm2d.forward,
    # which updates the running statistics; no noise is added.
    for i, (images, _) in enumerate(loader):
        if i >= batches:
            break
        model(images.to(DEVICE))
    model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--passes", type=int, default=10,
                        help="Forward passes averaged in 'noise' mode.")
    parser.add_argument("--recalibration-batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    transform = transforms.ToTensor()
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(str(DATA_ROOT), train=False, download=True,
                         transform=transform),
        batch_size=256, shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(str(DATA_ROOT), train=True, download=True,
                         transform=transform),
        batch_size=128, shuffle=True)

    checkpoint = Path(args.checkpoint)
    state = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    base = VGGA(use_batchnorm=True, noisy_bn=True).to(DEVICE)
    base.load_state_dict(state["model"])
    print("Checkpoint:", checkpoint, "| step", state.get("step"))

    results = []

    model = copy.deepcopy(base)
    set_noise_mode(model, "off")
    results.append(("clean", accuracy(model, test_loader)))

    model = copy.deepcopy(base)
    set_noise_mode(model, "sample")
    results.append(("noise", accuracy(model, test_loader, args.passes)))

    model = copy.deepcopy(base)
    set_noise_mode(model, "expected")
    results.append(("expected", accuracy(model, test_loader)))

    model = copy.deepcopy(base)
    recalibrate_bn(model, train_loader, args.recalibration_batches)
    results.append(("recalibrated", accuracy(model, test_loader)))

    print("\nTest accuracy of the noisy-BN model:")
    for mode, acc in results:
        print(f"  {mode:<13} {acc:6.2f}%")

    out_path = checkpoint.parent.parent / "noisy_bn_eval_modes.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "test_accuracy", "step"])
        for mode, acc in results:
            writer.writerow([mode, acc, state.get("step")])
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
