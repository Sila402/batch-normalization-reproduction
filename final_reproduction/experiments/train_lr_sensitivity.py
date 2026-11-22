
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from final_reproduction.models.vgg import VGGA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--model", choices=["with_bn", "without_bn"],
                        default="with_bn")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    root = Path(__file__).resolve().parents[1]
    data_root = root.parent / "data"
    output_dir = root / "results" / "lr_sensitivity" / (
        f"{args.model}_lr_{args.lr}_seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()

    dataset = datasets.CIFAR10(
        root=str(data_root),
        train=True,
        download=False,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = VGGA(
        use_batchnorm=(args.model == "with_bn")
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0,
        weight_decay=0,
    )

    criterion = nn.CrossEntropyLoss()
    iterator = iter(loader)
    rows = []
    start = time.perf_counter()

    print("Device:", device)
    print("Model:", args.model)
    print("Learning rate:", args.lr)
    print("Steps:", args.steps)

    model.train()

    for step in range(1, args.steps + 1):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, labels = next(iterator)

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        loss = criterion(predictions, labels)

        if not torch.isfinite(loss):
            print("Non-finite loss at step", step)
            break

        loss.backward()
        optimizer.step()

        accuracy = (
            (predictions.argmax(dim=1) == labels)
            .float().mean().item() * 100
        )

        rows.append({
            "step": step,
            "loss": loss.item(),
            "train_accuracy": accuracy,
        })

        if step == 1 or step % 100 == 0:
            print(
                f"Step {step}/{args.steps} | "
                f"Loss: {loss.item():.4f} | "
                f"Accuracy: {accuracy:.2f}%"
            )

    csv_path = output_dir / "metrics.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "loss", "train_accuracy"]
        )
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.perf_counter() - start

    print("Elapsed seconds:", round(elapsed, 2))
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
