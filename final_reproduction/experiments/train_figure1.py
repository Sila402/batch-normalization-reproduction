
import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from final_reproduction.models.vgg import VGGA


# =========================================
# CONFIGURATION
# =========================================

SEED = 42
BATCH_SIZE = 128
LEARNING_RATE = 0.1

EVAL_EVERY = 100
SAVE_EVERY = 100
SNAPSHOT_EVERY = 500

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / "data"
RESULTS_ROOT = ROOT / "results"


# =========================================
# RANDOM SEED
# =========================================

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def get_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    torch_state = state["torch"]
    if not isinstance(torch_state, torch.Tensor):
        torch_state = torch.as_tensor(
            torch_state, dtype=torch.uint8
        )

    torch.set_rng_state(
        torch_state.cpu().to(dtype=torch.uint8)
    )

    if DEVICE.type == "cuda" and state["cuda"] is not None:
        cuda_states = []

        for cuda_state in state["cuda"]:
            if not isinstance(cuda_state, torch.Tensor):
                cuda_state = torch.as_tensor(
                    cuda_state, dtype=torch.uint8
                )

            cuda_states.append(
                cuda_state.cpu().to(dtype=torch.uint8)
            )

        torch.cuda.set_rng_state_all(cuda_states)


# =========================================
# DATASET
# =========================================

def load_data():
    transform = transforms.ToTensor()

    train_data = datasets.CIFAR10(
        root=str(DATA_ROOT),
        train=True,
        download=False,
        transform=transform,
    )

    test_data = datasets.CIFAR10(
        root=str(DATA_ROOT),
        train=False,
        download=False,
        transform=transform,
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    return train_data, test_loader


# =========================================
# EVALUATION
# =========================================

@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()

    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    model.train()

    return 100.0 * correct / total


# =========================================
# CHECKPOINT FUNCTIONS
# =========================================

def save_checkpoint(path, state):
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    temp_path = path.with_name(
        path.name + ".tmp"
    )

    torch.save(state, temp_path)
    os.replace(temp_path, path)


def create_state(
    model,
    optimizer,
    step,
    best_accuracy,
    order,
    position,
    generator,
    config,
    running_loss,
    running_correct,
    running_total,
):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_accuracy": best_accuracy,
        "order": order,
        "position": position,
        "generator_state": generator.get_state(),
        "rng_state": get_rng_state(),
        "config": config,
        "running_loss": running_loss,
        "running_correct": running_correct,
        "running_total": running_total,
    }


# =========================================
# TRAINING
# =========================================

def train(use_bn, max_steps, resume):
    set_seed()

    model_name = (
        "with_bn" if use_bn else "without_bn"
    )

    run_dir = RESULTS_ROOT / model_name
    checkpoint_dir = run_dir / "checkpoints"

    checkpoint_dir.mkdir(
        parents=True, exist_ok=True
    )

    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"
    csv_path = run_dir / "metrics.csv"

    config = {
        "model": "VGGA",
        "use_bn": use_bn,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "SGD",
        "momentum": 0.0,
        "weight_decay": 0.0,
        "eval_every": EVAL_EVERY,
        "save_every": SAVE_EVERY,
    }

    train_data, test_loader = load_data()

    model = VGGA(
        use_batchnorm=use_bn
    ).to(DEVICE)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.0,
        weight_decay=0.0,
    )

    criterion = nn.CrossEntropyLoss()

    generator = torch.Generator()
    generator.manual_seed(SEED)

    step = 0
    best_accuracy = -1.0
    order = None
    position = 0

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    # =====================================
    # RESUME OR NEW TRAINING
    # =====================================

    if resume:
        if not latest_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {latest_path}"
            )

        # Load only checkpoints created by this project.
        # Load on CPU to keep RNG states on CPU.
        checkpoint = torch.load(
            latest_path,
            map_location="cpu",
            weights_only=False,
        )

        if checkpoint["config"] != config:
            raise ValueError(
                "Checkpoint configuration does not match."
            )

        model.load_state_dict(
            checkpoint["model"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        step = checkpoint["step"]
        best_accuracy = checkpoint["best_accuracy"]
        order = checkpoint["order"]
        position = checkpoint["position"]

        # Restore data-shuffling generator.
        generator_state = checkpoint[
            "generator_state"
        ]

        if not isinstance(
            generator_state, torch.Tensor
        ):
            generator_state = torch.as_tensor(
                generator_state,
                dtype=torch.uint8,
            )

        generator.set_state(
            generator_state.cpu().to(
                dtype=torch.uint8
            )
        )

        running_loss = checkpoint[
            "running_loss"
        ]

        running_correct = checkpoint[
            "running_correct"
        ]

        running_total = checkpoint[
            "running_total"
        ]

        restore_rng_state(
            checkpoint["rng_state"]
        )

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Metrics file missing: {csv_path}"
            )

        # Keep metrics up to checkpoint step.
        with open(
            csv_path, newline=""
        ) as f:
            rows = list(
                csv.DictReader(f)
            )

        rows = [
            row for row in rows
            if int(row["step"]) <= step
        ]

        with open(
            csv_path, "w", newline=""
        ) as f:
            writer = csv.writer(f)

            writer.writerow([
                "step",
                "train_loss",
                "train_accuracy",
                "test_accuracy",
            ])

            for row in rows:
                writer.writerow([
                    row["step"],
                    row["train_loss"],
                    row["train_accuracy"],
                    row["test_accuracy"],
                ])

        print(
            f"RESUMED FROM STEP {step}",
            flush=True,
        )

    else:
        if (
            latest_path.exists()
            or csv_path.exists()
        ):
            raise FileExistsError(
                "Existing experiment found. "
                "Use --resume to continue."
            )

        with open(
            csv_path, "w", newline=""
        ) as f:
            writer = csv.writer(f)

            writer.writerow([
                "step",
                "train_loss",
                "train_accuracy",
                "test_accuracy",
            ])

        print(
            "STARTING NEW TRAINING",
            flush=True,
        )

    if max_steps <= step:
        print(
            f"Already at step {step}. "
            "Specify a larger --steps value."
        )
        return

    print("Device:", DEVICE)
    print("Model:", model_name)
    print("Target steps:", max_steps)

    model.train()

    # =====================================
    # MAIN TRAINING LOOP
    # =====================================

    with open(
        csv_path, "a", newline=""
    ) as f:
        writer = csv.writer(f)

        while step < max_steps:

            # Shuffle at start of each epoch.
            if (
                order is None
                or position >= len(order)
            ):
                order = torch.randperm(
                    len(train_data),
                    generator=generator,
                )

                position = 0

            indices = order[
                position:position + BATCH_SIZE
            ].tolist()

            position += len(indices)

            batch = [
                train_data[i]
                for i in indices
            ]

            images = torch.stack(
                [item[0] for item in batch]
            ).to(DEVICE)

            labels = torch.tensor(
                [item[1] for item in batch],
                dtype=torch.long,
                device=DEVICE,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(images)
            loss = criterion(
                outputs, labels
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at step {step + 1}"
                )

            loss.backward()
            optimizer.step()

            step += 1

            running_loss += (
                loss.item() * len(indices)
            )

            running_correct += (
                outputs.argmax(dim=1)
                == labels
            ).sum().item()

            running_total += len(indices)

            improved = False

            # =============================
            # EVALUATION
            # =============================

            if step % EVAL_EVERY == 0:

                train_loss = (
                    running_loss
                    / running_total
                )

                train_accuracy = (
                    100.0
                    * running_correct
                    / running_total
                )

                test_accuracy = evaluate(
                    model,
                    test_loader,
                )

                print(
                    f"Step {step}/{max_steps} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Train: {train_accuracy:.2f}% | "
                    f"Test: {test_accuracy:.2f}%",
                    flush=True,
                )

                writer.writerow([
                    step,
                    train_loss,
                    train_accuracy,
                    test_accuracy,
                ])

                f.flush()

                if (
                    test_accuracy
                    > best_accuracy
                ):
                    best_accuracy = (
                        test_accuracy
                    )
                    improved = True

                    print(
                        f"New best: "
                        f"{best_accuracy:.2f}%",
                        flush=True,
                    )

                running_loss = 0.0
                running_correct = 0
                running_total = 0

            # =============================
            # SAVE CHECKPOINTS
            # =============================

            should_save = (
                step % SAVE_EVERY == 0
                or step == max_steps
            )

            if should_save or improved:

                state = create_state(
                    model,
                    optimizer,
                    step,
                    best_accuracy,
                    order,
                    position,
                    generator,
                    config,
                    running_loss,
                    running_correct,
                    running_total,
                )

                if improved:
                    save_checkpoint(
                        best_path,
                        state,
                    )

                    print(
                        "Saved BEST checkpoint",
                        flush=True,
                    )

                if should_save:
                    save_checkpoint(
                        latest_path,
                        state,
                    )

                    print(
                        "Saved LATEST checkpoint",
                        flush=True,
                    )

                if (
                    step % SNAPSHOT_EVERY
                    == 0
                ):
                    snapshot_path = (
                        checkpoint_dir
                        / f"step_{step:06d}.pt"
                    )

                    save_checkpoint(
                        snapshot_path,
                        state,
                    )

                    print(
                        "Saved SNAPSHOT checkpoint",
                        flush=True,
                    )

    print(
        f"\nTRAINING COMPLETED: {model_name}",
        flush=True,
    )


# =========================================
# COMMAND LINE
# =========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "with_bn",
            "without_bn",
        ],
        required=True,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=15000,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    if args.steps < 1:
        parser.error(
            "--steps must be at least 1"
        )

    train(
        use_bn=(
            args.model == "with_bn"
        ),
        max_steps=args.steps,
        resume=args.resume,
    )
