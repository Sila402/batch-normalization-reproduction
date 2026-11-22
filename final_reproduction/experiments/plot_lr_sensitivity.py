
from pathlib import Path
import csv

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "lr_sensitivity"
OUTPUT = DATA / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

learning_rates = [0.01, 0.1, 0.5]
models = ["with_bn", "without_bn"]

colors = {
    "with_bn": "tab:blue",
    "without_bn": "tab:orange",
}

for metric, ylabel, filename in [
    ("loss", "Training Loss", "lr_training_loss.png"),
    ("train_accuracy", "Minibatch Accuracy (%)",
     "lr_training_accuracy.png"),
]:
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4.5), sharey=True
    )

    for ax, lr in zip(axes, learning_rates):
        for model in models:
            csv_path = (
                DATA / f"{model}_lr_{lr}_seed_42" / "metrics.csv"
            )

            if not csv_path.exists():
                raise FileNotFoundError(csv_path)

            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))

            steps = [int(row["step"]) for row in rows]
            values = [float(row[metric]) for row in rows]

            ax.plot(
                steps,
                values,
                label="With BN" if model == "with_bn"
                else "Without BN",
                color=colors[model],
                alpha=0.8,
                linewidth=1.3,
            )

        ax.set_title(f"Learning Rate = {lr}")
        ax.set_xlabel("Training Steps")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(ylabel)
    axes[-1].legend()

    fig.suptitle(
        f"Batch Normalization: {ylabel} vs Learning Rate",
        fontsize=14,
    )

    # Loss grafiginin Y eksenini sinirla
    if metric == "loss":
        for ax in axes:
            ax.set_ylim(0, 3.5)

    fig.tight_layout()

    save_path = OUTPUT / filename
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print("Saved:", save_path)

    plt.show()

print("\nAll plots generated successfully!")
