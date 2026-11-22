
from pathlib import Path
import csv
import matplotlib.pyplot as plt

# Project directories
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "figure1"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics(model_name):
    csv_path = RESULTS_DIR / model_name / "metrics.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Empty metrics file: {csv_path}")

    return {
        key: [float(row[key]) for row in rows]
        for key in [
            "step",
            "train_loss",
            "train_accuracy",
            "test_accuracy",
        ]
    }


with_bn = load_metrics("with_bn")
without_bn = load_metrics("without_bn")

# Check both experiments completed 15,000 steps
for name, metrics in [
    ("with_bn", with_bn),
    ("without_bn", without_bn),
]:
    final_step = int(metrics["step"][-1])
    print(f"{name}: {final_step} steps, {len(metrics['step'])} measurements")

    if final_step != 15000:
        raise ValueError(f"{name} has not completed 15000 steps")


def plot_comparison(metric, ylabel, filename):
    plt.figure(figsize=(10, 6))

    plt.plot(
        with_bn["step"],
        with_bn[metric],
        label="With BatchNorm",
        color="royalblue",
        linewidth=2,
    )

    plt.plot(
        without_bn["step"],
        without_bn[metric],
        label="Without BatchNorm",
        color="darkorange",
        linewidth=2,
    )

    plt.xlabel("Training Steps")
    plt.ylabel(ylabel)
    plt.title(f"Batch Normalization Comparison - {ylabel}")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 15000)

    if "Accuracy" in ylabel:
        plt.ylim(0, 100)

    plt.tight_layout()

    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


plot_comparison(
    "train_loss",
    "Training Loss",
    "training_loss.png",
)

plot_comparison(
    "train_accuracy",
    "Training Accuracy (%)",
    "training_accuracy.png",
)

plot_comparison(
    "test_accuracy",
    "Test Accuracy (%)",
    "test_accuracy.png",
)

print("\nAll Figure 1 plots generated successfully!")
