import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS_PATH = Path("reports/clip_results.json")
OUTPUT_PLOT = Path("reports/reliability_diagram.png")


def compute_reliability_bins(confidences, accuracies, n_bins=10):
    """
    Returns:
       bin_centers: midpoint of each bin
       bin_conf: average confidence per bin
       bin_acc: accuracy per bin
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1

    bin_conf, bin_acc, bin_counts = [], [], []

    for i in range(n_bins):
        in_bin = bin_indices == i

        if np.sum(in_bin) == 0:
            bin_conf.append(0)
            bin_acc.append(0)
            bin_counts.append(0)
            continue

        bin_conf.append(confidences[in_bin].mean())
        bin_acc.append(accuracies[in_bin].mean())
        bin_counts.append(np.sum(in_bin))

    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers, np.array(bin_conf), np.array(bin_acc), np.array(bin_counts)


def plot_reliability(bin_centers, bin_conf, bin_acc):
    plt.figure(figsize=(7, 7))

    # Perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

    # Reliability curve
    plt.plot(bin_conf, bin_acc, marker="o", linewidth=2, label="Model Calibration")

    # Bars showing bin coverage
    plt.bar(
        bin_centers,
        bin_acc - 0,
        width=0.09,
        alpha=0.1,
        color="blue",
        label="Bin Accuracy",
    )

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram (Calibration Curve)")
    plt.legend()
    plt.grid(alpha=0.3)

    OUTPUT_PLOT.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(OUTPUT_PLOT, dpi=300)
    plt.close()


def main():
    print("=== Generating Reliability Diagram ===")

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing results file: {RESULTS_PATH}")

    results = json.load(open(RESULTS_PATH, "r"))

    confidences = np.array([r["confidence"] for r in results])
    accuracies = np.array([r["pred_label"] == r["true_label"] for r in results], dtype=float)

    bin_centers, bin_conf, bin_acc, bin_counts = compute_reliability_bins(confidences, accuracies)

    print("Bin centers:   ", bin_centers)
    print("Bin conf avg:  ", bin_conf)
    print("Bin accuracy:  ", bin_acc)
    print("Bin counts:    ", bin_counts)

    plot_reliability(bin_centers, bin_conf, bin_acc)

    print(f"\nSaved reliability diagram → {OUTPUT_PLOT.absolute()}")
    print("Done.")


if __name__ == "__main__":
    main()
