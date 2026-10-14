import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import numpy as np

RESULTS_PATH = ROOT / "reports/clip_results.json"
CALIB_REPORT_PATH = ROOT / "reports/calibration_report.json"

def compute_ece(confidences, accuracies, n_bins=10):
    """Expected Calibration Error (ECE)."""
    confidences, accuracies = _validate_inputs(confidences, accuracies, n_bins)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bins[i], bins[i + 1]
        in_bin = (confidences >= bin_lower) & (confidences <= bin_upper if i == n_bins - 1 else confidences < bin_upper)

        if in_bin.sum() == 0:
            continue

        bin_conf = confidences[in_bin].mean()
        bin_acc = accuracies[in_bin].mean()
        bin_prob = in_bin.mean()

        ece += np.abs(bin_acc - bin_conf) * bin_prob

    return float(ece)


def compute_mce(confidences, accuracies, n_bins=10):
    """Maximum Calibration Error (MCE)."""
    confidences, accuracies = _validate_inputs(confidences, accuracies, n_bins)
    bins = np.linspace(0, 1, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bins[i], bins[i + 1]
        in_bin = (confidences >= bin_lower) & (confidences <= bin_upper if i == n_bins - 1 else confidences < bin_upper)

        if in_bin.sum() == 0:
            continue

        bin_conf = confidences[in_bin].mean()
        bin_acc = accuracies[in_bin].mean()
        mce = max(mce, abs(bin_acc - bin_conf))

    return float(mce)


def _validate_inputs(confidences, accuracies, n_bins):
    confidences = np.asarray(confidences, dtype=float)
    accuracies = np.asarray(accuracies, dtype=float)
    if n_bins < 1 or confidences.ndim != 1 or accuracies.shape != confidences.shape or len(confidences) == 0:
        raise ValueError("Expected nonempty, equally sized 1D arrays and at least one bin")
    if not np.all(np.isfinite(confidences)) or not np.all(np.isfinite(accuracies)):
        raise ValueError("Confidence and accuracy values must be finite")
    if np.any((confidences < 0) | (confidences > 1)) or np.any((accuracies < 0) | (accuracies > 1)):
        raise ValueError("Confidence and accuracy values must be between 0 and 1")
    return confidences, accuracies


def main():
    print("=== ASML VLM Benchmark — Calibration Evaluation ===")

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing results file: {RESULTS_PATH}")

    with RESULTS_PATH.open(encoding="utf-8") as source:
        results = json.load(source)

    # Extract confidences and correctness
    confidences = np.array([r["confidence"] for r in results])
    accuracies = np.array([r["pred_label"] == r["true_label"] for r in results], dtype=float)

    # Compute calibration metrics
    ece = compute_ece(confidences, accuracies, n_bins=10)
    mce = compute_mce(confidences, accuracies, n_bins=10)

    print(f"ECE  (Expected Calibration Error): {ece:.4f}")
    print(f"MCE  (Maximum Calibration Error): {mce:.4f}")

    # Save JSON report
    report = {
        "ece": ece,
        "mce": mce,
        "num_samples": len(results),
        "confidence_mean": float(confidences.mean()),
        "accuracy_mean": float(accuracies.mean()),
    }

    CALIB_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIB_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved calibration report → {CALIB_REPORT_PATH.absolute()}")
    print("Calibration evaluation complete.")


if __name__ == "__main__":
    main()
