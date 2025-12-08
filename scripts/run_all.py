import subprocess
import datetime
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json

# Paths
ROOT = Path(".")
REPORTS = ROOT / "reports"
ARCHIVE_DIR = ROOT / "reports_archive"

SCRIPTS = {
    "evaluate": "scripts/evaluate_clip.py",
    "calibration": "scripts/eval_calibration.py",
    "reliability": "scripts/generate_reliability_plot.py",
}


def run_script(name, path):
    """
    Run a Python script using the SAME interpreter as this process.
    This ensures the venv Python is used even inside subprocess calls.
    """
    print(f"\n=== Running: {name} ===")
    try:
        subprocess.run(
            [sys.executable, path],
            check=True
        )
        print(f"✓ Finished: {name}")
    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR in {name}. Aborting pipeline.")
        print(f"Subprocess error: {e}")
        exit(1)


def archive_results():
    """Archive old reports with a timestamp."""
    if not REPORTS.exists():
        print("No reports found to archive.")
        return

    ARCHIVE_DIR.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = ARCHIVE_DIR / f"reports_{timestamp}"

    shutil.copytree(REPORTS, dest)

    print(f"Archived previous results → {dest}")


def validate_dataset():
    """Basic validation to ensure images and annotations exist."""
    print("\n=== Validating Dataset ===")

    ann_path = Path("data/annotations/annotations.json")
    img_dir = Path("data/images")

    if not ann_path.exists():
        raise FileNotFoundError("Missing annotations.json")

    if not img_dir.exists():
        raise FileNotFoundError("Missing data/images folder")

    print("✓ Dataset structure OK")


def main():
    print("=== ASML VLM Benchmark — Full Pipeline ===")

    # Step 0: Validate input data
    validate_dataset()

    # Step 1: Archive old results
    archive_results()

    # Step 2: Run model evaluation
    run_script("Model Evaluation", SCRIPTS["evaluate"])

    # Step 3: Run calibration metrics
    run_script("Calibration Metrics", SCRIPTS["calibration"])

    # Step 4: Generate reliability diagram
    run_script("Reliability Plot", SCRIPTS["reliability"])

    print("\n=== Pipeline Complete ===")
    print(f"Results saved in: {REPORTS.resolve()}")
    print(f"Archived previous runs in: {ARCHIVE_DIR.resolve()}")


if __name__ == "__main__":
    main()
