import sys
from pathlib import Path

# Add project root for src imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
from tqdm import tqdm

from src.dataset import WaferDataset
from src.model import CLIPWrapper
from src.metrics import compute_confidence, compute_entropy, compute_brier_score


def main():
    print("=== ASML VLM Benchmark — CLIP Evaluation ===")

    # Paths
    ann_path = ROOT / "data" / "annotations" / "annotations.json"
    img_dir = ROOT / "data" / "images"
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    # Load dataset
    dataset = WaferDataset(ann_path, img_dir)

    class_names = sorted(list({ann["label"] for ann in dataset.annotations}))
    print(f"Loaded {len(dataset)} samples across {len(class_names)} classes: {class_names}")

    # Load model
    model = CLIPWrapper()
    print("Model loaded successfully.\n")

    results = []

    for i, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        image = sample["image"]
        true_label = sample["label"]

        # Retrieve filename from annotations (Ground truth)
        ann = dataset.annotations[i]
        filename = ann["image"]

        # Run model
        probs = model.predict(image, class_names)

        # Metrics
        confidence = compute_confidence(probs)
        entropy = compute_entropy(probs)
        brier = compute_brier_score(probs, class_names.index(true_label))

        pred_label = class_names[int(probs.argmax())]

        result = {
            "image": filename,
            "true_label": true_label,
            "pred_label": pred_label,
            "probabilities": {class_names[j]: float(probs[j]) for j in range(len(class_names))},
            "confidence": confidence,
            "entropy": entropy,
            "brier": brier
        }

        results.append(result)

    # Save results
    out_file = reports_dir / "clip_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results → {out_file}")


if __name__ == "__main__":
    main()
