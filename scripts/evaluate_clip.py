import json
import os
import math
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

ANNOT_PATH = "data/annotations/annotations.json"
IMAGE_DIR = "data/images"

# Classification labels
CLASSES = ["none", "scratch", "particle"]


def load_dataset():
    with open(ANNOT_PATH, "r") as f:
        return json.load(f)


def compute_uncertainty_metrics(probs, true_label):
    """
    Computes:
    - confidence: max probability
    - entropy: sum of -p log p
    - brier score: squared error between prob vector and one-hot truth
    """

    confidence = max(probs)

    # entropy
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)

    # brier score
    true_vec = [1.0 if CLASSES[i] == true_label else 0.0 for i in range(len(CLASSES))]
    brier = sum((probs[i] - true_vec[i]) ** 2 for i in range(len(CLASSES)))

    return confidence, entropy, brier


def predict_image(model, processor, image_path):
    # Prompt templates for CLIP
    text_inputs = [f"This image contains a {cls} defect." for cls in CLASSES]

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=text_inputs,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits_per_image[0]
    probs = logits.softmax(dim=0)

    return probs.cpu().numpy().tolist()


def main():
    print("DEBUG: evaluate_clip.py started")
    print("Loading CLIP model (using safetensors only)...")

    # Force safetensors-only loading to bypass CVE restrictions
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        use_safetensors=True
    )

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    dataset = load_dataset()
    print(f"DEBUG: Loaded {len(dataset)} annotations")

    results = []
    os.makedirs("reports", exist_ok=True)

    for ann in dataset:
        print(f"DEBUG: Processing {ann['image']}")
        img_path = os.path.join(IMAGE_DIR, ann["image"])

        probs = predict_image(model, processor, img_path)

        # predicted class
        pred_idx = int(torch.tensor(probs).argmax())
        pred_label = CLASSES[pred_idx]

        # compute probabilistic metrics
        confidence, entropy, brier = compute_uncertainty_metrics(probs, ann["label"])

        # save structured results
        result = {
            "image": ann["image"],
            "true_label": ann["label"],
            "pred_label": pred_label,
            "probabilities": {
                CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))
            },
            "confidence": float(confidence),
            "entropy": float(entropy),
            "brier_score": float(brier)
        }

        results.append(result)

        print(
            f"{ann['image']} | true={ann['label']} | pred={pred_label} "
            f"| confidence={confidence:.3f} | entropy={entropy:.3f} | brier={brier:.3f}"
        )

    # save all results
    out_path = "reports/clip_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation complete! Results saved to {out_path}")


if __name__ == "__main__":
    main()
