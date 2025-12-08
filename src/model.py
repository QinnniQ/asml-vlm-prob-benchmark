import torch
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor


class CLIPWrapper:
    """
    Wrapper around HuggingFace CLIP for wafer defect classification.
    Ensures safe model loading using safetensors (no torch.load).
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        print("Loading CLIP model (safetensors only)...")

        # Device setup
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load processor
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Load model WITHOUT using torch.load
        self.model = CLIPModel.from_pretrained(
            model_name,
            use_safetensors=True,      # <— Forces safe checkpoint loading
            local_files_only=False,
            trust_remote_code=False
        ).to(self.device)

        self.model.eval()

    def predict(self, image, class_names):
        """
        Run CLIP zero-shot classification.
        image: PIL.Image
        class_names: list of str
        Returns:
            probs: torch.tensor of shape (num_classes,)
        """

        prompts = [f"a photo of {label}" for label in class_names]

        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

            # CLIP similarity scores → logits_per_image
            logits = outputs.logits_per_image  # shape: (1, num_classes)
            probs = logits.softmax(dim=-1).squeeze(0)

        return probs.cpu()
