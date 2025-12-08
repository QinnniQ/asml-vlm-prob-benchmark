import json
from pathlib import Path
from PIL import Image

class WaferDataset:
    """
    Lightweight dataset loader for synthetic wafer imagery and annotations.
    """

    def __init__(self, annotations_path: str, images_dir: str):
        self.annotations_path = Path(annotations_path)
        self.images_dir = Path(images_dir)

        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotations not found: {self.annotations_path}")

        self.annotations = json.load(open(self.annotations_path, "r"))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_path = self.images_dir / ann["image"]

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        return {
            "image": image,
            "label": ann["label"],
            "prompt": ann["prompt"],
        }

    def get_classes(self):
        """Return sorted list of unique class labels."""
        return sorted(list({ann["label"] for ann in self.annotations}))
