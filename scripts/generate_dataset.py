from PIL import Image, ImageDraw
import random
import json
import os

DATA_DIR = "data/images"
ANNOT_DIR = "data/annotations"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ANNOT_DIR, exist_ok=True)

def generate_blank_wafer():
    img = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse([(20, 20), (236, 236)], outline="black", width=3)
    return img, draw

def add_scratch(draw):
    x1, y1 = random.randint(30, 200), random.randint(30, 200)
    x2, y2 = x1 + random.randint(20, 40), y1 + random.randint(5, 15)
    draw.line((x1, y1, x2, y2), fill="red", width=2)
    return "scratch"

def add_particle(draw):
    x, y = random.randint(40, 200), random.randint(40, 200)
    r = random.randint(5, 10)
    draw.ellipse([(x-r, y-r), (x+r, y+r)], fill="blue")
    return "particle"

def add_alignment_mark(draw):
    # simple cross mark
    x, y = 128, 128
    draw.line((x - 10, y, x + 10, y), fill="black", width=3)
    draw.line((x, y - 10, x, y + 10), fill="black", width=3)
    return "alignment_mark"

def generate_image(idx):
    img, draw = generate_blank_wafer()
    defect_type = random.choice(["scratch", "particle", "none"])

    if defect_type == "scratch":
        label = add_scratch(draw)
    elif defect_type == "particle":
        label = add_particle(draw)
    else:
        label = "none"

    filename = f"wafer_{idx}.png"
    img.save(os.path.join(DATA_DIR, filename))

    annotation = {
        "image": filename,
        "task": "defect_classification",
        "label": label,
        "prompt": "Identify the defect type in the wafer image."
    }

    return annotation

def main():
    annotations = []
    for i in range(50):  # generate 50 images to start
        ann = generate_image(i)
        annotations.append(ann)

    with open(os.path.join(ANNOT_DIR, "annotations.json"), "w") as f:
        json.dump(annotations, f, indent=2)

    print("Dataset generated successfully!")

if __name__ == "__main__":
    main()
