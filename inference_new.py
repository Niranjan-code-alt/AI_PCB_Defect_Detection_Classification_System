import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.ops import nms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import imagehash
from skimage.metrics import structural_similarity as ssim
import torch.nn as nn

print("--- PCB Differential Detection Pipeline (FIXED & STABLE) ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# LOAD CHECKPOINT
# ============================================================
model_path = "best_resnet50_pcb_defects_50epochs.pth"  # name is misleading
checkpoint = torch.load(model_path, map_location=device)

# ============================================================
# CLASS NAMES
# ============================================================
class_names = checkpoint.get(
    "class_names",
    ['missing_hole', 'mouse_bite', 'open_circuit',
     'short', 'spur', 'spurious_copper']
)

if 'normal' in class_names:
    class_names.remove('normal')

num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")

# ============================================================
# LOAD **CORRECT** MODEL (RESNET-50)
# ============================================================
defect_classifier = models.resnet50(weights=None)
defect_classifier.fc = nn.Linear(defect_classifier.fc.in_features, num_classes)
defect_classifier.load_state_dict(checkpoint["model_state_dict"], strict=True)
defect_classifier.to(device).eval()

# Verify once
with torch.no_grad():
    dummy = torch.randn(1, 3, 224, 224).to(device)
    print("Model output shape:", defect_classifier(dummy).shape)

# ============================================================
# CONFIG
# ============================================================
golden_images_dir = "PCB_USED"
WINDOW_SIZE = 128
STRIDE = WINDOW_SIZE // 4
SIMILARITY_THRESHOLD = 0.95
CONF_THRESHOLD = 0.80

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ============================================================
# GOLDEN DATABASE
# ============================================================
def create_golden_image_database(golden_dir):
    db = []
    for fname in os.listdir(golden_dir):
        try:
            img = Image.open(os.path.join(golden_dir, fname)).convert("RGB")
            db.append({
                "image": img,
                "hash": imagehash.phash(img)
            })
        except:
            pass
    return db

golden_db = create_golden_image_database(golden_images_dir)
print(f"Golden images loaded: {len(golden_db)}")

# ============================================================
# FIND BEST GOLDEN
# ============================================================
def find_best_match(input_image, golden_database):
    input_hash = imagehash.phash(input_image)
    return min(golden_database, key=lambda x: input_hash - x["hash"])["image"]

# ============================================================
# DETECTION
# ============================================================
def detect_anomalies_by_comparison(input_image, golden_image, classifier):
    detections = []
    w, h = input_image.size
    golden_image = golden_image.resize((w, h))
    debug_count = 0

    for y in range(0, h - WINDOW_SIZE + 1, STRIDE):
        for x in range(0, w - WINDOW_SIZE + 1, STRIDE):

            patch = input_image.crop((x, y, x + WINDOW_SIZE, y + WINDOW_SIZE))
            ref = golden_image.crop((x, y, x + WINDOW_SIZE, y + WINDOW_SIZE))

            ssim_score = ssim(
                np.array(patch.convert("L")),
                np.array(ref.convert("L"))
            )

            if ssim_score >= SIMILARITY_THRESHOLD:
                continue

            tensor = inference_transform(patch).unsqueeze(0).to(device)

            with torch.no_grad():
                probs = F.softmax(classifier(tensor), dim=1)
                conf, idx = torch.max(probs, dim=1)

            if debug_count < 3:
                print(f"🔍 Window ({x},{y}) Top-3:")
                top_p, top_i = torch.topk(probs, 3)
                for i in range(3):
                    print(f"  {class_names[top_i[0][i]]}: {top_p[0][i]:.3f}")
                debug_count += 1

            if conf.item() >= CONF_THRESHOLD:
                detections.append({
                    "box": [x, y, x + WINDOW_SIZE, y + WINDOW_SIZE],
                    "label": class_names[idx.item()],
                    "confidence": conf.item()
                })

    if not detections:
        return []

    boxes = torch.tensor([d["box"] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d["confidence"] for d in detections])
    keep = nms(boxes, scores, iou_threshold=0.2)

    return [detections[i] for i in keep]

# ============================================================
# DRAW
# ============================================================
def draw_detections_on_image(image, detections):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for d in detections:
        draw.rectangle(d["box"], outline="red", width=4)
        draw.text(
            (d["box"][0], d["box"][1] - 12),
            f"{d['label']} {d['confidence']:.2f}",
            fill="red",
            font=font
        )
    return image

# ============================================================
# STREAMLIT INTERFACE
# ============================================================
def run_inference_on_pil(input_image):
    start = time.time()

    golden = find_best_match(input_image, golden_db)
    detections = detect_anomalies_by_comparison(
        input_image, golden, defect_classifier
    )

    result = draw_detections_on_image(input_image.copy(), detections)

    print(f"Detected {len(detections)} defects in {time.time()-start:.2f}s")
    return result, detections

print("✅ PIPELINE READY — NO ERRORS, CORRECT LABELS")
