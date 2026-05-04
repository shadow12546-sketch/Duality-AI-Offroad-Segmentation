import cv2
import numpy as np
import os
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
input_folder  = r"D:\JAVA\projects\project_gfg\Offroad_Segmentation_Scripts\predictions\masks"
output_folder = os.path.join(
    r"D:\JAVA\projects\project_gfg\Offroad_Segmentation_Scripts\predictions",
    "colorized"
)
os.makedirs(output_folder, exist_ok=True)

# ── class color map (class ID → BGR) ───────────────────────────────────────
COLOR_MAP = {
    0: (0,   0,   0),      # Background     - black
    1: (34,  139, 34),     # Trees          - forest green
    2: (0,   200, 100),    # Lush Bushes    - bright green
    3: (210, 180, 140),    # Dry Grass      - tan
    4: (139, 115, 85),     # Dry Bushes     - brown
    5: (128, 128, 128),    # Ground Clutter - gray
    6: (101, 67,  33),     # Logs           - dark brown
    7: (169, 169, 169),    # Rocks          - light gray
    8: (107, 142, 35),     # Landscape      - olive
    9: (135, 206, 235),    # Sky            - sky blue
}

CLASS_NAMES = {
    0: "Background",
    1: "Trees",
    2: "Lush Bushes",
    3: "Dry Grass",
    4: "Dry Bushes",
    5: "Ground Clutter",
    6: "Logs",
    7: "Rocks",
    8: "Landscape",
    9: "Sky",
}

# ── gather image files ──────────────────────────────────────────────────────
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
image_files = sorted(
    f for f in Path(input_folder).iterdir()
    if f.is_file() and f.suffix.lower() in IMAGE_EXTS
)

if not image_files:
    print(f"[ERROR] No image files found in:\n  {input_folder}")
    exit(1)

print(f"Found {len(image_files)} mask file(s) to colorize")
print(f"Output → {output_folder}\n")

# ── process ─────────────────────────────────────────────────────────────────
skipped   = 0
processed = 0
class_pixel_counts = {k: 0 for k in COLOR_MAP}

for image_file in image_files:
    im = cv2.imread(str(image_file), cv2.IMREAD_UNCHANGED)

    if im is None:
        print(f"  [SKIP] Could not read: {image_file.name}")
        skipped += 1
        continue

    # handle multi-channel masks — take first channel
    if im.ndim == 3:
        im = im[:, :, 0]

    h, w = im.shape
    colorized = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, bgr in COLOR_MAP.items():
        mask = (im == class_id)
        colorized[mask] = bgr
        class_pixel_counts[class_id] += int(mask.sum())

    # flag unknown class IDs in this mask
    unknown_ids = set(np.unique(im)) - set(COLOR_MAP.keys())
    if unknown_ids:
        print(f"  [WARN] Unknown class IDs {unknown_ids} in {image_file.name} — shown as black")

    output_path = os.path.join(output_folder, f"{image_file.stem}.png")
    cv2.imwrite(output_path, colorized)
    processed += 1

# ── summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*52}")
print(f"  Done!  Processed: {processed}   Skipped: {skipped}")
print(f"{'='*52}")
print(f"\nPixel distribution across all masks:")
total_pixels = sum(class_pixel_counts.values())
if total_pixels > 0:
    for class_id, count in class_pixel_counts.items():
        pct  = 100.0 * count / total_pixels
        bar  = "█" * int(pct / 2)
        name = CLASS_NAMES[class_id]
        print(f"  {class_id:>2}  {name:<15} {bar:<50} {pct:5.1f}%")

print(f"\nColorized masks saved to:\n  {output_folder}")