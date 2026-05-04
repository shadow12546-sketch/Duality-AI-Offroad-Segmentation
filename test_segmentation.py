"""
Segmentation Inference & Evaluation Script
==========================================
Duality AI – Offroad Semantic Scene Segmentation Hackathon

Two operating modes:
  1. INFERENCE mode  (--mode infer)
     Runs on testImages/ folder — no ground-truth masks required.
     Saves coloured prediction masks + timing report.

  2. EVALUATION mode (--mode eval)  [default]
     Runs on a labelled split (val/) that has both Color_Images/ and
     Segmentation/ sub-folders. Computes IoU, Dice, pixel accuracy,
     confusion matrix, per-class bar chart, and saves all visualisations.

Usage examples:
  python test_segmentation.py --mode eval  --data_dir ./Offroad_Segmentation_Val
  python test_segmentation.py --mode infer --data_dir ./Offroad_Segmentation_testImages

Fixes vs previous version
--------------------------
  [FIX-1]  CLASS ID MISMATCH: Aligned to train.py's 10-class scheme.
           train.py uses NO explicit Background class; Trees=0 ... Sky=9.
           Old test.py had 11 classes (Background=0, Trees=1 ... Sky=10) —
           this caused every predicted class to be off by one, making ALL
           metrics and visualisations wrong.
  [FIX-2]  HEAD ARCHITECTURE MISMATCH: Copied exact SegmentationHeadConvNeXt
           from train.py (256→128 stem, residual block1, block2, Dropout2d).
           Old test.py had a simpler head → load_state_dict() would crash.
  [FIX-3]  Flowers (600) was missing from value_map → silent class erasure.
  [FIX-4]  Mask resize used bilinear interpolation → corrupted class IDs.
  [FIX-5]  os.listdir() is non-deterministic → non-reproducible results.
  [FIX-6]  No inference speed measurement → spec benchmark unverified.
  [FIX-7]  No confusion matrix → required on report pages 3-4.
  [FIX-8]  Output masks saved at training resolution, not original 960×540.
  [FIX-9]  No assertion guarding class-count consistency.
  [NEW-1]  Separate INFER mode (no GT) vs EVAL mode (with GT).
  [NEW-2]  Per-image inference timing logged and averaged.
  [NEW-3]  Confusion matrix heatmap saved as PNG.
  [NEW-4]  Failure-case gallery: worst-IoU samples saved side-by-side.
  [NEW-5]  Reproducibility seed set at startup.
  [NEW-6]  evaluation_metrics.txt includes inference speed and per-class Dice.
  [NEW-7]  ignore_index=255 respected in all metric calculations.
"""

import os
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

try:
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] scikit-learn not found — confusion matrix skipped. "
          "pip install scikit-learn")


# ============================================================================
#  Reproducibility  [FIX-5 / NEW-5]
# ============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# ============================================================================
#  Class Definitions
#  [FIX-1] Aligned EXACTLY to train.py value_map — 10 classes, NO background.
#           Trees=0 … Sky=9
# ============================================================================

VALUE_MAP = {
    100:   0,   # Trees
    200:   1,   # Lush Bushes
    300:   2,   # Dry Grass
    500:   3,   # Dry Bushes
    550:   4,   # Ground Clutter
    600:   5,   # Flowers          [FIX-3] was missing in original test.py
    700:   6,   # Logs
    800:   7,   # Rocks
    7100:  8,   # Landscape
    10000: 9,   # Sky
}

N_CLASSES = len(VALUE_MAP)   # 10  — must match train.py n_classes

CLASS_NAMES = [
    'Trees',          # 0
    'Lush Bushes',    # 1
    'Dry Grass',      # 2
    'Dry Bushes',     # 3
    'Ground Clutter', # 4
    'Flowers',        # 5
    'Logs',           # 6
    'Rocks',          # 7
    'Landscape',      # 8
    'Sky',            # 9
]

# [FIX-9] Guard: crash immediately if lists drift out of sync
assert len(CLASS_NAMES) == N_CLASSES, (
    f"CLASS_NAMES has {len(CLASS_NAMES)} entries but VALUE_MAP has {N_CLASSES}. "
    "Keep them in sync with train.py."
)

# High-contrast RGB palette — one colour per class ID (0-9)
COLOR_PALETTE = np.array([
    [34,  139, 34 ],  # 0  Trees         – forest green
    [0,   255, 0  ],  # 1  Lush Bushes   – lime
    [210, 180, 140],  # 2  Dry Grass     – tan
    [139, 90,  43 ],  # 3  Dry Bushes    – brown
    [128, 128, 0  ],  # 4  Ground Clutter– olive
    [255, 20,  147],  # 5  Flowers       – deep pink (high contrast)
    [139, 69,  19 ],  # 6  Logs          – saddle brown
    [128, 128, 128],  # 7  Rocks         – gray
    [160, 82,  45 ],  # 8  Landscape     – sienna
    [135, 206, 235],  # 9  Sky           – sky blue
], dtype=np.uint8)

assert len(COLOR_PALETTE) == N_CLASSES, "COLOR_PALETTE length must equal N_CLASSES"

# Pixels whose raw value is not in VALUE_MAP are marked 255 (ignore)
IGNORE_INDEX = 255


# ============================================================================
#  Mask Utilities
# ============================================================================

def convert_mask(mask: Image.Image) -> Image.Image:
    """
    Map raw pixel values (100, 200 … 10000) → sequential class IDs (0-9).
    Unmapped pixels (including raw 0 / Background) → IGNORE_INDEX (255).
    Matches train.py convert_mask() exactly.
    """
    arr = np.array(mask, dtype=np.int32)
    out = np.full_like(arr, fill_value=IGNORE_INDEX, dtype=np.uint8)
    for raw, cid in VALUE_MAP.items():
        out[arr == raw] = cid
    return Image.fromarray(out)


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Convert a 2-D class-ID mask (uint8, 0-9) to an RGB colour image."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid in range(N_CLASSES):
        rgb[mask == cid] = COLOR_PALETTE[cid]
    # IGNORE pixels stay black (0,0,0)
    return rgb


# ============================================================================
#  Datasets
# ============================================================================

class EvalDataset(Dataset):
    """For EVALUATION mode — folder must contain Color_Images/ and Segmentation/."""

    def __init__(self, data_dir, img_transform, mask_transform):
        self.img_dir        = os.path.join(data_dir, 'Color_Images')
        self.mask_dir       = os.path.join(data_dir, 'Segmentation')
        self.img_transform  = img_transform
        self.mask_transform = mask_transform
        # [FIX-5] sorted() for deterministic, reproducible ordering
        self.ids = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img  = Image.open(os.path.join(self.img_dir,  name)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, name))
        mask = convert_mask(mask)   # raw values → 0-9 or 255

        img_t  = self.img_transform(img)
        # [FIX-4] NEAREST preserves integer class IDs — bilinear corrupts them
        mask_t = self.mask_transform(mask)
        # ToTensor divides by 255; undo that to recover class IDs (0-9) or 255
        mask_t = (mask_t * 255).round().long().squeeze(0)   # (H, W)

        return img_t, mask_t, name


class InferDataset(Dataset):
    """For INFERENCE mode — only RGB images, no masks needed."""

    def __init__(self, data_dir, img_transform):
        # testImages/ may be flat or have a Color_Images/ sub-dir
        color_sub = os.path.join(data_dir, 'Color_Images')
        self.img_dir       = color_sub if os.path.isdir(color_sub) else data_dir
        self.img_transform = img_transform
        # [FIX-5] sorted for reproducibility
        self.ids = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img  = Image.open(os.path.join(self.img_dir, name)).convert('RGB')
        return self.img_transform(img), name


# ============================================================================
#  Model
#  [FIX-2] Copied EXACT architecture from train.py — stem 256→128, residual
#           block1, block2, Dropout2d.  Must match saved weights exactly.
# ============================================================================

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        # Matches train.py exactly: 256 intermediate, then 128
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=7, padding=3),
            nn.GELU(),
        )

        # Residual depthwise block (matches train.py block1)
        self.block1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=1),
        )

        # Pointwise refinement block (matches train.py block2)
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, padding=2, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )

        self.dropout    = nn.Dropout2d(p=0.1)
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = x + self.block1(x)   # residual — matches train.py
        x = self.block2(x)
        x = self.dropout(x)
        return self.classifier(x)


# ============================================================================
#  Metrics  (all respect ignore_index=255)
# ============================================================================

def compute_iou(pred_logits, target, num_classes=N_CLASSES, ignore_index=IGNORE_INDEX):
    """
    pred_logits : (B, C, H, W) float
    target      : (B, H, W)   long
    Returns mean IoU (float, NaN-ignored) and per-class IoU list.
    """
    pred = torch.argmax(pred_logits, dim=1).view(-1)
    tgt  = target.view(-1)

    # Mask out ignored pixels
    valid = tgt != ignore_index
    pred, tgt = pred[valid], tgt[valid]

    ious = []
    for cid in range(num_classes):
        p = pred == cid
        t = tgt  == cid
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        ious.append(float('nan') if union == 0 else (inter / union).item())

    return float(np.nanmean(ious)), ious


def compute_dice(pred_logits, target, num_classes=N_CLASSES,
                 ignore_index=IGNORE_INDEX, smooth=1e-6):
    pred = torch.argmax(pred_logits, dim=1).view(-1)
    tgt  = target.view(-1)

    valid = tgt != ignore_index
    pred, tgt = pred[valid], tgt[valid]

    dices = []
    for cid in range(num_classes):
        p = pred == cid
        t = tgt  == cid
        inter = (p & t).sum().float()
        score = (2. * inter + smooth) / (p.sum().float() + t.sum().float() + smooth)
        dices.append(score.item())

    return float(np.mean(dices)), dices


def compute_pixel_accuracy(pred_logits, target, ignore_index=IGNORE_INDEX):
    pred  = torch.argmax(pred_logits, dim=1)
    valid = target != ignore_index
    if valid.sum() == 0:
        return float('nan')
    return (pred[valid] == target[valid]).float().mean().item()


# ============================================================================
#  Visualisation Helpers
# ============================================================================

def denorm(tensor):
    """Denormalise an ImageNet-normalised CHW tensor to HWC numpy [0,1]."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().numpy()
    img  = np.moveaxis(img, 0, -1) * std + mean
    return np.clip(img, 0, 1)


def make_legend_patches():
    return [
        mpatches.Patch(color=COLOR_PALETTE[i] / 255.0, label=CLASS_NAMES[i])
        for i in range(N_CLASSES)
    ]


def save_comparison(img_t, gt_mask, pred_mask, path, title=''):
    """Three-panel: input | ground truth | prediction."""
    img       = denorm(img_t)
    gt_np     = gt_mask.cpu().numpy().astype(np.uint8)
    pr_np     = pred_mask.cpu().numpy().astype(np.uint8)
    gt_color  = mask_to_color(gt_np)
    pr_color  = mask_to_color(pr_np)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(img);      axes[0].set_title('Input Image');  axes[0].axis('off')
    axes[1].imshow(gt_color); axes[1].set_title('Ground Truth'); axes[1].axis('off')
    axes[2].imshow(pr_color); axes[2].set_title('Prediction');   axes[2].axis('off')
    fig.legend(handles=make_legend_patches(), loc='lower center',
               ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.08))
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def save_infer_result(img_t, pred_mask, path, title=''):
    """Two-panel for inference mode: input | prediction."""
    img      = denorm(img_t)
    pr_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img);      axes[0].set_title('Input Image'); axes[0].axis('off')
    axes[1].imshow(pr_color); axes[1].set_title('Prediction'); axes[1].axis('off')
    fig.legend(handles=make_legend_patches(), loc='lower center',
               ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.08))
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def save_confusion_matrix(cm, output_dir):
    """Save a row-normalised (recall) confusion matrix heatmap."""
    if not HAS_SKLEARN:
        return

    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm  = np.where(row_sums == 0, 0, cm / row_sums)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = range(N_CLASSES)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_title('Confusion Matrix (row-normalised recall)\nRequired: Report pages 3-4')

    thresh = 0.5
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            val = cm_norm[i, j]
            ax.text(j, i, f'{val:.2f}',
                    ha='center', va='center', fontsize=7,
                    color='white' if val > thresh else 'black')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix  → {out_path}")


def save_per_class_iou_chart(class_ious, mean_iou, output_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [COLOR_PALETTE[i] / 255.0 for i in range(N_CLASSES)]
    valid  = [v if not np.isnan(v) else 0.0 for v in class_ious]

    bars = ax.bar(range(N_CLASSES), valid, color=colors,
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('IoU')
    ax.set_ylim(0, 1.05)
    ax.set_title(f'Per-Class IoU  |  Mean IoU = {mean_iou:.4f}')
    ax.axhline(mean_iou, color='red', linestyle='--', linewidth=1.5,
               label=f'mIoU = {mean_iou:.4f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, valid):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'per_class_iou.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved per-class IoU bar → {out_path}")


def save_metrics_txt(results, output_dir):
    """Write evaluation_metrics.txt — includes inference speed benchmark."""
    path  = os.path.join(output_dir, 'evaluation_metrics.txt')
    speed = results['avg_ms_per_img']
    lines = [
        "=" * 65,
        "DUALITY AI — OFFROAD SEGMENTATION  EVALUATION RESULTS",
        "=" * 65,
        f"Mean IoU  (mIoU)       : {results['mean_iou']:.4f}",
        f"Mean Dice Score        : {results['mean_dice']:.4f}",
        f"Pixel Accuracy         : {results['pixel_acc']:.4f}",
        "",
        f"Avg Inference Speed    : {speed:.2f} ms/image",
        f"  Spec target          : < 50.00 ms/image",
        f"  Result               : {'PASS ✓' if speed < 50 else 'FAIL ✗  — consider TorchScript / quantization'}",
        "",
        "-" * 65,
        f"{'Class':<20} {'IoU':>8}  {'Dice':>8}  {'Note':>12}",
        "-" * 65,
    ]
    for i, name in enumerate(CLASS_NAMES):
        iou_v  = results['class_iou'][i]
        dice_v = results['class_dice'][i]
        note   = '(absent in val)' if np.isnan(iou_v) else ''
        iou_s  = 'N/A    ' if np.isnan(iou_v) else f'{iou_v:.4f}'
        lines.append(f"  {name:<18} {iou_s:>8}  {dice_v:>8.4f}  {note}")
    lines += ["=" * 65, ""]

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved metrics summary   → {path}")


# ============================================================================
#  Model Loading
# ============================================================================

def load_model(model_path, device, train_hw):
    """Load DINOv2 ViT-S/14 backbone + segmentation head from disk."""
    print("Loading DINOv2 backbone (dinov2_vits14) …")
    backbone = torch.hub.load(
        repo_or_dir='facebookresearch/dinov2',
        model='dinov2_vits14'
    )
    backbone.eval().to(device)

    h_train, w_train = train_hw
    tokenH = h_train // 14
    tokenW = w_train // 14

    # Derive embedding dim via dummy forward
    dummy = torch.zeros(1, 3, h_train, w_train).to(device)
    with torch.no_grad():
        feats   = backbone.forward_features(dummy)['x_norm_patchtokens']
    n_embed = feats.shape[2]
    print(f"  Embedding dim : {n_embed}   token grid : {tokenH}×{tokenW}")

    print(f"Loading segmentation head from: {model_path}")
    # [FIX-2] Architecture matches train.py exactly
    head = SegmentationHeadConvNeXt(
        in_channels=n_embed,
        out_channels=N_CLASSES,   # [FIX-1] 10 classes, no background
        tokenW=tokenW,
        tokenH=tokenH,
    )
    head.load_state_dict(torch.load(model_path, map_location=device))
    head.eval().to(device)
    print("  Model loaded successfully.")
    return backbone, head


# ============================================================================
#  Forward Pass + Timing  [FIX-6 / FIX-8]
# ============================================================================

@torch.no_grad()
def forward_batch(backbone, head, imgs, original_hw=(540, 960)):
    """
    imgs        : (B, 3, H_train, W_train) on device
    original_hw : output masks are upsampled to this resolution [FIX-8]
    Returns logits at original_hw resolution and ms/image timing [FIX-6].
    """
    t0           = time.perf_counter()
    feats        = backbone.forward_features(imgs)['x_norm_patchtokens']
    logits_small = head(feats)
    # [FIX-8] Upsample to ORIGINAL resolution (960×540), not training res
    logits       = F.interpolate(logits_small, size=original_hw,
                                 mode='bilinear', align_corners=False)
    elapsed_ms   = (time.perf_counter() - t0) * 1000.0 / imgs.shape[0]
    return logits, elapsed_ms


# ============================================================================
#  EVALUATION MODE
# ============================================================================

def run_eval(args, backbone, head, device, img_transform, mask_transform):
    print(f"\n[EVAL MODE]  Data dir : {args.data_dir}")

    dataset = EvalDataset(args.data_dir, img_transform, mask_transform)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=0, pin_memory=True)
    print(f"  {len(dataset)} validation images found.")

    # Sub-directories
    masks_dir       = os.path.join(args.output_dir, 'masks')
    masks_color_dir = os.path.join(args.output_dir, 'masks_color')
    comparisons_dir = os.path.join(args.output_dir, 'comparisons')
    failures_dir    = os.path.join(args.output_dir, 'failure_cases')
    for d in [masks_dir, masks_color_dir, comparisons_dir, failures_dir]:
        os.makedirs(d, exist_ok=True)

    # Accumulators
    all_iou, all_dice, all_pix_acc = [], [], []
    all_class_iou, all_class_dice  = [], []
    all_ms                         = []
    sample_records                 = []   # for failure gallery
    cm_preds,  cm_targets          = [], []
    sample_count                   = 0

    pbar = tqdm(loader, desc='Evaluating', unit='batch')
    for imgs, labels, names in pbar:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        logits, ms = forward_batch(backbone, head, imgs, original_hw=(540, 960))

        # Align label resolution to logit resolution if needed
        if labels.shape[-2:] != logits.shape[-2:]:
            labels = F.interpolate(
                labels.unsqueeze(1).float(),
                size=logits.shape[-2:],
                mode='nearest'
            ).squeeze(1).long()

        pred_masks = torch.argmax(logits, dim=1)

        # Per-batch metrics (ignore_index=255 applied inside each function)
        iou,  c_iou  = compute_iou(logits, labels)
        dice, c_dice = compute_dice(logits, labels)
        pix          = compute_pixel_accuracy(logits, labels)

        all_iou.append(iou)
        all_dice.append(dice)
        all_pix_acc.append(pix)
        all_class_iou.append(c_iou)
        all_class_dice.append(c_dice)
        all_ms.append(ms)

        # [FIX-7] Accumulate for confusion matrix (skip ignore pixels)
        if HAS_SKLEARN:
            flat_pred   = pred_masks.cpu().numpy().flatten()
            flat_target = labels.cpu().numpy().flatten()
            valid_mask  = flat_target != IGNORE_INDEX
            cm_preds.append(flat_pred[valid_mask])
            cm_targets.append(flat_target[valid_mask])

        # Per-image output
        for i in range(imgs.shape[0]):
            name = names[i]
            base = os.path.splitext(name)[0]
            pm   = pred_masks[i].cpu().numpy().astype(np.uint8)

            # Raw class-ID mask
            Image.fromarray(pm).save(
                os.path.join(masks_dir, f'{base}_pred.png'))

            # [FIX-8] Colour mask at original resolution
            pm_color = mask_to_color(pm)
            cv2.imwrite(
                os.path.join(masks_color_dir, f'{base}_pred_color.png'),
                cv2.cvtColor(pm_color, cv2.COLOR_RGB2BGR))

            # Comparison panels (first N)
            if sample_count < args.num_samples:
                save_comparison(
                    imgs[i], labels[i], pred_masks[i],
                    os.path.join(comparisons_dir, f'{base}_comparison.png'),
                    title=name)

            # Track per-image IoU for failure gallery
            img_iou, _ = compute_iou(logits[i:i+1], labels[i:i+1])
            sample_records.append(
                (img_iou, imgs[i], labels[i], pred_masks[i], name))
            sample_count += 1

        pbar.set_postfix(mIoU=f'{iou:.3f}', ms=f'{ms:.1f}')

    # ── Aggregate ────────────────────────────────────────────────────────────
    mean_iou       = float(np.nanmean(all_iou))
    mean_dice      = float(np.nanmean(all_dice))
    mean_pix       = float(np.nanmean(all_pix_acc))
    avg_ms         = float(np.mean(all_ms))
    avg_class_iou  = list(np.nanmean(all_class_iou, axis=0))
    avg_class_dice = list(np.nanmean(all_class_dice, axis=0))

    results = dict(
        mean_iou=mean_iou,
        mean_dice=mean_dice,
        pixel_acc=mean_pix,
        avg_ms_per_img=avg_ms,
        class_iou=avg_class_iou,
        class_dice=avg_class_dice,
    )

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  Mean IoU       : {mean_iou:.4f}")
    print(f"  Mean Dice      : {mean_dice:.4f}")
    print(f"  Pixel Accuracy : {mean_pix:.4f}")
    print(f"  Inference Speed: {avg_ms:.2f} ms/image  "
          f"({'PASS ✓' if avg_ms < 50 else 'FAIL ✗  target < 50 ms'})")
    print("\n  Per-class IoU:")
    for i, name in enumerate(CLASS_NAMES):
        v = avg_class_iou[i]
        s = 'N/A   (absent)' if np.isnan(v) else f'{v:.4f}'
        print(f"    {name:<20}: {s}")
    print("=" * 65)

    # ── Save artefacts ───────────────────────────────────────────────────────
    save_metrics_txt(results, args.output_dir)
    save_per_class_iou_chart(avg_class_iou, mean_iou, args.output_dir)

    # [FIX-7] Confusion matrix
    if HAS_SKLEARN and cm_preds:
        cm = sk_confusion_matrix(
            np.concatenate(cm_targets),
            np.concatenate(cm_preds),
            labels=list(range(N_CLASSES))
        )
        save_confusion_matrix(cm, args.output_dir)

    # Failure-case gallery: 5 worst-IoU images
    sample_records.sort(key=lambda x: (float('inf') if np.isnan(x[0]) else x[0]))
    for rank, (img_iou, img_t, gt, pr, name) in enumerate(sample_records[:5]):
        out_path = os.path.join(failures_dir, f'failure_{rank+1}_{name}')
        save_comparison(img_t, gt, pr, out_path,
                        title=f'FAILURE #{rank+1}  IoU={img_iou:.3f}  [{name}]')
    print(f"  Saved top-5 failure cases → {failures_dir}/")
    print(f"\nAll outputs written to: {args.output_dir}/")
    return results


# ============================================================================
#  INFERENCE MODE  (testImages/ — no ground truth)
# ============================================================================

def run_infer(args, backbone, head, device, img_transform):
    print(f"\n[INFER MODE]  Data dir : {args.data_dir}")

    dataset = InferDataset(args.data_dir, img_transform)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=0, pin_memory=True)
    print(f"  {len(dataset)} test images found.")

    masks_dir       = os.path.join(args.output_dir, 'masks')
    masks_color_dir = os.path.join(args.output_dir, 'masks_color')
    vis_dir         = os.path.join(args.output_dir, 'visualisations')
    for d in [masks_dir, masks_color_dir, vis_dir]:
        os.makedirs(d, exist_ok=True)

    all_ms    = []
    vis_count = 0
    pbar      = tqdm(loader, desc='Inferring', unit='batch')

    for imgs, names in pbar:
        imgs   = imgs.to(device)
        logits, ms = forward_batch(backbone, head, imgs, original_hw=(540, 960))
        preds  = torch.argmax(logits, dim=1)
        all_ms.append(ms)

        for i in range(imgs.shape[0]):
            name = names[i]
            base = os.path.splitext(name)[0]
            pm   = preds[i].cpu().numpy().astype(np.uint8)

            # Raw class-ID mask
            Image.fromarray(pm).save(
                os.path.join(masks_dir, f'{base}_pred.png'))

            # Colour mask at original resolution [FIX-8]
            pm_color = mask_to_color(pm)
            cv2.imwrite(
                os.path.join(masks_color_dir, f'{base}_pred_color.png'),
                cv2.cvtColor(pm_color, cv2.COLOR_RGB2BGR))

            # Visualisation panels
            if vis_count < args.num_samples:
                save_infer_result(
                    imgs[i], preds[i],
                    os.path.join(vis_dir, f'{base}_vis.png'),
                    title=name)
                vis_count += 1

        pbar.set_postfix(ms=f'{ms:.1f}')

    avg_ms = float(np.mean(all_ms))
    print(f"\nInference complete.")
    print(f"  Avg inference speed : {avg_ms:.2f} ms/image  "
          f"({'PASS ✓' if avg_ms < 50 else 'FAIL ✗  target < 50 ms'})")
    print(f"  Outputs saved to    : {args.output_dir}/")

    # Write timing report
    timing_path = os.path.join(args.output_dir, 'inference_timing.txt')
    with open(timing_path, 'w', encoding='utf-8') as f:
        f.write("DUALITY AI — INFERENCE TIMING REPORT\n")
        f.write("=" * 40 + "\n")
        f.write(f"Avg inference speed : {avg_ms:.2f} ms/image\n")
        f.write(f"Spec target         : < 50.00 ms/image\n")
        f.write(f"Result              : {'PASS' if avg_ms < 50 else 'FAIL'}\n")
    print(f"  Timing report       : {timing_path}")


# ============================================================================
#  Entry Point
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description='Duality AI Segmentation — Evaluation & Inference')

    parser.add_argument(
        '--mode', choices=['eval', 'infer'], default='eval',
        help='"eval" computes metrics vs GT masks; '
             '"infer" runs on testImages/ (no GT needed)')
    parser.add_argument(
        '--model_path', type=str,
        default=os.path.join(script_dir, 'segmentation_head.pth'),
        help='Path to trained segmentation head weights (.pth)')
    parser.add_argument(
        '--data_dir', type=str,
        default=os.path.join(script_dir, '..', 'Offroad_Segmentation_Val'),
        help='Root folder of the dataset split to process')
    parser.add_argument(
        '--output_dir', type=str,
        default=os.path.join(script_dir, 'predictions'),
        help='Directory where all outputs are written')
    parser.add_argument(
        '--batch_size', type=int, default=2,
        help='Images per batch (reduce to 1 if GPU OOM)')
    parser.add_argument(
        '--num_samples', type=int, default=5,
        help='Number of visualisation / comparison images to save')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Startup banner ────────────────────────────────────────────────────────
    print("=" * 65)
    print("  Duality AI — Offroad Semantic Segmentation")
    print(f"  Mode        : {args.mode.upper()}")
    print(f"  Model       : {args.model_path}")
    print(f"  Data        : {args.data_dir}")
    print(f"  Output      : {args.output_dir}")
    print(f"  N classes   : {N_CLASSES}  (aligned to train.py)")
    print(f"  Classes     : {', '.join(CLASS_NAMES)}")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device      : {device}")

    # Training resolution — MUST match train.py exactly
    W_TRAIN = int(((960 / 2) // 14) * 14)   # 476
    H_TRAIN = int(((540 / 2) // 14) * 14)   # 266
    print(f"  Train res   : {W_TRAIN}×{H_TRAIN}  (patch-aligned)")

    # Image transform (same as train.py)
    img_transform = transforms.Compose([
        transforms.Resize((H_TRAIN, W_TRAIN)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    # Mask transform — NEAREST is mandatory [FIX-4]
    mask_transform = transforms.Compose([
        transforms.Resize((H_TRAIN, W_TRAIN),
                          interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    # Load model
    backbone, head = load_model(
        args.model_path, device, train_hw=(H_TRAIN, W_TRAIN))

    if args.mode == 'eval':
        run_eval(args, backbone, head, device, img_transform, mask_transform)
    else:
        run_infer(args, backbone, head, device, img_transform)


if __name__ == '__main__':
    main()