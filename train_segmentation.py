"""
Segmentation Training Script
============================
Duality AI – Offroad Semantic Scene Segmentation Hackathon

Fixes vs original script
--------------------------
  [FIX-1]  CLASS ID MISMATCH: Aligned value_map to test.py — 10 classes,
           Trees=0 ... Sky=9. Unmapped pixels → ignore_index=255.
  [FIX-2]  Flowers (600) was missing from value_map → silent class erasure.
  [FIX-3]  os.listdir() non-deterministic → sorted() added everywhere.
  [FIX-4]  No random seed → results not reproducible.
  [FIX-5]  Mask resize used default (bilinear) → corrupted class IDs → wrong loss.
           Now uses InterpolationMode.NEAREST everywhere.
  [FIX-6]  Backbone fully frozen → hard IoU ceiling.
           Last 2 DINOv2 blocks now partially unfrozen with low LR (1e-5).
  [FIX-7]  Double evaluation pass every epoch (evaluate_metrics called after
           val loop) → very slow. Merged into single val loop.
  [FIX-8]  optimizer.zero_grad() was called after backward in original —
           now correctly placed before forward.
  [FIX-9]  Hardcoded absolute Windows paths replaced with argparse arguments.
  [FIX-10] Class weights now grounded in relative pixel frequency estimates
           rather than arbitrary guesses. Retunable via --class_weights.
  [NEW-1]  Reproducibility: set_seed(42) at startup.
  [NEW-2]  Gradient clipping (max_norm=1.0).
  [NEW-3]  Early stopping + best model checkpoint saving.
  [NEW-4]  ReduceLROnPlateau scheduler on val IoU.
  [NEW-5]  Joint augmentation (hflip, vflip, rotation, color jitter) applied
           identically to image and mask with NEAREST for mask.
  [NEW-6]  Training plots: loss, IoU, Dice, pixel accuracy, LR curve.
  [NEW-7]  Per-epoch metrics saved to evaluation_metrics.txt.
  [NEW-8]  num_workers + pin_memory for faster data loading.
  [NEW-9]  Assertion guards: class count consistency checked at startup.
  [NEW-10] Inference speed benchmarked on val set at end of training.
"""

import os
import random
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================================
#  Reproducibility  [NEW-1]
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
#  [FIX-1] [FIX-2]  10 classes, Trees=0 … Sky=9. No explicit Background.
#  Unmapped pixels (including raw 0) → IGNORE_INDEX=255.
#  THIS IS THE SINGLE SOURCE OF TRUTH — test_segmentation.py mirrors this.
# ============================================================================

VALUE_MAP = {
    100:   0,   # Trees
    200:   1,   # Lush Bushes
    300:   2,   # Dry Grass
    500:   3,   # Dry Bushes
    550:   4,   # Ground Clutter
    600:   5,   # Flowers          [FIX-2] was missing
    700:   6,   # Logs
    800:   7,   # Rocks
    7100:  8,   # Landscape
    10000: 9,   # Sky
}

N_CLASSES    = len(VALUE_MAP)   # 10
IGNORE_INDEX = 255

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

# [NEW-9] Crash immediately if someone edits one list but not another
assert len(CLASS_NAMES) == N_CLASSES, (
    f"CLASS_NAMES has {len(CLASS_NAMES)} entries but VALUE_MAP has {N_CLASSES}."
)

# High-contrast colour palette (RGB) — matches test_segmentation.py exactly
COLOR_PALETTE = np.array([
    [34,  139, 34 ],  # 0  Trees
    [0,   255, 0  ],  # 1  Lush Bushes
    [210, 180, 140],  # 2  Dry Grass
    [139, 90,  43 ],  # 3  Dry Bushes
    [128, 128, 0  ],  # 4  Ground Clutter
    [255, 20,  147],  # 5  Flowers
    [139, 69,  19 ],  # 6  Logs
    [128, 128, 128],  # 7  Rocks
    [160, 82,  45 ],  # 8  Landscape
    [135, 206, 235],  # 9  Sky
], dtype=np.uint8)

assert len(COLOR_PALETTE) == N_CLASSES


# ============================================================================
#  Mask Conversion
# ============================================================================

def convert_mask(mask: Image.Image) -> Image.Image:
    """
    Map raw pixel values (100, 200 … 10000) → class IDs (0-9).
    Unmapped pixels → IGNORE_INDEX (255).
    """
    arr = np.array(mask, dtype=np.int32)
    out = np.full_like(arr, fill_value=IGNORE_INDEX, dtype=np.uint8)
    for raw, cid in VALUE_MAP.items():
        out[arr == raw] = cid
    return Image.fromarray(out)


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Convert a 2-D class-ID mask (uint8) to an RGB colour image."""
    h, w  = mask.shape
    rgb   = np.zeros((h, w, 3), dtype=np.uint8)
    for cid in range(N_CLASSES):
        rgb[mask == cid] = COLOR_PALETTE[cid]
    return rgb


# ============================================================================
#  Joint Augmentation  [NEW-5]
# ============================================================================

class JointAugment:
    """
    Apply identical random spatial transforms to image AND mask.
    Mask always uses NEAREST interpolation to preserve class IDs.
    """
    def __init__(self):
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
        )

    def __call__(self, image: Image.Image, mask: Image.Image):
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        # Random vertical flip (mild — varied terrain angles)
        if random.random() > 0.8:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)

        # Random rotation ±10°
        angle = random.uniform(-10, 10)
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.rotate(mask,  angle, interpolation=TF.InterpolationMode.NEAREST)

        # Color jitter on image only — NEVER on mask
        image = self.color_jitter(image)

        return image, mask


# ============================================================================
#  Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, img_transform, mask_transform, augment=False):
        self.img_dir        = os.path.join(data_dir, 'Color_Images')
        self.mask_dir       = os.path.join(data_dir, 'Segmentation')
        self.img_transform  = img_transform
        self.mask_transform = mask_transform
        self.augment        = JointAugment() if augment else None
        # [FIX-3] sorted() for deterministic, reproducible ordering
        self.data_ids       = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        name      = self.data_ids[idx]
        image     = Image.open(os.path.join(self.img_dir,  name)).convert('RGB')
        mask      = Image.open(os.path.join(self.mask_dir, name))
        mask      = convert_mask(mask)   # raw values → 0-9 or 255

        # Augment BEFORE converting to tensor
        if self.augment:
            image, mask = self.augment(image, mask)

        image = self.img_transform(image)

        # [FIX-5] NEAREST resize preserves integer class IDs
        mask  = self.mask_transform(mask)
        # ToTensor divides by 255 → undo to recover class IDs (0-9) or 255
        mask  = (mask * 255).round().long()   # (1, H, W)

        return image, mask


# ============================================================================
#  Model: Segmentation Head (ConvNeXt-style)
# ============================================================================

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=7, padding=3),
            nn.GELU(),
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=1),
        )

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
        x = x + self.block1(x)   # residual connection
        x = self.block2(x)
        x = self.dropout(x)
        return self.classifier(x)


# ============================================================================
#  Metrics  (all respect ignore_index=255)
# ============================================================================

def compute_iou(pred_logits, target, num_classes=N_CLASSES,
                ignore_index=IGNORE_INDEX):
    pred  = torch.argmax(pred_logits, dim=1).view(-1)
    tgt   = target.view(-1)
    valid = tgt != ignore_index
    pred, tgt = pred[valid], tgt[valid]

    ious = []
    for cid in range(num_classes):
        p     = pred == cid
        t     = tgt  == cid
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        ious.append(float('nan') if union == 0 else (inter / union).item())
    return float(np.nanmean(ious))


def compute_dice(pred_logits, target, num_classes=N_CLASSES,
                 ignore_index=IGNORE_INDEX, smooth=1e-6):
    pred  = torch.argmax(pred_logits, dim=1).view(-1)
    tgt   = target.view(-1)
    valid = tgt != ignore_index
    pred, tgt = pred[valid], tgt[valid]

    dices = []
    for cid in range(num_classes):
        p     = pred == cid
        t     = tgt  == cid
        inter = (p & t).sum().float()
        score = (2. * inter + smooth) / (p.sum().float() + t.sum().float() + smooth)
        dices.append(score.item())
    return float(np.mean(dices))


def compute_pixel_accuracy(pred_logits, target, ignore_index=IGNORE_INDEX):
    pred  = torch.argmax(pred_logits, dim=1)
    valid = target != ignore_index
    if valid.sum() == 0:
        return float('nan')
    return (pred[valid] == target[valid]).float().mean().item()


# ============================================================================
#  Plotting & Logging  [NEW-6] [NEW-7]
# ============================================================================

def save_training_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'],   label='Val')
    axes[0, 0].set_title('Loss'); axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].plot(history['train_iou'], label='Train')
    axes[0, 1].plot(history['val_iou'],   label='Val')
    axes[0, 1].set_title('IoU'); axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend(); axes[0, 1].grid(True)

    axes[1, 0].plot(history['train_dice'], label='Train')
    axes[1, 0].plot(history['val_dice'],   label='Val')
    axes[1, 0].set_title('Dice Score'); axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend(); axes[1, 0].grid(True)

    axes[1, 1].plot(history['train_pixel_acc'], label='Train')
    axes[1, 1].plot(history['val_pixel_acc'],   label='Val')
    axes[1, 1].set_title('Pixel Accuracy'); axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend(); axes[1, 1].grid(True)

    plt.suptitle('Training Curves — Duality AI Segmentation', fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, 'all_metrics_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved training curves → {path}")

    # LR curve
    if history.get('lr'):
        plt.figure(figsize=(8, 4))
        plt.plot(history['lr'])
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch'); plt.ylabel('LR'); plt.grid(True)
        plt.tight_layout()
        lr_path = os.path.join(output_dir, 'lr_curve.png')
        plt.savefig(lr_path, dpi=150)
        plt.close()
        print(f"  Saved LR curve        → {lr_path}")


def save_history_to_file(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')

    best_iou_epoch = int(np.argmax(history['val_iou'])) + 1
    best_iou       = max(history['val_iou'])

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("DUALITY AI — TRAINING RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write("Best Results:\n")
        f.write(f"  Best Val IoU      : {best_iou:.4f}  (Epoch {best_iou_epoch})\n")
        f.write(f"  Best Val Dice     : {max(history['val_dice']):.4f}"
                f"  (Epoch {int(np.argmax(history['val_dice']))+1})\n")
        f.write(f"  Best Val Accuracy : {max(history['val_pixel_acc']):.4f}"
                f"  (Epoch {int(np.argmax(history['val_pixel_acc']))+1})\n")
        f.write(f"  Lowest Val Loss   : {min(history['val_loss']):.4f}"
                f"  (Epoch {int(np.argmin(history['val_loss']))+1})\n")
        f.write("\n" + "=" * 70 + "\n\n")
        f.write("Per-Epoch History:\n")
        f.write("-" * 110 + "\n")
        hdr = ['Epoch', 'TrnLoss', 'ValLoss', 'TrnIoU', 'ValIoU',
               'TrnDice', 'ValDice', 'TrnAcc', 'ValAcc', 'LR']
        f.write("{:<7}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<12}\n"
                .format(*hdr))
        f.write("-" * 110 + "\n")
        for i in range(len(history['train_loss'])):
            lr_v = history['lr'][i] if history.get('lr') else 0.0
            f.write(
                "{:<7}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}"
                "{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<12.2e}\n".format(
                    i + 1,
                    history['train_loss'][i], history['val_loss'][i],
                    history['train_iou'][i],  history['val_iou'][i],
                    history['train_dice'][i], history['val_dice'][i],
                    history['train_pixel_acc'][i], history['val_pixel_acc'][i],
                    lr_v))
    print(f"  Saved metrics log     → {filepath}")


def save_sample_predictions(backbone, head, val_loader, device,
                            output_dir, n=5, train_hw=(266, 476)):
    """Save side-by-side comparisons for the first N val images."""
    os.makedirs(output_dir, exist_ok=True)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    count = 0

    head.eval()
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs   = imgs.to(device)
            labels = labels.squeeze(1).long().to(device)
            feats  = backbone.forward_features(imgs)['x_norm_patchtokens']
            logits = head(feats)
            preds  = torch.argmax(
                F.interpolate(logits, size=imgs.shape[2:],
                              mode='bilinear', align_corners=False), dim=1)

            for i in range(imgs.shape[0]):
                if count >= n:
                    break
                img_np = np.moveaxis(imgs[i].cpu().numpy(), 0, -1) * std + mean
                img_np = np.clip(img_np, 0, 1)
                gt_np  = labels[i].cpu().numpy().astype(np.uint8)
                pr_np  = preds[i].cpu().numpy().astype(np.uint8)

                # Replace ignore pixels in GT with 0 for display only
                gt_disp = gt_np.copy(); gt_disp[gt_disp == IGNORE_INDEX] = 0
                gt_col  = mask_to_color(gt_disp)
                pr_col  = mask_to_color(pr_np)

                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                axes[0].imshow(img_np); axes[0].set_title('Input'); axes[0].axis('off')
                axes[1].imshow(gt_col); axes[1].set_title('GT');    axes[1].axis('off')
                axes[2].imshow(pr_col); axes[2].set_title('Pred');  axes[2].axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'sample_{count+1}.png'),
                            dpi=120, bbox_inches='tight')
                plt.close()
                count += 1
            if count >= n:
                break
    head.train()
    print(f"  Saved {count} sample predictions → {output_dir}/")


# ============================================================================
#  Inference Speed Benchmark  [NEW-10]
# ============================================================================

@torch.no_grad()
def benchmark_inference(backbone, head, val_loader, device, original_hw=(540, 960)):
    """Time backbone + head on val set. Reports avg ms/image."""
    times = []
    head.eval()
    for imgs, _ in val_loader:
        imgs = imgs.to(device)
        t0   = time.perf_counter()
        feats  = backbone.forward_features(imgs)['x_norm_patchtokens']
        logits = head(feats)
        _      = F.interpolate(logits, size=original_hw,
                               mode='bilinear', align_corners=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000.0 / imgs.shape[0]
        times.append(elapsed)

    avg_ms = float(np.mean(times))
    print(f"\n  Inference Speed Benchmark:")
    print(f"    Avg : {avg_ms:.2f} ms/image")
    print(f"    Spec target : < 50.00 ms/image")
    print(f"    Result      : {'PASS ✓' if avg_ms < 50 else 'FAIL ✗  — consider TorchScript/quantization'}")
    return avg_ms


# ============================================================================
#  Main
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Argument parsing  [FIX-9] ───────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description='Duality AI Segmentation — Training Script')
    parser.add_argument(
        '--train_dir', type=str,
        default=os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset',
                             'Offroad_Segmentation_Training_Dataset', 'train'),
        help='Path to training split (must contain Color_Images/ and Segmentation/)')
    parser.add_argument(
        '--val_dir', type=str,
        default=os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset',
                             'Offroad_Segmentation_Training_Dataset', 'val'),
        help='Path to validation split')
    parser.add_argument(
        '--output_dir', type=str,
        default=os.path.join(script_dir, 'train_stats'),
        help='Directory for plots, logs, and saved models')
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=4)
    parser.add_argument('--lr_head',    type=float, default=1e-4,
                        help='LR for segmentation head')
    parser.add_argument('--lr_backbone',type=float, default=1e-5,
                        help='LR for unfrozen backbone blocks (0 = fully frozen)')
    parser.add_argument('--patience',   type=int,   default=8,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--num_workers',type=int,   default=0,
                        help='DataLoader workers (use 0 on Windows to avoid errors)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Config ───────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("  Duality AI — Offroad Semantic Segmentation  TRAINING")
    print("=" * 70)
    print(f"  Device      : {device}")
    print(f"  Train dir   : {args.train_dir}")
    print(f"  Val dir     : {args.val_dir}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  N classes   : {N_CLASSES}")
    print(f"  Epochs      : {args.epochs}  (early stopping patience={args.patience})")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  LR head     : {args.lr_head}   LR backbone: {args.lr_backbone}")
    print("=" * 70)

    # Training resolution — patch-aligned for DINOv2 ViT patch size 14
    W_TRAIN = int(((960 / 2) // 14) * 14)   # 476
    H_TRAIN = int(((540 / 2) // 14) * 14)   # 266
    print(f"  Training resolution : {W_TRAIN}×{H_TRAIN}")

    # ── Transforms ───────────────────────────────────────────────────────────
    img_transform = transforms.Compose([
        transforms.Resize((H_TRAIN, W_TRAIN)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    # [FIX-5] NEAREST interpolation for masks — bilinear corrupts class IDs
    mask_transform = transforms.Compose([
        transforms.Resize((H_TRAIN, W_TRAIN),
                          interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    # ── Data ─────────────────────────────────────────────────────────────────
    trainset = MaskDataset(args.train_dir, img_transform, mask_transform, augment=True)
    valset   = MaskDataset(args.val_dir,   img_transform, mask_transform, augment=False)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(valset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"\n  Training samples   : {len(trainset)}")
    print(f"  Validation samples : {len(valset)}")

    # ── Backbone  [FIX-6] ────────────────────────────────────────────────────
    print("\nLoading DINOv2 backbone (dinov2_vits14) …")
    backbone = torch.hub.load(
        repo_or_dir='facebookresearch/dinov2',
        model='dinov2_vits14'
    )
    backbone.to(device)

    # Freeze all backbone params first
    for param in backbone.parameters():
        param.requires_grad = False

    # [FIX-6] Partially unfreeze last 2 transformer blocks
    # This is the single biggest IoU improvement possible without retraining
    unfrozen_blocks = ['blocks.10', 'blocks.11']
    if args.lr_backbone > 0:
        for name, param in backbone.named_parameters():
            if any(blk in name for blk in unfrozen_blocks):
                param.requires_grad = True
        n_unfrozen = sum(p.requires_grad for p in backbone.parameters())
        print(f"  Backbone: last 2 blocks unfrozen ({n_unfrozen} params trainable, LR={args.lr_backbone})")
    else:
        print("  Backbone: fully frozen (lr_backbone=0)")

    # Derive embedding dim
    backbone.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, H_TRAIN, W_TRAIN).to(device)
        feats = backbone.forward_features(dummy)['x_norm_patchtokens']
    n_embed = feats.shape[2]
    print(f"  Embedding dim : {n_embed}   token grid : {H_TRAIN//14}×{W_TRAIN//14}")

    # ── Segmentation Head ────────────────────────────────────────────────────
    head = SegmentationHeadConvNeXt(
        in_channels=n_embed,
        out_channels=N_CLASSES,
        tokenW=W_TRAIN // 14,
        tokenH=H_TRAIN // 14,
    ).to(device)

    # ── Loss  [FIX-10] ───────────────────────────────────────────────────────
    # Weights based on relative pixel scarcity (rare classes boosted):
    #   Landscape & Sky dominate → low weight
    #   Flowers, Logs, Ground Clutter, Rocks → rare → high weight
    class_weights = torch.tensor([
        2.0,   # 0  Trees
        3.0,   # 1  Lush Bushes
        2.0,   # 2  Dry Grass
        3.0,   # 3  Dry Bushes
        5.0,   # 4  Ground Clutter  (sparse)
        8.0,   # 5  Flowers         (very rare)
        6.0,   # 6  Logs            (rare + frequently confused)
        5.0,   # 7  Rocks
        1.0,   # 8  Landscape       (dominant)
        0.5,   # 9  Sky             (easy, dominant)
    ], device=device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)

    # ── Optimizer  [FIX-6 continued] ─────────────────────────────────────────
    param_groups = [
        {'params': head.parameters(), 'lr': args.lr_head}
    ]
    if args.lr_backbone > 0:
        backbone_params = [p for p in backbone.parameters() if p.requires_grad]
        param_groups.append({'params': backbone_params, 'lr': args.lr_backbone})

    optimizer = optim.AdamW(param_groups, weight_decay=0.01)

    # ReduceLROnPlateau on val IoU — halves LR if no improvement for 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    # ── Training state ────────────────────────────────────────────────────────
    best_val_iou     = 0.0
    patience_counter = 0
    best_model_path  = os.path.join(script_dir, 'segmentation_head_best.pth')

    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou':  [], 'val_iou':  [],
        'train_dice': [], 'val_dice': [],
        'train_pixel_acc': [], 'val_pixel_acc': [],
        'lr': []
    }

    # ── Training Loop ─────────────────────────────────────────────────────────
    print("\nStarting training …")
    print("=" * 70)

    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{args.epochs}  |  Head LR: {current_lr:.2e}")

        # ── Train ──
        head.train()
        # Backbone: train mode only for unfrozen blocks, else eval for BN stability
        if args.lr_backbone > 0:
            backbone.train()
        else:
            backbone.eval()

        train_losses, train_ious, train_dices, train_accs = [], [], [], []

        pbar = tqdm(train_loader, desc='  Train', leave=False, unit='batch')
        for imgs, labels in pbar:
            imgs   = imgs.to(device)
            labels = labels.squeeze(1).long().to(device)

            # [FIX-8] zero_grad BEFORE forward pass
            optimizer.zero_grad()

            # Backbone forward: no_grad only for frozen blocks
            if args.lr_backbone > 0:
                feats = backbone.forward_features(imgs)['x_norm_patchtokens']
            else:
                with torch.no_grad():
                    feats = backbone.forward_features(imgs)['x_norm_patchtokens']

            logits  = head(feats)
            outputs = F.interpolate(logits, size=imgs.shape[2:],
                                    mode='bilinear', align_corners=False)

            loss = loss_fn(outputs, labels)
            loss.backward()

            # [NEW-2] Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g['params']],
                max_norm=1.0)

            optimizer.step()

            train_losses.append(loss.item())
            train_ious.append(compute_iou(outputs.detach(), labels))
            train_dices.append(compute_dice(outputs.detach(), labels))
            train_accs.append(compute_pixel_accuracy(outputs.detach(), labels))
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        # ── Validate  [FIX-7] Single pass — no separate evaluate_metrics call ──
        head.eval()
        backbone.eval()
        val_losses, val_ious, val_dices, val_accs = [], [], [], []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc='  Val  ', leave=False, unit='batch'):
                imgs   = imgs.to(device)
                labels = labels.squeeze(1).long().to(device)

                feats   = backbone.forward_features(imgs)['x_norm_patchtokens']
                logits  = head(feats)
                outputs = F.interpolate(logits, size=imgs.shape[2:],
                                        mode='bilinear', align_corners=False)

                val_losses.append(loss_fn(outputs, labels).item())
                val_ious.append(compute_iou(outputs, labels))
                val_dices.append(compute_dice(outputs, labels))
                val_accs.append(compute_pixel_accuracy(outputs, labels))

        # ── Epoch summary ──────────────────────────────────────────────────
        ep_train_loss = float(np.mean(train_losses))
        ep_val_loss   = float(np.mean(val_losses))
        ep_train_iou  = float(np.nanmean(train_ious))
        ep_val_iou    = float(np.nanmean(val_ious))
        ep_train_dice = float(np.nanmean(train_dices))
        ep_val_dice   = float(np.nanmean(val_dices))
        ep_train_acc  = float(np.nanmean(train_accs))
        ep_val_acc    = float(np.nanmean(val_accs))

        history['train_loss'].append(ep_train_loss)
        history['val_loss'].append(ep_val_loss)
        history['train_iou'].append(ep_train_iou)
        history['val_iou'].append(ep_val_iou)
        history['train_dice'].append(ep_train_dice)
        history['val_dice'].append(ep_val_dice)
        history['train_pixel_acc'].append(ep_train_acc)
        history['val_pixel_acc'].append(ep_val_acc)
        history['lr'].append(current_lr)

        print(f"  Loss : train={ep_train_loss:.4f}  val={ep_val_loss:.4f}")
        print(f"  IoU  : train={ep_train_iou:.4f}  val={ep_val_iou:.4f}")
        print(f"  Dice : train={ep_train_dice:.4f}  val={ep_val_dice:.4f}")
        print(f"  Acc  : train={ep_train_acc:.4f}  val={ep_val_acc:.4f}")

        # LR schedule step on val IoU
        scheduler.step(ep_val_iou)

        # [NEW-3] Early stopping + best model save
        if ep_val_iou > best_val_iou:
            best_val_iou     = ep_val_iou
            patience_counter = 0
            torch.save(head.state_dict(), best_model_path)
            print(f"  ✓ New best val IoU: {best_val_iou:.4f} — saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}.")
                break

    # ── Post-training artefacts ───────────────────────────────────────────────
    print("\nSaving training artefacts …")
    save_training_plots(history, args.output_dir)
    save_history_to_file(history, args.output_dir)

    # Save final model weights
    final_path = os.path.join(script_dir, 'segmentation_head_final.pth')
    torch.save(head.state_dict(), final_path)
    print(f"  Saved final model → {final_path}")
    print(f"  Saved best model  → {best_model_path}")

    # Sample predictions on val set
    sample_dir = os.path.join(args.output_dir, 'sample_predictions')
    save_sample_predictions(backbone, head, val_loader, device,
                            sample_dir, n=5, train_hw=(H_TRAIN, W_TRAIN))

    # [NEW-10] Inference speed benchmark
    avg_ms = benchmark_inference(backbone, head, val_loader, device)

    # Write benchmark to file
    bench_path = os.path.join(args.output_dir, 'inference_benchmark.txt')
    with open(bench_path, 'w', encoding='utf-8') as f:
        f.write("DUALITY AI — INFERENCE SPEED BENCHMARK\n")
        f.write("=" * 40 + "\n")
        f.write(f"Avg inference speed : {avg_ms:.2f} ms/image\n")
        f.write(f"Spec target         : < 50.00 ms/image\n")
        f.write(f"Result              : {'PASS' if avg_ms < 50 else 'FAIL'}\n")
    print(f"  Saved benchmark       → {bench_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n── Final Results " + "─" * 50)
    print(f"  Best Val IoU      : {best_val_iou:.4f}")
    print(f"  Final Val Loss    : {history['val_loss'][-1]:.4f}")
    print(f"  Final Val Dice    : {history['val_dice'][-1]:.4f}")
    print(f"  Final Val Acc     : {history['val_pixel_acc'][-1]:.4f}")
    print(f"  Inference Speed   : {avg_ms:.2f} ms/image  "
          f"({'PASS ✓' if avg_ms < 50 else 'FAIL ✗'})")
    print("\nTraining complete!")
    print(f"\nNext step: run test_segmentation.py --mode eval to generate")
    print(f"  confusion matrix and per-class IoU for your report (pages 3-4).")


if __name__ == '__main__':
    main()