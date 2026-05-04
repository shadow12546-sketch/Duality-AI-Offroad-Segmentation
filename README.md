# Duality AI — Offroad Semantic Scene Segmentation

## Team
- Name: Logic Legends
- Hackathon: Hack on Titan

## Project Overview
Semantic segmentation of offroad desert environments using DINOv2 ViT-S/14 backbone with a custom ConvNeXt-style segmentation head. Trained on synthetic data generated from Duality AI's Falcon digital twin platform.

## Model Architecture
- Backbone : DINOv2 ViT-S/14 (last 2 transformer blocks unfrozen)
- Head     : Custom ConvNeXt-style segmentation head with residual connections
- Classes  : 10 (Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Flowers, Logs, Rocks, Landscape, Sky)
- Input res: 476×266 (patch-aligned for ViT patch size 14)

## Key Improvements Over Baseline
1. Fixed class ID mismatch (10 classes, no background)
2. Fixed mask resize to use NEAREST interpolation
3. Added missing Flowers class (ID 600)
4. Partial backbone unfreezing (last 2 blocks, LR=1e-5)
5. Joint augmentation (hflip, vflip, rotation, color jitter)
6. Frequency-based class weights for rare classes
7. Early stopping + best model checkpointing
8. Gradient clipping (max_norm=1.0)
9. Reproducibility seed (42)
10. Inference speed benchmark vs 50ms spec

## Environment Setup
conda activate EDU

## Training
python train_segmentation.py --train_dir <path_to_train> --val_dir <path_to_val> --epochs 50 --batch_size 4 --lr_head 1e-4 --lr_backbone 1e-5 --output_dir train_stats

## Evaluation (with ground truth)
python test_segmentation.py --mode eval --model_path segmentation_head_best.pth --data_dir <path_to_val> --output_dir predictions

## Inference (test images, no ground truth)
python test_segmentation.py --mode infer --model_path segmentation_head_best.pth --data_dir <path_to_testImages> --output_dir predictions

## Model Weights
Download segmentation_head_best.pth from: [Google Drive Link]
Place in: Offroad_Segmentation_Scripts/segmentation_head_best.pth

## Results
| Metric          | Score   |
|-----------------|---------|
| Val IoU (mIoU)  | 0.2951  |
| Val Dice        | 0.4864  |
| Pixel Accuracy  | 0.6894  |
| Inference Speed | 3.18 ms |

## Per-Class IoU
| Class          | IoU    |
|----------------|--------|
| Trees          | 0.1888 |
| Lush Bushes    | 0.2423 |
| Dry Grass      | 0.4975 |
| Dry Bushes     | 0.0086 |
| Ground Clutter | 0.1790 |
| Flowers        | 0.1504 |
| Logs           | 0.0333 |
| Rocks          | 0.1593 |
| Landscape      | 0.2829 |
| Sky            | 0.8647 |

## Classes
| ID    | Class          | Weight |
|-------|----------------|--------|
| 100   | Trees          | 2.0    |
| 200   | Lush Bushes    | 3.0    |
| 300   | Dry Grass      | 2.0    |
| 500   | Dry Bushes     | 3.0    |
| 550   | Ground Clutter | 5.0    |
| 600   | Flowers        | 8.0    |
| 700   | Logs           | 6.0    |
| 800   | Rocks          | 5.0    |
| 7100  | Landscape      | 1.0    |
| 10000 | Sky            | 0.5    |

## Outputs
After evaluation, find results in predictions/
- evaluation_metrics.txt — IoU, Dice, Pixel Accuracy, inference speed
- confusion_matrix.png   — Row-normalised confusion matrix
- per_class_iou.png      — Per-class IoU bar chart
- failure_cases/         — Top 5 worst predictions
- comparisons/           — Input vs GT vs Prediction panels
- masks_color/           — Coloured prediction masks

After training, find results in train_stats/
- all_metrics_curves.png — Loss, IoU, Dice, Accuracy curves
- lr_curve.png           — Learning rate schedule
- evaluation_metrics.txt — Per-epoch metrics log
