# Intel Image Classification — Transfer Learning Ablation Study

A systematic study of transfer learning techniques on the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) (6 classes: buildings, forest, glacier, mountain, sea, street), using a pretrained ResNet backbone.

Instead of just training a model and reporting one accuracy number, this project runs a controlled ablation study — starting from a frozen baseline and adding one technique at a time — to figure out *which* fine-tuning decisions actually move the needle, and which don't.

## Results

| # | Experiment | Change from previous | Test Accuracy |
|---|------------|----------------------|----------------|
| 1 | Baseline | Frozen backbone, train FC head only (lr=0.01) | 89.46% |
| 2 | Unfreeze layer4 | Unfroze `layer4` + FC (lr=0.01, unchanged) | 89.5% |
| 3 | Lower learning rate | lr dropped to 0.0001 | 92.9% |
| 4 | Dropout | Added `Dropout(p=0.5)` before FC | 94% |
| 5 | Weight decay | Added `weight_decay=1e-4` to Adam | 93% |
| 6 | LR scheduler | Added `StepLR(step_size=3, gamma=0.1)` | 93% |
| 7 | Combined | Dropout + weight decay + scheduler + tuned lr, all together | **94.1%** |

**Final model (Experiment 7) — 94.1% test accuracy.**

## Key findings

- **Unfreezing a layer without adjusting the learning rate does almost nothing.** Exp 2 unfroze `layer4` but kept exp 1's lr (0.01) — accuracy barely moved (89.46% → 89.5%). The pretrained weights in `layer4` were too sensitive to survive large gradient updates, so the extra trainable capacity was effectively wasted.
- **Learning rate was the single biggest lever.** Dropping lr from 0.01 → 0.0001 (exp 3) jumped accuracy from 89.5% → 92.9% — by far the largest single-experiment gain in the study. Pretrained layers need to be fine-tuned gently, not blasted with the same lr used for a randomly-initialized head.
- **Dropout was the strongest regularizer tested.** Adding `Dropout(0.5)` (94%) gave a bigger boost than either weight decay or the LR scheduler individually — consistent with the model (unfrozen layer4 + a relatively small dataset) being prone to overfitting, which dropout directly counters.
- **Weight decay and the LR scheduler each gave small, roughly equal gains (~93%)** on their own — real, but more marginal than the lr fix and dropout. The scheduler in particular may have had limited room to help given the short (10-epoch) training runs used here.
- **Combining everything (exp 7) edged out every individual technique**, landing at 94.1% — suggesting the gains from dropout, weight decay, and scheduling are mostly additive, even if the returns beyond dropout alone are small.

## Error analysis (Experiment 7 — final model)

<img src="confusion_matrix.png" width="500">

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| 0 | 0.95 | 0.92 | 0.93 | 437 |
| 1 | 0.98 | 1.00 | 0.99 | 474 |
| 2 | 0.92 | 0.90 | 0.91 | 553 |
| 3 | 0.91 | 0.89 | 0.90 | 525 |
| 4 | 0.95 | 0.98 | 0.97 | 510 |
| 5 | 0.94 | 0.96 | 0.95 | 501 |
| **Accuracy** | | | **0.94** | 3000 |

The model isn't uniformly good across classes — it's near-perfect on one class (1) but shows two consistent confusion pairs:

- **Classes 0 ↔ 5**: 32 samples of class 0 predicted as class 5, and 18 of class 5 predicted as class 0 — a two-way confusion, likely between visually similar scene categories (e.g. buildings/street, both containing urban structures).
- **Classes 2 ↔ 3**: 40 samples of class 2 predicted as class 3, and 39 of class 3 predicted as class 2 — the model's second-largest source of error, plausibly glacier/mountain, which share snow, rock, and similar color/texture profiles.

This kind of class-pair confusion is expected for scene classification — the errors line up with genuine visual overlap between categories rather than random noise, which is a good sign that the model has learned meaningful features rather than memorizing the training set.

## Setup

- **Backbone**: ResNet (ImageNet-pretrained)
- **Framework**: PyTorch
- **Optimizer**: Adam
- **Loss**: CrossEntropyLoss
- **Epochs**: 10 per experiment
- **Environment**: Google Colab (T4 GPU)

## What I'd try next

- Longer training runs to see if the LR scheduler earns its keep with more epochs to work with
- Targeted augmentation or a slightly deeper head for the 0↔5 and 2↔3 confusion pairs specifically
- Move to object detection (YOLO) as the next step in the learning roadmap
