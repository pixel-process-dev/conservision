# ConserVision: Wildlife Camera Trap Classification

Multi-model ensemble approach for the [DrivenData ConserVision](https://www.drivendata.org/competitions/87/competition-conservation-imagery/) wildlife image classification competition. Classifies camera trap images into 8 species categories using an ensemble of fine-tuned vision models with diversity-driven design.

**Result:** Top 2% perfromance out of 500+ competitors (Top Rank #11, log loss: 0.8101)

---

## Approach

The core insight driving this project: **ensemble diversity matters more than individual model strength.** Rather than optimizing a single architecture, the pipeline generates maximally decorrelated predictions through three axes of variation, then combines them with simple averaging.

All training done on consumer hardware (NVIDIA GeForce RTX 2060).


### Stage 1: Detection & Cropping

[MegaDetectorV6](https://github.com/microsoft/CameraTrap/blob/main/megadetector.md) extracts animal crops from raw camera trap images at multiple confidence thresholds. Lower thresholds (0.05) include more borderline detections, producing ~20K crops; higher thresholds (0.1) are more selective at ~18.5K. The different training distributions created by each threshold proved to be the single most impactful source of ensemble diversity. Full uncropped images are also used as a third input path, providing scene-level context that complements the crop-focused models.

### Stage 2: Classification Models

Eight backbone architectures are trained across detection thresholds using stratified group k-fold cross-validation (5 folds):

| Model | Backbone | Input | Folds | Mean F1 |
|-------|----------|-------|:-----:|:-------:|
| swinv2_.1 | SwinV2-Base | Crop (0.1) | 5 | 0.685 |
| convnext_.1 | ConvNeXt-Base | Crop (0.1) | 5 | 0.681 |
| dinov2_.1 | DINOv2 ViT-B/14 | Crop (0.1) | 5 | 0.680 |
| eva02_1 | EVA02 CLIP ViT-B/16 | Crop (0.1) | 5 | 0.677 |
| evaV2_.05 | EVA02 CLIP ViT-B/16 | Crop (0.05) | 5 | 0.671 |
| swinv2_.05 | SwinV2-Base | Crop (0.05) | 5 | 0.668 |
| dino_05 | DINOv2 ViT-B/14 | Crop (0.05) | 5 | 0.667 |
| convnext_.05 | ConvNeXt-Base | Crop (0.05) | 5 | 0.662 |


All models share the same training recipe: differential learning rates (head vs backbone), cosine annealing, and inverse-frequency class weighting. 

Models vary in in terms of augmentation (including mixup) and crop thresholds.


### Stage 3: Ensemble

Multiple ensemble approaches were tested including equal-weight probability averaging across all models, class and model weights, and a meta learner approach. **The meta-learning appraoch provided the best overall results.**

## Classes

| Class | Train Count | Description |
|-------|:-----------:|-------------|
| antelope_duiker | 2,474 | Antelopes and duikers |
| bird | 1,641 | Various bird species |
| blank | 2,213 | Empty frames / no animal |
| civet_genet | 2,423 | Civets and genets |
| hog | 978 | Wild hogs |
| leopard | 2,254 | Leopards |
| monkey_prosimian | 2,492 | Monkeys and prosimians |
| rodent | 2,013 | Rodent species |

## Repository Structure

```
conservision/
├── configs/               # Model training configurations (JSON)
├── data/
│   └── competition/       # Competition data (not tracked)
│       ├── train_features/
│       ├── test_features/
│       └── train_labels.csv
├── models/                # Trained checkpoints + fold predictions
│   ├── swinv2_.1_folds/
│   ├── convnext_.05_folds/
│   └── ...
├── scripts/
│   ├── 01_detection_cropping.py    # Create crop data
│   ├── 02_train.py                 # Train a single model from config
│   ├── 02_train_ovr.py             # Train a one-vs-rest single model from config
│   ├── 03_kfold_runner.py          # K-fold wrapper for training pipeline
│   ├── 03_kfold_runner_ovr.py      # K-fold wrapper for OVR training pipeline
│   ├── 04_predict.py               # Generate predictions including variations like tta
│   ├── 05_aggregate_folds.py       # Generate average model level predictions from kfolds
│   └── 06_ensemble.py              # Evaluates multiple ensemble approaches
├── notebooks/             # Exploratory analysis (includes final submission creation)
├── src/                   # Base code used in scripts (data loading and aug, metrics)
└── README.md
```


## Reproducing Results

### Requirements
- Python 3.10+, CUDA-capable GPU
- PyTorch, timm, LightGBM, scikit-learn
- MegaDetectorV6 for crop extraction

Full details in environment.yml

### Workflow

```bash
# Generate crops at multiple thresholds
python scripts/01_detection_cropping.py --data_dir data/competition/ --conf_threshold .1

# Train single model
python scripts/02_train.py --config configs/swinv2_.1.json --data_dir data/competition/

# Train batch k-fold models
python scripts/03_kfold_runner.py --data_dir data/competition/ --config configs/train_jobs.json

### OPTIONAL One-vs-Rest
# Train single OVR model
python scripts/02_train_ovr.py --config configs/ovr_blank_dinov2.json --data_dir data/competition/

# Train batch k-fold models
python scripts/03_kfold_runner_ovr.py --data_dir data/competition/ --config configs/train_jobs_ovr.json

# Generate predictions
python scripts/04_predict.py --data_dir data/competition/ --models_dir models/ --crop_map configs/crop_map.json

# Generate model level results from folds
python scripts/05_aggregate_folds.py --models models/

# Test and evaluate ensemble approaches
python scripts/0ensemble.py --config configs/ensemble.json

```


## Tools & Frameworks

- **[timm](https://github.com/huggingface/pytorch-image-models)** — Pretrained backbones and training utilities
- **[MegaDetectorV6](https://github.com/microsoft/CameraTrap)** — Animal detection and cropping
- **[MLflow](https://mlflow.org/)** — Experiment tracking across 40+ trained models
- **[LightGBM](https://lightgbm.readthedocs.io/)** — Meta-learner experiments
