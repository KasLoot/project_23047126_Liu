# Hand Gesture Multi-Task

This directory contains the source code for training, evaluating, and visualizing a multi-task framework designed for Hand Gesture Recognition. The models jointly tackle Object Detection, Image Classification, and Semantic Segmentation.

## Directory Structure

*   `dataloader.py`: Contains PyTorch Dataset classes (e.g., `HandGestureDataset_v2`), data augmentation pipelines, and collate functions for processing bounding boxes, classification labels, and segmentation masks.
*   `model.py`: Implements four distinct multi-task architectures (`RGB_V1` to `RGB_V4`). Modules range from base CNNs to advanced blocks (MBConv, SPPF, C2PSA, CrossAttention FPN).
*   `train.py`: The unified training routine handling a two-stage training strategy (`s1` and `s2`).
*   `evaluate.py`: Evaluates a trained model checkpoint on the validation/test set and generates performance metrics like F1-score, loss, and confusion matrices.
*   `visualise.py`: Generates qualitative visualizations of model predictions (bounding boxes, classification labels, segmentation masks) overlaid on images.
*   `utils.py`: Helper functions encompassing specialized unified loss functions, logging setups, metric computations, and bounding box conversions.

## Models

The framework supports four variants of model backbones, which determine complexity and performance tradeoffs. You must specify the architecture via the `--model` flag.

1.  `rgb_v1`: Standard Convolutional Network with a straightforward FPN neck.
2.  `rgb_v2`: Advanced backbone using MobileNetV2-style MBConv blocks (Depthwise Convolution + Squeeze-and-Excitation).
3.  `rgb_v3`: Employs Nano-scale YOLO-style CSP blocks (C3k2) and SPPF modules.
4.  `rgb_v4`: Extends V3 with Lightweight Cross-Attention, bounding maximum spatial context via a C2PSA block and custom CrossAttention Neck for better spatial feature fusion.

---

## Usage

### 0. Setup
1. Config python environment with pthon==3.11
2.  Install dependencies:
```bash
pip install -r requirements.txt
```

### 1. Training

Model training supports two stages: `s1` (Stage-1, e.g., initial training) and `s2` (Stage-2, fine-tuning). 

```bash
# Example: Train RGB_V3 for Stage 1
python src/train.py \
    --stage s1 \
    --model rgb_v3 \
    --train_dir dataset/dataset_v1/train \
    --val_dir dataset/dataset_v1/val \
    --epochs 10 \
    --batch_size 16 \
    --run_name "my_experiment"

# Example: Train RGB_V3 for Stage 2 (Fine-tuning)
python src/train.py \
    --stage s2 \
    --model rgb_v3 \
    --epochs 50 \
    --batch_size 32 \
    --run_name "my_experiment" \
    --stage1_weights weights/rgb_v3/s1/my_experiment/best_model.pth
```
**Key Arguments:**
*   `--stage`: Must be `s1` or `s2`.
*   `--model`: Architecture choice (`rgb_v1`, `rgb_v2`, `rgb_v3`, `rgb_v4`).
*   `--run_name`: Identifier for the directory where results and weights will be saved.

### 2. Evaluation

To evaluate a previously trained checkpoint on your validation or testing data:

```bash
# Example: Evaluate model
python src/evaluate.py \
    --model rgb_v3 \
    --stage s2 \
    --data_dir dataset/dataset_v1/val \
    --batch_size 8 \
    --run_name "my_experiment" 
```

The evaluation script locates checkpoints based on defaults (`weights/<model>/<stage>/<run_name>/best_model.pth`), but you can strictly override it via the `--weights path/to/checkpoint.pth` argument. Evaluation output (e.g. confusion matrices, JSON logs) will be deposited in the `results/` folder under your experiment's run name.

### 3. Visualization

You can execute the visualizer to output ground-truth vs. prediction collages on a specific test dataset. Images are saved as plot files, containing the bounded boxes, classification string overlays, and masked regions.

```bash
python src/visualise.py \
    --model rgb_v3 \
    --stage s2 \
    --data_dir dataset/dataset_v1/test \
    --num_samples 10 \
    --run_name "my_experiment"
```

Results are saved to `results/<model_name>/visualise/<run_name>/`.

---