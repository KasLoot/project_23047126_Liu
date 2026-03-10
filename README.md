# Hand Gesture Multi-Task Pipeline (Reorganized)

This project is reorganized so the active code is in `src/` with one script per task:

- `src/dataloader.py`: dataset loading, resizing, augmentation (`SegAugment_v2`), collate function.
- `src/model.py`: full multi-task model definition (`RGB_V2`) for detection + classification + segmentation.
- `src/utils.py`: shared losses, anchor/box utilities, metrics, logging, and helper functions.
- `src/train.py`: unified training entry point for Stage-1 (`s1`) and Stage-2 (`s2`).
- `src/evaluate.py`: quantitative evaluation on a dataset split (metrics + confusion matrix).
- `src/visualise.py`: qualitative visualization images from model predictions.

## Output Folders

- Model checkpoints are saved to: `weights/`
	- `weights/s1_best_model.pth`, `weights/s1_last_model.pth`
	- `weights/s2_best_model.pth`, `weights/s2_last_model.pth`
- Training/evaluation/visualization artifacts are saved to: `results/`
	- training curves/logs/history: `results/training/...`
	- evaluation reports: `results/evaluation/...`
	- qualitative images: `results/visualise/...`

## Dataset Layout Expected

Each split directory (for example `dataset/dataset_v1/train`) should contain:

- `image_tensors/`
- `mask_tensors/`
- `image_info.json`

## How to Run

Run all commands from project root.

### 1) Stage-1 Training (Detection-focused)

```bash
python src/train.py --stage s1 --run_name exp_s1
```

Optional overrides:

```bash
python src/train.py --stage s1 --run_name exp_s1 --epochs 50 --batch_size 16 --lr 1e-3
```

### 2) Stage-2 Training (Classification + Segmentation Fine-tune)

By default, Stage-2 loads `weights/s1_best_model.pth`.

```bash
python src/train.py --stage s2 --run_name exp_s2
```

If your Stage-1 best checkpoint is elsewhere:

```bash
python src/train.py --stage s2 --run_name exp_s2 --stage1_weights weights/s1_best_model.pth
```

### 3) Evaluation

```bash
python src/evaluate.py --weights weights/s2_best_model.pth --data_dir dataset/dataset_v1/test --run_name eval_s2
```

Main outputs:

- `results/evaluation/eval_s2/eval.txt`
- `results/evaluation/eval_s2/metrics.json`
- `results/evaluation/eval_s2/classification_confusion_matrix.png`

### 4) Prediction Visualisation

```bash
python src/visualise.py --weights weights/s2_best_model.pth --data_dir dataset/dataset_v1/test --num_samples 10 --run_name vis_s2
```

Images are written to `results/visualise/vis_s2/`.

### 5) (Optional) Augmentation Preview

```bash
python src/dataloader.py --root_dir dataset/dataset_v1/train --num_samples 8 --out_dir results/visualise/augmented_preview

### 6) Data Utilities Menu

Run:

```bash
python src/utils.py
```

You will see a terminal menu:

- `1) Preprocess data`
- `2) Show data distribution`
- `0) Exit`

If you choose `Preprocess data`, it asks for:

- origin dataset folder path
- output dataset folder path

Then it runs, in order:

- `gether_images_and_masks`
- `image_to_tensor`
- `balance_data_distribution`
- `get_class_distribution`

## Legacy Wrappers

- `preprocess_data.py` now delegates to shared functions in `src/utils.py`.
- `show_data_distribution.py` now delegates to shared functions in `src/utils.py` and saves plots to `results/data_distribution_analysis`.
```

## Notes

- Default input resolution is `480 x 640`.
- Number of classes is set to `10` by default.
- If CUDA is available, scripts automatically use GPU.
