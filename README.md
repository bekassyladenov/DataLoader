🌱 Weed Detection Pipeline using Faster R-CNN (with Augmentations & Evaluation)
This repository provides a clean and modular PyTorch pipeline for training and evaluating a Faster R-CNN object detection model on a weed-only dataset annotated in YOLO format. It includes data loading, basic augmentation, training with evaluation, prediction visualization, and unit testing.

🧩 Features
Custom Dataset class for YOLOv8-format weed annotations

Image resizing + bounding box conversion

Horizontal flip augmentation (configurable)

Training script with per-epoch evaluation using mAP

Evaluation script with torchmetrics

Batch-wise prediction visualization with bounding boxes

Automated test coverage with pytest


⚙️ Setup
Install dependencies:

pip install torch torchvision torchmetrics opencv-python matplotlib pytest
🚀 Quickstart
1. Prepare the Dataset
Ensure:

Images are in /images/[split]/ (e.g., .jpg, .png)

Corresponding YOLO .txt label files are in /labels/[split]/

Each .txt file: class cx cy w h (normalized)

2. Train and Evaluate per Epoch

python train_eval_rcnn.py \
  --train_img_dir dataset/images/train \
  --train_lbl_dir dataset/labels/train \
  --val_img_dir dataset/images/val \
  --val_lbl_dir dataset/labels/val \
  --batch_size 4 \
  --num_epochs 20 \
  --lr 1e-4 \
  --output_dir outputs
Checkpointed models and best model (by mAP@0.5) will be saved in outputs/.

3. Visualize DataLoader with Annotations

python visualise_loader.py \
  --img_dir dataset/images/val \
  --lbl_dir dataset/labels/val \
  --save_dir visualizations \
  --batch_size 4 \
  --max_batches 2 \
  --max_images 3
4. Run Predictions & Show Results
bash
Copy
Edit
python predict_and_visualize.py
Shows the first 5 test images with predicted boxes.

🧠 Model Architecture
Backbone: mobilenet_v3_large_fpn

Detector: Faster R-CNN

Input Size: (800, 800)

Classes: 2 (background + weed)

📌 Notes
This project is optimized for small datasets (tree nursery weeds).

Built-in transform: horizontal flip (p=0.5)

Extendable with Albumentations or torchvision transforms.

👨‍🔬 Author
Bekassyl Adenov
Bachelor Thesis: Riga Technical University
Topic: Tree nurseries weed RGB’s camera raw data pre-processing and preparing a dataset for deep learning tasks
