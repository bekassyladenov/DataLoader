# Weed Detection Pipeline using Faster R-CNN

This repository implements a PyTorch-based object detection pipeline to identify weeds from images labeled in YOLO format. It includes:

- Custom dataset and dataloader for YOLOv8-format weed annotations
- Data preprocessing and horizontal flip augmentation
- Faster R-CNN training with MobileNetV3 backbone
- Evaluation using torchmetrics (mAP, mAR)
- Visualization of predicted bounding boxes
- Pytest test suite for data loading validation

---

## Project Structure

```
├── weeds_loader.py            # Dataset, transform, and DataLoader
├── train_eval_rcnn.py         # Full training loop with eval + checkpointing
├── visualise_loader.py        # DataLoader visualization with boxes
├── predict_and_visualize.py   # Inference and bounding box plot
├── test_weeds_loader.py       # Pytest coverage of DataLoader integrity
├── dataset/
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
```

---

## Setup

Install the required dependencies:

```bash
pip install torch torchvision torchmetrics opencv-python matplotlib pytest
```

---

## Usage

### 1. Prepare your dataset

- Place `.jpg` or `.png` files into `dataset/images/{split}/`
- Place corresponding YOLO `.txt` labels into `dataset/labels/{split}/`
- YOLO format: `class cx cy width height` (normalized)

### 2. Train and Evaluate

```bash
python train_eval_rcnn.py \
  --train_img_dir dataset/images/train \
  --train_lbl_dir dataset/labels/train \
  --val_img_dir dataset/images/val \
  --val_lbl_dir dataset/labels/val \
  --batch_size 4 \
  --num_epochs 20 \
  --lr 1e-4 \
  --output_dir outputs
```

### 3. Visualize Data Batches

```bash
python visualise_loader.py \
  --img_dir dataset/images/val \
  --lbl_dir dataset/labels/val \
  --save_dir visualizations \
  --batch_size 4 \
  --max_batches 2 \
  --max_images 3
```

### 4. Run Inference & Plot

```bash
python predict_and_visualize.py
```

---

## Testing

Run all unit tests:

```bash
pytest test_weeds_loader.py \
  --img_dir=dataset/images/test \
  --lbl_dir=dataset/labels/test
```

Tests include:
- Image format and normalization checks
- Bounding box shape and validity
- Label formatting and values

---

## Model Details

- **Architecture**: Faster R-CNN with FPN
- **Backbone**: MobileNetV3-Large
- **Input Size**: 800 × 800
- **Classes**: 2 (background, weed)

---

## Author

Bekassyl Adenov  
Bachelor Thesis, Riga Technical University  
"Weed Detection in Tree Nurseries Using Deep Learning"

