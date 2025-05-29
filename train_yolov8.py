from ultralytics import YOLO

# 1) Load a pretrained small YOLOv8 model
model = YOLO('yolov8s.pt')      # or 'yolov8n.pt' for nano

# 2) Train on your weeds dataset
model.train(
    data='/workspaces/DataLoader/dataset/data.yaml',           # your data config
    imgsz=800,                  # 800×800 input
    epochs=30,                  # train longer if you like
    batch=8,                    # adjust down if out of memory
    name='weeds_yolov8s',       # output folder name
    project='runs_yolo8',
    workers=2
)

# 3) Evaluate on val set (mAP@50, etc.)
metrics = model.val()
print(metrics)
