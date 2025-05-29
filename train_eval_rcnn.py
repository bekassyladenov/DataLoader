# File: train_eval_rcnn.py

import os
import time
import argparse

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from weeds_loader import make_loader

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    running_loss = 0.0
    for imgs, targets in data_loader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    avg_loss = running_loss / len(data_loader)
    print(f"Epoch {epoch:>2} — train loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
    with torch.no_grad():
        for imgs, targets in data_loader:
            imgs = [img.to(device) for img in imgs]
            gt = [{"boxes": t["boxes"].to(device),
                   "labels": t["labels"].to(device)} for t in targets]
            outputs = model(imgs)
            preds = [{
                "boxes": out["boxes"].cpu(),
                "scores": out["scores"].cpu(),
                "labels": out["labels"].cpu()
            } for out in outputs]

            metric.update(preds, gt)

    results = metric.compute()
    map50 = results["map_50"].item()
    map50_95 = results["map"].item()
    print(f"       — mAP@0.50:      {map50:.4f}")
    print(f"       — mAP@0.50–0.95: {map50_95:.4f}")
    return map50, map50_95

def main():
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN and evaluate per epoch"
    )
    parser.add_argument("--train_img_dir",  required=True)
    parser.add_argument("--train_lbl_dir",  required=True)
    parser.add_argument("--val_img_dir",    required=True)
    parser.add_argument("--val_lbl_dir",    required=True)
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--num_epochs",     type=int, default=20)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--num_classes",    type=int,   default=2,
                        help="including background")
    parser.add_argument("--output_dir",     default="outputs")
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Data loaders
    train_loader = make_loader(
        image_dir=args.train_img_dir,
        label_dir=args.train_lbl_dir,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = make_loader(
        image_dir=args.val_img_dir,
        label_dir=args.val_lbl_dir,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Model & optimizer
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, args.num_classes)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training + evaluation loop
    best_map50 = 0.0
    for epoch in range(1, args.num_epochs + 1):
        start = time.time()
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        train_time = time.time() - start

        print(f"  Evaluating after epoch {epoch} (took {train_time:.1f}s)...")
        map50, map50_95 = evaluate(model, val_loader, device)

        # Save checkpoint
        ckpt_path = os.path.join(
            args.output_dir, f"fasterrcnn_epoch{epoch}.pth"
        )
        torch.save(model.state_dict(), ckpt_path)

        # Track best
        if map50 > best_map50:
            best_map50 = map50
            best_path = os.path.join(args.output_dir, "best_fasterrcnn.pth")
            torch.save(model.state_dict(), best_path)

    print(f"\n➡️  Finished. Best mAP@0.50: {best_map50:.4f} (checkpoint saved to {best_path})")

if __name__ == "__main__":
    main()
