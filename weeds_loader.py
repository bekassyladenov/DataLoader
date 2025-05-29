import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class WeedDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=(800,800), transforms=None):
        self.image_dir  = image_dir
        self.label_dir  = label_dir
        self.img_size   = img_size
        self.images     = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png')))
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        fn     = self.images[idx]
        img_path = os.path.join(self.image_dir, fn)
        img    = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        # Resize
        if self.img_size:
            img = cv2.resize(img, self.img_size)
        img_t = torch.from_numpy(img).permute(2,0,1).float().div(255.0)

        # Load labels (YOLO format: cls cx cy w h)
        boxes, labels = [], []
        lbl_path = os.path.join(self.label_dir, fn.rsplit('.',1)[0] + '.txt')
        if os.path.isfile(lbl_path):
            for line in open(lbl_path):
                cls, cx, cy, bw, bh = map(float, line.split())
                cx,cy,bw,bh = cx*w0, cy*h0, bw*w0, bh*h0
                x1,y1 = cx - bw/2, cy - bh/2
                x2,y2 = cx + bw/2, cy + bh/2
                # Scale to resized image
                if self.img_size:
                    sx, sy = self.img_size[0]/w0, self.img_size[1]/h0
                    x1,x2 = x1*sx, x2*sx
                    y1,y2 = y1*sy, y2*sy
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls))
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        # Apply optional transform (e.g., horizontal flip)
        if self.transforms is not None:
            img_t, target = self.transforms(img_t, target)

        return img_t, target

# Simple horizontal flip transform
import random

def hflip_transform(img, target, p=0.5):
    if random.random() < p:
        _, H, W = img.shape
        img = img.flip(-1)
        boxes = target["boxes"]
        x1 = W - boxes[:, 2]
        x2 = W - boxes[:, 0]
        target["boxes"] = torch.stack([x1, boxes[:,1], x2, boxes[:,3]], dim=1)
    return img, target

# DataLoader factory

def make_loader(image_dir, label_dir, batch_size=4, shuffle=True, img_size=(800,800)):
    ds = WeedDataset(image_dir, label_dir, img_size=img_size, transforms=hflip_transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: tuple(zip(*batch))
    )