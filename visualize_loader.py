# File: visualize_loader.py

import os
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from weeds_loader import make_loader

def validate_and_save(loader, save_dir, num_classes=2, max_batches=None, max_images_per_batch=None):
    """
    For each batch in loader (up to max_batches):
      1) run the same sanity checks
      2) save up to max_images_per_batch from each batch into save_dir
    If max_batches or max_images_per_batch is None, it will process/save everything.
    """
    os.makedirs(save_dir, exist_ok=True)

    for batch_idx, (imgs, targets) in enumerate(loader):
        # stop if we've done enough batches
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Sanity-check
        for img in imgs:
            assert isinstance(img, torch.Tensor), "Image not a Tensor"
            assert img.ndim == 3, f"Expected img [C,H,W], got {img.shape}"
            assert 0.0 <= img.min() <= img.max() <= 1.0, "Image values out of [0,1]"
        for tgt in targets:
            assert isinstance(tgt, dict), "Target not a dict"
            assert "boxes" in tgt and "labels" in tgt, "Missing keys in target"

        # Save images with boxes
        for i, (img, tgt) in enumerate(zip(imgs, targets)):
            if max_images_per_batch is not None and i >= max_images_per_batch:
                break

            np_img = img.permute(1,2,0).cpu().numpy()
            fig, ax = plt.subplots(figsize=(6,6))
            ax.imshow(np_img)
            for box in tgt["boxes"].cpu().numpy():
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                ax.add_patch(rect)
            ax.axis('off')

            out_path = os.path.join(save_dir, f"batch{batch_idx:03d}_img{i:03d}.png")
            fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(f"Saved: {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir',   required=True,  help='path to images')
    parser.add_argument('--lbl_dir',   required=True,  help='path to labels')
    parser.add_argument('--save_dir',  default='visualizations', help='where to write PNGs')
    parser.add_argument('--batch_size',type=int, default=4)
    parser.add_argument('--num_classes',type=int, default=2)
    parser.add_argument('--max_batches',type=int, help='how many batches to process; default all')
    parser.add_argument('--max_images', type=int, help='max images per batch; default all')
    args = parser.parse_args()

    loader = make_loader(
        image_dir=args.img_dir,
        label_dir=args.lbl_dir,
        batch_size=args.batch_size,
        shuffle=False
    )
    validate_and_save(
        loader,
        save_dir=args.save_dir,
        num_classes=args.num_classes,
        max_batches=args.max_batches,
        max_images_per_batch=args.max_images
    )
