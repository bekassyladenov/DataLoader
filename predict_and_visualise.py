import torch, matplotlib.pyplot as plt, matplotlib.patches as patches
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from weeds_loader import make_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load model
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
in_feats = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes=2)
model.load_state_dict(torch.load("rcnn_mobilenet_epoch10.pth", map_location=device))
model.to(device).eval()

# Inference loader
test_loader = make_loader(
    image_dir="/workspaces/DataLoader/dataset/images/test",
    label_dir="/workspaces/DataLoader/dataset/labels/test",
    batch_size=1, shuffle=False, img_size=(800,800)
)

# Predict & plot first 5
for i, (imgs, _) in enumerate(test_loader):
    if i>=5: break
    img = imgs[0].to(device)
    with torch.no_grad():
        out = model([img])[0]
    fig,ax = plt.subplots(1)
    ax.imshow(img.permute(1,2,0).cpu())
    for box,score in zip(out["boxes"].cpu(), out["scores"].cpu()):
        if score<0.05: continue
        x1,y1,x2,y2 = box
        rect = patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.axis("off"); plt.show()
