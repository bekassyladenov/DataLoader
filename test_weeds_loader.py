# File: test_weeds_loader.py
import pytest
import torch
from weeds_loader import make_loader

# Adjust these paths to your dataset directories before running tests
def pytest_addoption(parser):
    parser.addoption("--img_dir", action="store", default=" /workspaces/DataLoader/dataset/images/test",
                     help="Directory with training images")
    parser.addoption("--lbl_dir", action="store", default=" /workspaces/DataLoader/dataset/labels/test",
                     help="Directory with training labels")

@pytest.fixture(scope="session")
def loader(request):
    img_dir = request.config.getoption("--img_dir")
    lbl_dir = request.config.getoption("--lbl_dir")
    # Create a small loader for testing
    return make_loader(
        image_dir=img_dir,
        label_dir=lbl_dir,
        batch_size=2,
        shuffle=False
    )

def test_loader_iterates(loader):
    # Ensure we can get at least one batch
    batch = next(iter(loader))
    assert isinstance(batch, tuple) and len(batch) == 2

def test_images_tensor(loader):
    imgs, _ = next(iter(loader))
    for img in imgs:
        assert torch.is_tensor(img), "Image should be a Tensor"
        assert img.ndim == 3, f"Image tensor must be 3D, got {img.shape}"
        assert img.dtype == torch.float32, f"Image dtype should be float32, got {img.dtype}"
        assert 0.0 <= img.min().item() <= img.max().item() <= 1.0, \
               f"Image values should be in [0,1], got min={img.min().item()}, max={img.max().item()}"

def test_targets_structure(loader):
    _, targets = next(iter(loader))
    for tgt in targets:
        assert isinstance(tgt, dict), "Each target must be a dict"
        assert "boxes" in tgt and "labels" in tgt, "Target missing 'boxes' or 'labels'"

def test_boxes_and_labels(loader):
    imgs, targets = next(iter(loader))
    # Use first image's size for bounds
    _, H, W = imgs[0].shape
    for tgt in targets:
        boxes = tgt["boxes"]
        labels = tgt["labels"]
        # Boxes shape
        assert torch.is_tensor(boxes), "boxes must be a Tensor"
        assert boxes.ndim == 2 and boxes.size(1) == 4, f"boxes must be [N,4], got {boxes.shape}"
        # Labels shape
        assert torch.is_tensor(labels), "labels must be a Tensor"
        assert labels.ndim == 1, f"labels must be 1D, got {labels.shape}"
        # Box bounds
        if boxes.numel() > 0:
            x1, y1, x2, y2 = boxes.unbind(1)
            assert (x1 >= 0).all() and (x2 <= W).all(), "Box x-coords out of bounds"
            assert (y1 >= 0).all() and (y2 <= H).all(), "Box y-coords out of bounds"
            assert (x2 > x1).all() and (y2 > y1).all(), "Box x2<=x1 or y2<=y1"
        # Label values
        if labels.numel() > 0:
            assert labels.dtype in (torch.int64, torch.int32), "Labels must be integer type"
            assert labels.min().item() >= 0, "Labels contain negative values"

# To run tests:
# pytest test_weeds_loader.py --img_dir=YOUR/img/path --lbl_dir=YOUR/lbl/path
