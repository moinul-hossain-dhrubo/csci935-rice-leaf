"""Stage 4: ResNet-18 features (512 numbers per image), computed once and saved.

Pretrained torchvision ResNet-18 (ImageNet weights, IMAGENET1K_V1) with its
final classification layer removed; the output of the global average pooling
layer (512 numbers) is the feature vector. The network is never trained.

Each image: read at 1/4 size, BGR -> RGB, turned upright if landscape,
resized to 224 x 448 (w x h, keeps the photos' tall shape; ResNet's global
average pooling accepts any input size), scaled to 0-1 and normalised with
the ImageNet mean and standard deviation the weights were trained with.

The weights (about 45 MB) download automatically on first use to the PyTorch
cache folder (macOS/Linux: ~/.cache/torch/hub/checkpoints/, Windows:
C:\\Users\\<name>\\.cache\\torch\\hub\\checkpoints\\), file resnet18-f37072fd.pth.

Output
  Dhan-Shomadhan/_interim/features_resnet18.npz   X (1106 x 512), files,
                                                  background, disease
  results/logs/04_resnet_features_log.txt

Run from the folder that contains Dhan-Shomadhan/:
  python 04_resnet_features.py
"""
import time

import cv2
import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18

from common import INTERIM_DIR, RESULTS_DIR, load_splits

OUT_FILE = INTERIM_DIR / "features_resnet18.npz"
LOG_FILE = RESULTS_DIR / "logs" / "04_resnet_features_log.txt"
INPUT_SIZE = (224, 448)                     # (width, height)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
BATCH = 32


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")          # Apple-silicon GPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_extractor(device):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()           # drop the 1000-class ImageNet layer
    model.eval()                             # fixed batch-norm statistics, no dropout
    return model.to(device)


def prepare(path):
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
    if img is None:
        raise IOError(f"Cannot read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       # OpenCV reads BGR; the weights expect RGB
    if img.shape[1] > img.shape[0]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)                     # height x width x channel -> channel first


def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    rows = load_splits()
    device = pick_device()
    model = build_extractor(device)
    t0 = time.time()
    feats = []
    with torch.no_grad():                    # no training: no gradients needed
        for start in range(0, len(rows), BATCH):
            batch = np.stack([prepare(r["path"]) for r in rows[start:start + BATCH]])
            out = model(torch.from_numpy(batch).to(device))
            feats.append(out.cpu().numpy().astype(np.float32))
            print(f"  {min(start + BATCH, len(rows))}/{len(rows)} ({time.time() - t0:.0f} s)")
    X = np.concatenate(feats)
    np.savez_compressed(OUT_FILE, X=X,
                        files=np.array([r["file"] for r in rows]),
                        background=np.array([r["background"] for r in rows]),
                        disease=np.array([r["disease"] for r in rows]))
    lines = [
        f"Device: {device}; torch {torch.__version__}",
        f"Weights: ResNet18_Weights.IMAGENET1K_V1; input {INPUT_SIZE[0]} x {INPUT_SIZE[1]} (w x h)",
        f"Images: {X.shape[0]}; features per image: {X.shape[1]}",
        f"Any NaN or inf: {bool(~np.isfinite(X).all())}",
        f"Values: min {X.min():.3f}, max {X.max():.3f}, mean {X.mean():.3f}; "
        f"share of exact zeros {(X == 0).mean():.3f}",
        f"Time: {time.time() - t0:.0f} s",
        f"Saved {OUT_FILE}",
    ]
    for line in lines:
        print(line)
    LOG_FILE.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
