"""Step 4/4 -- extract the ResNet-50 features (the ANN scaffold) for every image.

Each image (rows in the order of ``nsd_ids.npy``, step 2) is passed once through an
ImageNet-trained ResNet-50 (torchvision IMAGENET1K_V1).
Activations are read from 15 layers -- every 8th Conv2d and every 8th ReLU (in forward
call order: Conv2d-1, -9, ..., -49 and ReLU-1, -9, ..., -49) plus the global average
pool -- and each layer larger than 3,500 values is reduced to 3,500 dimensions with a
fixed sparse random projection (scikit-learn SparseRandomProjection, eps=0.1,
random_state=42). Concatenated: 14 x 3,500 + 2,048 = 51,048 features per image.

Images: the 73,000 NSD stimuli as distributed by MindEye2
(HuggingFace dataset ``pscotti/mindeyev2``, file ``coco_images_224_float16.hdf5``:
float16 in [0, 1], shape (73000, 3, 224, 224), row = 0-based NSD id), normalised with
the ImageNet mean/std.

The features are saved raw (float32); the training code z-scores every column when it
loads them. Output: ``ann_features.npy`` of shape (70502, 51048). A GPU is recommended.

    python extract_ann_features.py \
        --images_hdf5 /path/to/coco_images_224_float16.hdf5 \
        --nsd_ids $CCN_DATA_ROOT/nsd/data/nsd_ids.npy \
        --out $CCN_DATA_ROOT/nsd/data/ann_features.npy
"""
import argparse
import os
from collections import OrderedDict

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.random_projection import SparseRandomProjection
from torchvision.models import resnet50

LAYER_TYPES = ["Conv2d", "ReLU"]
SAMPLING_RATE = 8                 # keep every 8th layer of each type
TARGET_DIMS = 3500                # sparse-random-projection size per layer
SRP_SEED = 42
# Column layout of the 51,048-d feature vector (layer order of the concatenation).
LAYERS = ["Conv2d-1", "Conv2d-9", "Conv2d-17", "Conv2d-25", "Conv2d-33", "Conv2d-41", "Conv2d-49",
          "ReLU-1", "ReLU-9", "ReLU-17", "ReLU-25", "ReLU-33", "ReLU-41", "ReLU-49",
          "AdaptiveAvgPool2d-1"]
N_FEATURES = 51048

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

_srp_cache = {}


def get_srp(input_dim):
    """One fixed projection per input size (it depends only on the size and the seed)."""
    if input_dim not in _srp_cache:
        srp = SparseRandomProjection(n_components=TARGET_DIMS, eps=0.1, random_state=SRP_SEED)
        srp.fit(np.zeros((1, input_dim), dtype=np.float32))
        _srp_cache[input_dim] = srp
    return _srp_cache[input_dim]


def make_relu_not_inplace(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.ReLU(inplace=False))
        elif len(list(child.children())) > 0:
            make_relu_not_inplace(child)


@torch.no_grad()
def layer_activations(model, x):
    """Forward pass recording every module output, named '<Class>-<k>' by call order
    (a ReLU module reused within a block counts once per call)."""
    outputs = OrderedDict()
    counts = {}

    def hook(module, _inputs, output):
        cls = type(module).__name__
        counts[cls] = counts.get(cls, 0) + 1
        if isinstance(output, torch.Tensor):
            outputs[f"{cls}-{counts[cls]}"] = output.detach().float().cpu()

    hooks = [m.register_forward_hook(hook) for m in model.modules()
             if not isinstance(m, (nn.Sequential, nn.ModuleList))]
    try:
        model(x)
    finally:
        for h in hooks:
            h.remove()
    return outputs


def selected_layers(names):
    """Every SAMPLING_RATE-th Conv2d and ReLU, then the pooling layer (Linear dropped)."""
    chosen = []
    for t in LAYER_TYPES:
        of_type = [n for n in names if t in n]
        chosen += of_type[::SAMPLING_RATE]
    chosen += [n for n in names if "AdaptiveAvgPool2d" in n]
    return chosen


@torch.no_grad()
def extract_batch(model, x):
    acts = layer_activations(model, x)
    chosen = selected_layers(list(acts.keys()))
    if chosen != LAYERS:
        raise RuntimeError(f"unexpected layer selection: {chosen}")
    cols = []
    for name in LAYERS:
        flat = acts[name].reshape(acts[name].shape[0], -1).numpy()
        if flat.shape[1] > TARGET_DIMS:
            flat = get_srp(flat.shape[1]).transform(flat)
        cols.append(np.asarray(flat, dtype=np.float32))
    return np.concatenate(cols, axis=1)


def load_images(images, rows):
    """Read images by 0-based NSD id (h5py needs increasing indices) and normalise."""
    order = np.argsort(rows)
    batch = images[rows[order].tolist()][np.argsort(order)].astype(np.float32)
    return torch.from_numpy((batch - IMAGENET_MEAN) / IMAGENET_STD)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--images_hdf5", required=True, help="MindEye2 coco_images_224_float16.hdf5")
    ap.add_argument("--nsd_ids", required=True, help="nsd_ids.npy from build_index_and_labels.py")
    ap.add_argument("--out", required=True, help="output .npy, e.g. ann_features.npy")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows (for testing)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet50(weights="IMAGENET1K_V1")
    make_relu_not_inplace(model)
    model = model.to(device).eval()

    rows = np.array([int(i) for i in np.load(args.nsd_ids)], dtype=np.int64)
    if args.limit is not None:
        rows = rows[:args.limit]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.float32,
                                    shape=(len(rows), N_FEATURES))
    print(f"[ann] {len(rows)} images on {device} -> {args.out} ({len(rows)}, {N_FEATURES})", flush=True)
    with h5py.File(args.images_hdf5, "r") as f:
        images = f["images"]
        for start in range(0, len(rows), args.batch_size):
            x = load_images(images, rows[start:start + args.batch_size]).to(device)
            feats = extract_batch(model, x)
            out[start:start + len(feats)] = feats
            if (start // args.batch_size) % 100 == 0:
                print(f"  {start + len(feats)}/{len(rows)}", flush=True)
    out.flush()
    print(f"[ann] done: {args.out}")


if __name__ == "__main__":
    main()
