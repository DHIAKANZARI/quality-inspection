"""Dataset loading and synthetic defect generation.

Two data sources are supported:

* ``fashion``: Fashion-MNIST garment images. Defects (holes, stains, tears,
  print errors) are injected synthetically so the pipeline can be trained and
  demonstrated without a proprietary dataset.
* ``folder``: any product. Put images in ``<root>/good/`` and
  ``<root>/defect/`` and the same pipeline trains on them.
"""
from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

FASHION_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FASHION_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
DEFECT_TYPES = ("hole", "stain", "tear", "misprint")
IMAGE_SIZE = 28


# --------------------------------------------------------------------------- #
# Fashion-MNIST
# --------------------------------------------------------------------------- #
def _download(name: str, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / FASHION_FILES[name]
    if not path.exists():
        urllib.request.urlretrieve(FASHION_URL + FASHION_FILES[name], path)
    return path


def load_fashion_mnist(cache: str | Path = "data/fashion", split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """Return ``(images, garment_labels)``; images are uint8 arrays of shape (n, 28, 28)."""
    cache = Path(cache)
    with gzip.open(_download(f"{split}_images", cache), "rb") as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
    with gzip.open(_download(f"{split}_labels", cache), "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return images.copy(), labels.copy()


# --------------------------------------------------------------------------- #
# Synthetic defects
# --------------------------------------------------------------------------- #
def _garment_mask(img: np.ndarray) -> np.ndarray:
    return img > 30


def _random_point_on_garment(img: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    ys, xs = np.nonzero(_garment_mask(img))
    if len(ys) == 0:
        return IMAGE_SIZE // 2, IMAGE_SIZE // 2
    i = rng.integers(len(ys))
    return int(ys[i]), int(xs[i])


def add_defect(img: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of ``img`` with one defect of type ``kind`` applied."""
    out = img.astype(np.float32).copy()
    yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    cy, cx = _random_point_on_garment(img, rng)

    if kind == "hole":
        r = rng.uniform(1.5, 3.0)
        out[(yy - cy) ** 2 + (xx - cx) ** 2 <= r**2] = 0
    elif kind == "stain":
        r = rng.uniform(2.0, 4.0)
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r**2))
        darken = rng.uniform(0.45, 0.7)
        out = out * (1 - darken * blob * _garment_mask(img))
    elif kind == "tear":
        length = rng.integers(6, 12)
        angle = rng.uniform(0, np.pi)
        for t in np.linspace(-length / 2, length / 2, length * 3):
            y = int(round(cy + t * np.sin(angle)))
            x = int(round(cx + t * np.cos(angle)))
            if 0 <= y < IMAGE_SIZE and 0 <= x < IMAGE_SIZE:
                out[y, x] = 0
    elif kind == "misprint":
        h, w = rng.integers(3, 6), rng.integers(3, 6)
        y0 = int(np.clip(cy - h // 2, 0, IMAGE_SIZE - h))
        x0 = int(np.clip(cx - w // 2, 0, IMAGE_SIZE - w))
        out[y0:y0 + h, x0:x0 + w] = 255 - out[y0:y0 + h, x0:x0 + w]
    else:
        raise ValueError(f"unknown defect type: {kind}")
    return np.clip(out, 0, 255).astype(np.uint8)


def make_inspection_dataset(
    images: np.ndarray, defect_rate: float = 0.5, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inject defects into a fraction of ``images``.

    Returns ``(images, is_defect, defect_type)`` where ``defect_type`` is ``"none"``
    for good items.
    """
    rng = np.random.default_rng(seed)
    n = len(images)
    is_defect = rng.random(n) < defect_rate
    kinds = np.array(["none"] * n, dtype=object)
    out = images.copy()
    for i in np.nonzero(is_defect)[0]:
        kind = DEFECT_TYPES[rng.integers(len(DEFECT_TYPES))]
        out[i] = add_defect(images[i], kind, rng)
        kinds[i] = kind
    return out, is_defect.astype(int), kinds


# --------------------------------------------------------------------------- #
# Folder datasets (any product)
# --------------------------------------------------------------------------- #
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_image(path: str | Path, size: int = IMAGE_SIZE) -> np.ndarray:
    """Load an image as a grayscale uint8 array resized to ``size`` x ``size``."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize((size, size), Image.Resampling.BOX), dtype=np.uint8)


def load_folder_dataset(root: str | Path, size: int = IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Load ``<root>/good`` (label 0) and ``<root>/defect`` (label 1)."""
    root = Path(root)
    images, labels = [], []
    for label, sub in ((0, "good"), (1, "defect")):
        folder = root / sub
        if not folder.is_dir():
            raise FileNotFoundError(f"missing folder: {folder}")
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                images.append(load_image(p, size))
                labels.append(label)
    if not images:
        raise ValueError(f"no images found under {root}")
    return np.stack(images), np.array(labels)
