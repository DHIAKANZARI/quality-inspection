"""Feature extraction and the defect classifier."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline


class ImageFeatures(BaseEstimator, TransformerMixin):
    """HOG descriptors + local intensity statistics.

    HOG captures edges and shape (tears, holes break the garment's contour);
    the patch statistics capture local darkening or inversion (stains, misprints).
    """

    def __init__(self, pixels_per_cell: int = 4, patch: int = 4):
        self.pixels_per_cell = pixels_per_cell
        self.patch = patch

    def fit(self, X, y=None):
        return self

    def _one(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32) / 255.0
        h = hog(
            img,
            orientations=9,
            pixels_per_cell=(self.pixels_per_cell, self.pixels_per_cell),
            cells_per_block=(2, 2),
            feature_vector=True,
        )
        p = self.patch
        n = img.shape[0] // p
        patches = img[: n * p, : n * p].reshape(n, p, n, p).swapaxes(1, 2).reshape(n * n, p * p)
        mask = img > 0.12
        # holes/tears: background pixels surrounded by garment
        interior_gaps = (~mask[1:-1, 1:-1]) & mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
        stats = np.concatenate([
            patches.mean(1), patches.std(1), patches.min(1),
            [interior_gaps.sum(), img[mask].std() if mask.any() else 0.0],
        ])
        return np.concatenate([h, stats])

    def transform(self, X):
        X = np.asarray(X)
        return np.stack([self._one(img) for img in X])


class CNNClassifier(BaseEstimator):
    """Small convolutional network with a scikit-learn style API (fit / predict_proba)."""

    def __init__(self, epochs: int = 8, batch_size: int = 128, lr: float = 2e-3, seed: int = 0,
                 verbose: bool = True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.verbose = verbose

    @staticmethod
    def _net():
        import torch.nn as nn

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
            )

        return nn.Sequential(
            block(1, 32), nn.MaxPool2d(2),     # 28 -> 14
            block(32, 64), nn.MaxPool2d(2),    # 14 -> 7
            block(64, 96),
            nn.AdaptiveMaxPool2d(1),           # a defect anywhere in the image triggers the feature
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(96, 2),
        )

    @staticmethod
    def _tensor(X):
        import torch
        return torch.from_numpy(np.asarray(X, dtype=np.float32) / 255.0).unsqueeze(1)

    def fit(self, X, y):
        import torch
        import torch.nn.functional as F

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.net_ = self._net()
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        steps = self.epochs * int(np.ceil(len(X) / self.batch_size))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
        Xt, yt = self._tensor(X), torch.as_tensor(np.asarray(y), dtype=torch.long)
        for epoch in range(self.epochs):
            self.net_.train()
            order = rng.permutation(len(Xt))
            total = 0.0
            for i in range(0, len(order), self.batch_size):
                idx = order[i:i + self.batch_size]
                xb, yb = Xt[idx], yt[idx]
                if rng.random() < 0.5:  # horizontal flip augmentation
                    xb = torch.flip(xb, dims=[3])
                loss = F.cross_entropy(self.net_(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                total += loss.item() * len(idx)
            if self.verbose:
                print(f"  epoch {epoch + 1}/{self.epochs}  loss {total / len(Xt):.4f}")
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        import torch

        self.net_.eval()
        out = []
        with torch.no_grad():
            Xt = self._tensor(X)
            for i in range(0, len(Xt), 512):
                out.append(torch.softmax(self.net_(Xt[i:i + 512]), 1).numpy())
        return np.concatenate(out)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_model(kind: str = "cnn", seed: int = 0, **kwargs):
    """``cnn`` (PyTorch, default) or ``classic`` (HOG + gradient boosting baseline)."""
    if kind == "cnn":
        return CNNClassifier(seed=seed, **kwargs)
    if kind == "classic":
        return Pipeline([
            ("features", ImageFeatures()),
            ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=seed)),
        ])
    raise ValueError(f"unknown model kind: {kind}")


def save_model(model, path: str | Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "meta": meta or {}}, path)


def load_model(path: str | Path) -> tuple[object, dict]:
    bundle = joblib.load(path)
    return bundle["model"], bundle["meta"]
