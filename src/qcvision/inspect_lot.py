"""Lot (shipment) inspection: classify every item, then accept or reject the lot."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import IMAGE_EXTS, load_image


@dataclass
class LotReport:
    items: pd.DataFrame
    defect_rate: float
    max_defect_rate: float
    decision: str = field(init=False)

    def __post_init__(self):
        self.decision = "ACCEPT" if self.defect_rate <= self.max_defect_rate else "REJECT"

    def summary(self) -> str:
        n = len(self.items)
        d = int(self.items["defective"].sum())
        return (
            f"Lot decision: {self.decision}  |  {d}/{n} items flagged "
            f"({self.defect_rate:.1%}, tolerance {self.max_defect_rate:.1%})"
        )


def inspect_images(model, images: np.ndarray, names: list[str], threshold: float = 0.5,
                   max_defect_rate: float = 0.05) -> LotReport:
    proba = model.predict_proba(images)[:, 1]
    df = pd.DataFrame({
        "item": names,
        "defect_probability": proba.round(4),
        "defective": proba >= threshold,
    })
    rate = float(df["defective"].mean()) if len(df) else 0.0
    return LotReport(items=df, defect_rate=rate, max_defect_rate=max_defect_rate)


def inspect_folder(model, folder: str | Path, threshold: float = 0.5,
                   max_defect_rate: float = 0.05) -> LotReport:
    folder = Path(folder)
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise ValueError(f"no images in {folder}")
    images = np.stack([load_image(p) for p in paths])
    return inspect_images(model, images, [p.name for p in paths], threshold, max_defect_rate)
