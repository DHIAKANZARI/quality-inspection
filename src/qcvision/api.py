"""REST API for item and lot inspection.

    uvicorn qcvision.api:app --reload
"""
from __future__ import annotations

import io
import os
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from .data import IMAGE_SIZE
from .inspect_lot import inspect_images
from .model import load_model

app = FastAPI(title="QC Vision", version="1.0.0",
              description="Detect defective items from product photos.")


@lru_cache(maxsize=1)
def get_model():
    path = os.environ.get("QC_MODEL", "models/qc_model.joblib")
    if not os.path.exists(path):
        raise HTTPException(503, f"model not found at {path}; run `python -m qcvision.cli train` first")
    return load_model(path)


async def _read(file: UploadFile) -> np.ndarray:
    try:
        im = Image.open(io.BytesIO(await file.read())).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BOX)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"{file.filename}: not a valid image") from exc
    return np.asarray(im, dtype=np.uint8)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model")
def model_info():
    _, meta = get_model()
    return meta


@app.post("/inspect/item")
async def inspect_item(file: UploadFile = File(...), threshold: float = 0.5):
    model, _ = get_model()
    img = await _read(file)
    p = float(model.predict_proba(img[None])[0, 1])
    return {"item": file.filename, "defect_probability": round(p, 4), "defective": p >= threshold}


@app.post("/inspect/lot")
async def inspect_lot(files: list[UploadFile] = File(...), threshold: float = 0.5,
                      max_defect_rate: float = 0.05):
    model, _ = get_model()
    imgs = np.stack([await _read(f) for f in files])
    report = inspect_images(model, imgs, [f.filename for f in files], threshold, max_defect_rate)
    return {
        "decision": report.decision,
        "defect_rate": round(report.defect_rate, 4),
        "max_defect_rate": max_defect_rate,
        "items": report.items.to_dict(orient="records"),
    }
