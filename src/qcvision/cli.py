"""Command line interface.

    python -m qcvision.cli train   [--source fashion|folder] [--data PATH]
    python -m qcvision.cli inspect --lot PATH_TO_FOLDER
    python -m qcvision.cli demo-lot --out data/demo_lot
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from .data import (DEFECT_TYPES, load_fashion_mnist, load_folder_dataset,
                   make_inspection_dataset)
from .inspect_lot import inspect_folder
from .model import build_model, load_model, save_model

DEFAULT_MODEL = "models/qc_model.joblib"


def cmd_train(args) -> None:
    t0 = time.time()
    kinds_test = None
    if args.source == "fashion":
        imgs, _ = load_fashion_mnist(args.data, "train")
        imgs = imgs[: args.n_train]
        X, y, _ = make_inspection_dataset(imgs, defect_rate=0.5, seed=args.seed)
        timgs, _ = load_fashion_mnist(args.data, "test")
        X_test, y_test, kinds_test = make_inspection_dataset(timgs[: args.n_test], 0.5, seed=args.seed + 1)
    else:
        X_all, y_all = load_folder_dataset(args.data)
        X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all,
                                                random_state=args.seed)

    print(f"Training on {len(X)} images ({y.mean():.0%} defective)...")
    extra = {"epochs": args.epochs} if args.model_kind == "cnn" else {}
    model = build_model(args.model_kind, seed=args.seed, **extra).fit(X, y)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "source": args.source,
        "model": args.model_kind,
        "n_train": int(len(X)),
        "n_test": int(len(X_test)),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "train_seconds": round(time.time() - t0, 1),
    }
    if kinds_test is not None:
        metrics["recall_by_defect_type"] = {
            k: round(float(pred[kinds_test == k].mean()), 4) for k in DEFECT_TYPES
        }
        metrics["false_alarm_rate"] = round(float(pred[kinds_test == "none"].mean()), 4)

    print(classification_report(y_test, pred, target_names=["good", "defect"]))
    print(json.dumps(metrics, indent=2))

    save_model(model, args.model, meta=metrics)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["good", "defect"],
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Test set confusion matrix")
    fig.tight_layout()
    fig.savefig(reports / "confusion_matrix.png", dpi=120)
    print(f"Model saved to {args.model}")


def cmd_demo_lot(args) -> None:
    """Write a folder of garment images with some defects, to try `inspect` on."""
    imgs, _ = load_fashion_mnist(args.data, "test")
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(imgs), args.size, replace=False)
    X, y, kinds = make_inspection_dataset(imgs[pick], defect_rate=args.defect_rate, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, (img, k) in enumerate(zip(X, kinds)):
        Image.fromarray(img).resize((112, 112), Image.NEAREST).save(out / f"item_{i:03d}_{k}.png")
    print(f"Wrote {len(X)} items to {out} ({y.sum()} with defects)")


def cmd_inspect(args) -> None:
    model, _ = load_model(args.model)
    report = inspect_folder(model, args.lot, threshold=args.threshold,
                            max_defect_rate=args.max_defect_rate)
    print(report.items.to_string(index=False))
    print()
    print(report.summary())
    if args.csv:
        report.items.to_csv(args.csv, index=False)
        print(f"Report written to {args.csv}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="qcvision", description="Visual quality inspection")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train the defect classifier")
    t.add_argument("--source", choices=["fashion", "folder"], default="fashion")
    t.add_argument("--data", default="data/fashion",
                   help="cache dir for fashion, or folder with good/ and defect/")
    t.add_argument("--model-kind", choices=["cnn", "classic"], default="cnn")
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--n-train", type=int, default=20000)
    t.add_argument("--n-test", type=int, default=4000)
    t.add_argument("--model", default=DEFAULT_MODEL)
    t.add_argument("--reports", default="reports")
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train)

    d = sub.add_parser("demo-lot", help="generate a sample shipment folder")
    d.add_argument("--data", default="data/fashion")
    d.add_argument("--out", default="data/demo_lot")
    d.add_argument("--size", type=int, default=40)
    d.add_argument("--defect-rate", type=float, default=0.1)
    d.add_argument("--seed", type=int, default=7)
    d.set_defaults(func=cmd_demo_lot)

    i = sub.add_parser("inspect", help="inspect a lot (folder of images)")
    i.add_argument("--lot", required=True)
    i.add_argument("--model", default=DEFAULT_MODEL)
    i.add_argument("--threshold", type=float, default=0.5)
    i.add_argument("--max-defect-rate", type=float, default=0.05)
    i.add_argument("--csv")
    i.set_defaults(func=cmd_inspect)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
