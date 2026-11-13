import io

import numpy as np
import pytest
from PIL import Image

from qcvision.data import (DEFECT_TYPES, add_defect, load_folder_dataset,
                           make_inspection_dataset)
from qcvision.inspect_lot import inspect_images
from qcvision.model import ImageFeatures, build_model


def fake_garments(n=200, seed=0):
    """Simple synthetic 'garments': bright rectangles with texture on black background."""
    rng = np.random.default_rng(seed)
    imgs = np.zeros((n, 28, 28), np.uint8)
    for i in range(n):
        h, w = rng.integers(14, 22), rng.integers(10, 18)
        y0, x0 = (28 - h) // 2, (28 - w) // 2
        imgs[i, y0:y0 + h, x0:x0 + w] = rng.integers(150, 230, (h, w))
    return imgs


@pytest.mark.parametrize("kind", DEFECT_TYPES)
def test_defect_changes_image(kind):
    img = fake_garments(1)[0]
    out = add_defect(img, kind, np.random.default_rng(1))
    assert out.shape == img.shape and out.dtype == np.uint8
    assert (out != img).any()


def test_unknown_defect_raises():
    with pytest.raises(ValueError):
        add_defect(fake_garments(1)[0], "scratch", np.random.default_rng(0))


def test_dataset_labels_consistent():
    X, y, kinds = make_inspection_dataset(fake_garments(100), defect_rate=0.3, seed=0)
    assert len(X) == len(y) == len(kinds) == 100
    assert set(kinds[y == 0]) == {"none"}
    assert "none" not in set(kinds[y == 1])


def test_features_shape():
    feats = ImageFeatures().transform(fake_garments(3))
    assert feats.ndim == 2 and feats.shape[0] == 3


def test_cnn_smoke():
    X, y, _ = make_inspection_dataset(fake_garments(200), defect_rate=0.5, seed=0)
    model = build_model("cnn", epochs=1, verbose=False).fit(X, y)
    proba = model.predict_proba(X[:10])
    assert proba.shape == (10, 2)
    assert np.allclose(proba.sum(1), 1, atol=1e-5)
    assert set(model.predict(X[:10])) <= {0, 1}


def test_model_learns_and_lot_decision():
    X, y, _ = make_inspection_dataset(fake_garments(600), defect_rate=0.5, seed=0)
    model = build_model("classic").fit(X[:500], y[:500])
    acc = (model.predict(X[500:]) == y[500:]).mean()
    assert acc > 0.8

    good = fake_garments(20, seed=5)
    report = inspect_images(model, good, [f"g{i}" for i in range(20)], max_defect_rate=0.2)
    assert report.decision == "ACCEPT"
    bad, _, _ = make_inspection_dataset(good, defect_rate=1.0, seed=3)
    report = inspect_images(model, bad, [f"b{i}" for i in range(20)], max_defect_rate=0.2)
    assert report.decision == "REJECT"


def test_folder_dataset(tmp_path):
    for sub, val in (("good", 200), ("defect", 50)):
        (tmp_path / sub).mkdir()
        for i in range(3):
            Image.fromarray(np.full((40, 40), val, np.uint8)).save(tmp_path / sub / f"{i}.png")
    X, y = load_folder_dataset(tmp_path)
    assert X.shape == (6, 28, 28)
    assert list(y) == [0, 0, 0, 1, 1, 1]


def test_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from qcvision import api
    from qcvision.model import save_model

    X, y, _ = make_inspection_dataset(fake_garments(300), defect_rate=0.5, seed=0)
    path = tmp_path / "m.joblib"
    save_model(build_model("classic").fit(X, y), path, {"source": "test"})
    monkeypatch.setenv("QC_MODEL", str(path))
    api.get_model.cache_clear()

    client = TestClient(api.app)
    assert client.get("/health").json() == {"status": "ok"}
    buf = io.BytesIO()
    Image.fromarray(X[0]).save(buf, format="PNG")
    r = client.post("/inspect/item", files={"file": ("a.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200 and "defect_probability" in r.json()
    r = client.post("/inspect/lot", files=[("files", ("a.png", buf.getvalue(), "image/png"))] * 3)
    assert r.json()["decision"] in {"ACCEPT", "REJECT"}
    r = client.post("/inspect/item", files={"file": ("x.txt", b"nope", "text/plain")})
    assert r.status_code == 400
