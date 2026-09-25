"""splits/check_images.py: detects missing/corrupt images and repairs them (network mocked)."""
import io
import sys

import numpy as np
import pandas as pd
from PIL import Image

from conftest import ROOT

sys.path.insert(0, str(ROOT / "splits"))
import check_images as ci  # noqa: E402


def _jpeg_bytes(seed=0):
    buf = io.BytesIO()
    arr = np.random.default_rng(seed).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


class _Resp:
    def __init__(self, content=b"", payload=None, headers=None):
        self.content, self._payload, self.headers = content, payload, headers or {}

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeISIC:
    """Mimics the ISIC API + S3: API 'size' excludes ~3 KB of XMP that S3 actually serves."""
    def __init__(self, truncate=False):
        self.truncate = truncate

    def get(self, url, timeout=None):
        if "api.isic-archive.com" in url:
            isic_id = url.rstrip("/").split("/")[-1]
            return _Resp(payload={"files": {"full": {"url": f"https://s3/{isic_id}.jpg", "size": 1}}})
        body = _jpeg_bytes()
        served = body[: len(body) // 2] if self.truncate else body
        return _Resp(content=served, headers={"Content-Length": str(len(body))})


def _split(tmp_path, ids):
    f = tmp_path / "split.csv"
    pd.DataFrame({"isic_id": ids, "split": ["train"] * len(ids)}).to_csv(f, index=False)
    return f


def test_detects_missing_and_corrupt(tmp_path):
    (tmp_path / "ISIC_1.jpg").write_bytes(_jpeg_bytes())
    good = _jpeg_bytes()
    (tmp_path / "ISIC_2.jpg").write_bytes(good[: len(good) // 2])  # truncated
    status = ci.check(_split(tmp_path, ["ISIC_1", "ISIC_2", "ISIC_3"]), tmp_path).set_index("isic_id")["status"]
    assert status.to_dict() == {"ISIC_1": "ok", "ISIC_2": "corrupt", "ISIC_3": "missing"}


def test_download_ignores_api_size_but_checks_content_length(tmp_path, monkeypatch):
    monkeypatch.setattr(ci.time, "sleep", lambda s: None)
    assert ci.download("ISIC_9", tmp_path, _FakeISIC()) == "ok"          # API size (1) is ignored
    assert ci.download("ISIC_8", tmp_path, _FakeISIC(truncate=True), retries=1) == "download_failed"
    assert not (tmp_path / "ISIC_8.jpg").exists()                       # no half-written file left behind
