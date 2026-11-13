"""Mask fusion (segmentation/fuse_masks.py) and the UNet-box prompt for SAM (segmentation/sam_masks.py)."""
import sys
from unittest.mock import MagicMock

import numpy as np

from conftest import ROOT

sys.path.insert(0, str(ROOT / "segmentation"))
sys.modules.setdefault("segment_anything", MagicMock(name="segment_anything"))   # SAM itself is not needed here


def _square(lo, hi, size=40):
    m = np.zeros((size, size), bool)
    m[lo:hi, lo:hi] = True
    return m


def test_fuse_rules():
    import fuse_masks as F
    a, b = _square(10, 30), _square(15, 35)
    assert F.fuse(a, b, "and").sum() == 15 * 15
    assert F.fuse(a, b, "or").sum() == (a | b).sum()
    assert np.array_equal(F.fuse(a, _square(36, 40), "agree"), a)       # disagree -> keep a
    assert F.fuse(a, b, "agree", min_agreement=0.5).sum() == 15 * 15     # agree -> overlap


def test_fuse_never_deletes_the_lesion():
    import fuse_masks as F
    a = _square(5, 15)
    assert np.array_equal(F.fuse(a, _square(25, 35), "and"), a)


def test_unet_box_grows_and_clips():
    import sam_masks as S
    box = S.unet_box(_square(0, 20, size=100))
    assert box[0] == 0 and box[1] == 0 and np.isclose(box[2], 21) and np.isclose(box[3], 21)
    assert S.unet_box(np.zeros((50, 50), bool)) is None
    assert S.unet_box(np.ones((50, 50), bool)) is None        # whole-frame mask is a failure, not a prompt
