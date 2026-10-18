import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "classifier"))
import train_multiclass as mc  # noqa: E402


def test_other_is_split_by_label_and_malignant_prob_sums_malignant_classes():
    assert mc.class_of("Melanoma in situ", 1) == "Melanoma"
    assert mc.class_of("Keratoacanthoma", 1) == "Other-malignant"
    assert mc.class_of("Dermatofibroma", 0) == "Other-benign"
    assert all(mc.CLASSES[i] not in {"Nevus", "SK", "Other-benign"} for i in mc.MALIGNANT)
    logits = torch.zeros(1, len(mc.CLASSES))
    assert abs(mc.malignant_prob(logits).item() - len(mc.MALIGNANT) / len(mc.CLASSES)) < 1e-6


def test_class_weights_upweight_rare_classes_and_keep_mean_loss_scale():
    y = np.array([0] * 90 + [2] * 10)
    w = mc.class_weights(y)
    assert w[2] > w[0]
    counts = np.bincount(y, minlength=len(mc.CLASSES))
    assert abs((w.numpy() * counts).sum() - counts.sum()) < 1e-3
