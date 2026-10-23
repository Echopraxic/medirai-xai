"""Classifier sprint trainer (classifier/train_sprint.py): feature hook safety under EMA copies, EMA BatchNorm stats,
melanin-like augmentation."""
import copy
import sys

import pytest
import torch

from conftest import ROOT, STUBBED

pytestmark = pytest.mark.skipif("torchvision" in STUBBED, reason="needs torchvision")
sys.path.insert(0, str(ROOT / "classifier"))


def test_resnet_features_come_from_the_module_that_ran():
    import train_sprint as T
    m = T.ResNetBinary().eval()
    twin = copy.deepcopy(m)                        # what AveragedModel does
    with torch.no_grad():
        head = [mod for mod in twin.net.modules() if isinstance(mod, torch.nn.Linear)][-1]
        head.weight.mul_(0)                        # change the copy's classifier head only
        x = torch.rand(2, 3, 64, 64)
        logits_a, f_a = m(x)
        logits_b, f_b = twin(x)
    assert f_a.shape == (2, 2048) and torch.allclose(f_a, f_b)     # same backbone -> same features, from each module
    assert not torch.allclose(logits_a, logits_b)
    assert len(m.pool._forward_hooks) == 0 and len(twin.pool._forward_hooks) == 0   # hooks removed after each call


def test_ema_averages_batchnorm_statistics():
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4))
    ema = torch.optim.swa_utils.AveragedModel(m, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.5),
                                              use_buffers=True)
    m.train()
    for _ in range(5):
        m(torch.randn(8, 3, 4, 4) * 3 + 2)
        ema.update_parameters(m)
    assert not torch.allclose(ema.module[1].running_mean, torch.zeros(4))   # would stay at init without use_buffers


def test_melanin_darken_reduces_blue_most():
    import train_sprint as T
    torch.manual_seed(0)
    x = torch.full((3, 4, 4), 0.8)
    aug = T.MelaninDarken(p=1.0)
    y = aug(x)
    r, g, b = (y[c].mean().item() for c in range(3))
    assert b < g < r < 0.8 and y.min() >= 0


def test_kd_loss_is_zero_when_student_matches_teacher_and_scales_with_t2():
    import train_sprint as T
    logits = torch.tensor([[2.0, -1.0], [0.5, 0.3]])
    for t in (1.0, 2.0):
        same = torch.softmax(logits / t, 1)
        assert T.kd_loss(logits, same, t).abs() < 1e-6
    other = torch.tensor([[0.5, 0.5], [0.9, 0.1]])
    assert T.kd_loss(logits, other, 2.0) > 0


def test_ensemble_teacher_rejects_mismatched_input_size(tmp_path):
    import json
    import train_sprint as T
    (tmp_path / "config.json").write_text(json.dumps({"trainer": "train_sprint", "input_size": 384,
                                                      "normalization": "medir"}))
    with pytest.raises(ValueError):
        T.EnsembleTeacher([str(tmp_path)], input_size=224)


def test_pair_dataset_gives_both_images_the_same_augmentation(tmp_path):
    import numpy as np
    import pandas as pd
    from PIL import Image
    import train_sprint as T
    rng = np.random.default_rng(0)
    arr = (rng.random((40, 40, 3)) * 255).astype("uint8")
    for name in ("a", "b"):                            # identical pixels: only the augmentation could differ
        Image.fromarray(arr).save(tmp_path / f"{name}.jpg", quality=100)
    train_tf, _ = T.make_transforms(32, T.MEDIR_STATS)
    ds = T.PairDataset(pd.DataFrame({"orig_id": ["a"], "synth_id": ["b"]}), tmp_path, train_tf)
    torch.manual_seed(1)
    xa, xb = ds[0]
    assert torch.allclose(xa, xb)
    torch.manual_seed(2)
    xa2, _ = ds[0]
    assert not torch.allclose(xa, xa2)                 # augmentation still random across calls


def test_symmetric_kl_is_zero_for_equal_logits_and_symmetric():
    import train_sprint as T
    a = torch.tensor([[2.0, -1.0], [0.1, 0.4]])
    b = torch.tensor([[0.0, 1.0], [1.0, -2.0]])
    assert T.symmetric_kl(a, a).abs() < 1e-6
    assert torch.allclose(T.symmetric_kl(a, b), T.symmetric_kl(b, a)) and T.symmetric_kl(a, b) > 0
