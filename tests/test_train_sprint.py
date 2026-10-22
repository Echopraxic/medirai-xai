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
