"""
Regression tests for the CODEBASE_TODO P1 fixes merged in PR #1 (b0c35cf). CPU-only, synthetic data.
Run from the workspace root:  python -m pytest tests -q

Not covered here because the logic lives inline in the train.py script (no importable unit):
P1-2 (epoch averaging), P1-4 (train/val objective), P1-12 (fabric optimizer), P1-14 (fusion train loop).
"""
import importlib.util
import json

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from conftest import ROOT, UNC


def _unc_utils():
    import utils  # uncertaintyNet-main/utils.py
    assert hasattr(utils, "calculate_dice")
    return utils


# --------------------------------------------------------------------------- P1-9: NLL path
def test_nll_accuracy_and_entropy_accept_log_probs():
    utils = _unc_utils()
    log_probs = torch.log_softmax(torch.tensor([[2.0, 0.0], [0.0, 3.0], [1.0, 0.0]]), dim=-1)
    target = torch.tensor([0, 1, 1])
    assert utils.calculate_accuracy(log_probs, target, nll_loss=True) == pytest.approx(2 / 3)  # crashed: exp(x, dim=-1)
    eoe, ee = utils.predictive_entropy(log_probs.unsqueeze(0), nll_loss=True, task="classification")
    assert np.isfinite(eoe) and eoe > 0


# --------------------------------------------------------------------------- P1-10: Dice
def test_dice_is_the_set_overlap_ratio():
    utils = _unc_utils()
    a = torch.zeros(1, 1, 4, 4)
    a[..., :2, :] = 1                       # 8 pixels
    b = torch.zeros(1, 1, 4, 4)
    b[..., :1, :] = 1                       # 4 pixels, all inside a
    assert float(utils.calculate_dice(a, a)) == pytest.approx(1.0)
    assert float(utils.calculate_dice(a, b)) == pytest.approx(2 * 4 / (8 + 4), abs=1e-4)   # 0.667
    assert float(utils.calculate_dice(a, 1 - a)) == pytest.approx(0.0, abs=1e-4)
    # the old per-pixel ratio gave ~1 for empty-vs-empty pixels and so overstated disjoint masks


def test_dice_loss_is_differentiable():
    utils = _unc_utils()
    pred = torch.full((2, 1, 8, 8), 0.5, requires_grad=True)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()
    loss = utils.DiceLoss()(pred, target)
    loss.backward()                           # .item() used to cut the graph
    assert pred.grad is not None and pred.grad.abs().sum() > 0


# --------------------------------------------------------------------------- P1-11: confidence fields
def test_test_logging_keys_separate_predicted_and_true_confidence():
    utils = _unc_utils()
    _, keys = utils.get_logging_keys("deterministic", "classification", test=True, dataset="medirv2")
    assert "Y_pred_conf" in keys and "Y_true_conf" in keys and "Y_conf" not in keys


# --------------------------------------------------------------------------- P1-3 / P1-6: configs
@pytest.mark.parametrize("cfg_path", sorted((UNC / "configs").glob("*.json")), ids=lambda p: p.name)
def test_configs_use_val_frequency_and_build(cfg_path, monkeypatch):
    from models import model
    cfg = json.loads(cfg_path.read_text())
    data = cfg["Training"]["Dataset"]
    assert "val_frequency" in data and "val_patience" not in data
    net = cfg["Network"]["Basic Setup"]
    assert net["network_type"] in {"deterministic", "bayesian", "partial_bayesian"}
    for name in ("ResNet18Hidden", "ResNet50Hidden", "WideResNet50Hidden", "UNet"):
        monkeypatch.setattr(model, name, lambda *a, **k: nn.Identity())  # no weight downloads
    model.get_model(net)  # "resunet" used to raise NotImplementedError


# --------------------------------------------------------------------------- P1-7 / P1-5: variational models
BAYES_CFG = {"prior_mu": 0, "prior_variance": 1, "posterior_mu": 0, "posterior_rho": -3, "num_samples": 2}


def test_fully_bayesian_conversion_skips_unnamed_batchnorm(tmp_path):
    from torchvision.models import resnet18
    from models import model_utils
    from models import bayes_layers
    net = resnet18(weights=None, num_classes=2)   # has BatchNorm at "layer2.0.downsample.1"
    vnet = model_utils.build_variational_model(net, "bayesian", str(tmp_path), BAYES_CFG, {}, {}, save_init=False)
    mods = list(vnet.modules())
    assert any(isinstance(m, bayes_layers.Conv2dReparameterization) for m in mods)
    assert not any(type(m) in (nn.Conv2d, nn.Linear) for m in mods)      # every conv/linear converted
    assert any(isinstance(m, nn.BatchNorm2d) for m in mods)               # BatchNorm left alone


def test_mean_posterior_sigma_matches_rho_init(tmp_path):
    from models import model_utils
    net = nn.Sequential(nn.Flatten(), nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    vnet = model_utils.build_variational_model(net, "bayesian", str(tmp_path), BAYES_CFG, {}, {}, save_init=False)
    expected = float(torch.nn.functional.softplus(torch.tensor(-3.0)))  # 0.0486; rho is drawn around -3
    assert model_utils.mean_posterior_sigma(vnet) == pytest.approx(expected, rel=0.05)
    assert np.isnan(model_utils.mean_posterior_sigma(nn.Linear(2, 2)))


# --------------------------------------------------------------------------- split reproducibility guard
def test_split_script_refuses_sklearn_1_8(monkeypatch):
    import sklearn
    monkeypatch.setattr(sklearn, "__version__", "1.8.0")
    spec = importlib.util.spec_from_file_location("split_guard", ROOT / "splits" / "make_isic_clinical_split.py")
    with pytest.raises(RuntimeError, match="1.8"):
        spec.loader.exec_module(importlib.util.module_from_spec(spec))


# --------------------------------------------------------------------------- NRC: abstract model
class _TinyNet(nn.Module):
    """2-class CNN whose prediction is fixed by the head bias (class 1 wins)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 2)
        with torch.no_grad():
            self.fc.bias.copy_(torch.tensor([0.0, 50.0]))

    def forward(self, x):
        return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))


def _mam(config=None, model=None):
    from medirai_abstract_model import MediraiAbstractModel
    cfg = {"image_size": 16, "device": "cpu"}
    cfg.update(config or {})
    return MediraiAbstractModel("tiny", model or _TinyNet(), cfg, mode="train")


def test_unlock_n_trailing_grads_unlocks_the_last_n():
    m = _mam(model=nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2)))
    for p in m.model.parameters():
        p.requires_grad = False
    m.unlock_n_trailing_grads(n=2)
    flags = [p.requires_grad for p in m.model.parameters()]
    assert flags == [False, False, False, False, True, True]   # was the inverse (P1-13)


def test_predict_uses_the_requested_device():
    m = _mam(config={"device": "cuda:7"})                       # a device that does not exist
    img = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    logits, probs = m.predict(img, "cpu")                       # tensor went to CONFIG['device'] (P1-19)
    assert probs.sum() == pytest.approx(1.0) and int(np.argmax(probs)) == 1


def test_gradcam_defaults_to_predicted_class(monkeypatch):
    import medirai_abstract_model as mam
    seen = {}

    class FakeCAM:
        def __init__(self, model, target_layers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __call__(self, input_tensor, targets):
            seen["category"] = targets[0].category
            return np.zeros((1, 16, 16), dtype=np.float32)

    monkeypatch.setattr(mam, "GradCAM", FakeCAM)
    m = _mam()
    img = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)  # ndarray, not a path
    m.generate_cam(img, [m.model.conv], save_name=None)
    assert seen["category"] == 1                                 # the model's prediction (P1-16)
    m.generate_cam(img, [m.model.conv], target_class=0, save_name=None)
    assert seen["category"] == 0                                 # explicit override still honoured
    matplotlib.pyplot.close("all")


def test_train_passes_configured_grad_acc(tmp_path, monkeypatch):
    import medirai_abstract_model as mam
    calls = {}

    class FakeTrainer:
        def __init__(self, *a, **k):
            pass

        def fit(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(mam, "Trainer", FakeTrainer)
    csv = tmp_path / "train.csv"
    pd.DataFrame({"image_path": ["a", "b"], "target": [0, 1]}).to_csv(csv, index=False)
    m = _mam(config={"grad_acc": 8, "learning_rate": 1e-3, "weight_decay": 0.0, "seed": 0})
    m.train(str(csv), scheduler=None)
    assert calls.get("gradient_accumulation") == 8               # silently 1 before (P1-22)


# --------------------------------------------------------------------------- P1-15: inference contract
def test_inference_engine_imports():
    import medirai_inference  # noqa: F401  (global_fusion did `from os import wait`, which is POSIX-only)


def test_inference_engine_returns_uniform_entries():
    from medirai_inference import MediraiInferenceEngine

    class Stub:
        def __init__(self, fn):
            self.predict = fn

    eng = object.__new__(MediraiInferenceEngine)
    eng.device = "cpu"
    eng.loaded_models = ["clip", "derm", "dnns", "dnn_fusion", "found_fusion", "global_fusion"]
    eng.clip = Stub(lambda img: torch.tensor([[0.0, 2.0]]))
    eng.derm = Stub(lambda img: np.array([[1.0, -1.0]]))
    eng.dnns = Stub(lambda img: ([np.array([0.2, 0.8])] * 4, [1, 1, 1, 1]))
    eng.dnn_fusion = Stub(lambda img, device: (np.array([0.0, 1.0]), np.array([0.27, 0.73])))
    eng.found_fusion = Stub(lambda img: torch.tensor([[3.0, 0.0]]))
    eng.global_fusion = Stub(lambda img: np.array([[0.0, 0.5]]))

    preds = eng.predict("unused.jpg")
    expected = {"clip": 1, "derm": 0, "densenet": 1, "efficientnet": 1, "inception": 1, "resnet": 1,
                "dnn_fusion": 1, "found_fusion": 0, "global_fusion": 1}
    assert set(preds) == set(expected)
    for name, entry in preds.items():
        assert isinstance(entry, dict) and {"probs", "pred"} <= set(entry), name   # bare labels before
        assert entry["pred"] == expected[name], name
        assert float(np.sum(entry["probs"])) == pytest.approx(1.0), name


# --------------------------------------------------------------------------- P1-20: BiomedCLIP features
def test_biomedclip_pred_is_top_class_and_image_is_rgb(tmp_path):
    import biomed_clip
    path = tmp_path / "gray.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(path)

    clip = object.__new__(biomed_clip.MediraiBiomedClip)
    clip.device = "cpu"
    clip.default_prompt = ["a", "b", "c", "d"]

    def preprocess(image):
        assert image.mode == "RGB"                                # no .convert('RGB') before
        return torch.zeros(3, 4, 4)

    clip.clip_preprocess = preprocess
    clip.clip_tokenizer = lambda prompt, context_length: torch.zeros(len(prompt), 5, dtype=torch.long)
    text = torch.tensor([[0.3, 0.0], [0.1, 0.0], [0.9, 0.0], [0.6, 0.0]])  # class ranking 2 > 3 > 0 > 1
    clip.clip_model = lambda img, txt: (torch.tensor([[1.0, 0.0]]), text, torch.tensor(1.0))

    preds, _ = clip.get_img_features(str(path))
    assert preds == 2   # argmax(sorted_indices) = argmax([2, 3, 0, 1]) = 1 before
