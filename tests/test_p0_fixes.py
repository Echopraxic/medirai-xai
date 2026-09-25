"""
Regression tests for the CODEBASE_TODO P0 fixes. CPU-only, synthetic data, no datasets required.
Run from the workspace root:  python -m pytest tests -q
"""
import importlib

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold

from conftest import ROOT


# --------------------------------------------------------------------------- P0-9: NRC imports
def test_nrc_data_loader_and_trainer_import():
    importlib.import_module("isic_data_loader")  # was a SyntaxError
    importlib.import_module("trainer")


# --------------------------------------------------------------------------- P0-7 / P1-18: data loader
@pytest.fixture
def lesion_image(tmp_path):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(40, 60, 3), dtype=np.uint8)  # non-square on purpose
    path = tmp_path / "melanoma_0001.jpg"
    Image.fromarray(arr).save(path, format="PNG")  # lossless, despite the name
    return path, arr


def test_no_blur_shortcut_for_kaggle_ids(lesion_image):
    from isic_data_loader import ISICDataset
    path, arr = lesion_image
    df = pd.DataFrame({"image_path": [str(path)], "target": [1], "image_id": ["melanoma_0001"]})
    img, _ = ISICDataset(df, (60, 40))[0]
    np.testing.assert_array_equal(np.asarray(img), arr)  # previously GaussianBlur(9) for 'melanoma' ids


def test_blur_is_opt_in_and_uniform(lesion_image):
    from isic_data_loader import ISICDataset
    path, arr = lesion_image
    df = pd.DataFrame({"image_path": [str(path)] * 2, "target": [0, 1], "image_id": ["ISIC_1", "melanoma_1"]})
    ds = ISICDataset(df, (60, 40), blur_radius=2)
    a, b = (np.asarray(ds[i][0]) for i in range(2))
    np.testing.assert_array_equal(a, b)            # same treatment regardless of source
    assert not np.array_equal(a, arr)              # and it really blurs when asked


def test_transpose_mod_swaps_axes(lesion_image):
    from isic_data_loader import ISICDatasetWMod
    path, arr = lesion_image
    df = pd.DataFrame({"image_path": [str(path)], "target": [0], "image_id": ["x"], "mod": ["transpose"]})
    ds = ISICDatasetWMod(df, (60, 40))
    out = np.asarray(ds._apply_mod(Image.fromarray(arr), "transpose", apply_noise=False))
    np.testing.assert_array_equal(out, arr.transpose(1, 0, 2))  # was a no-op


# --------------------------------------------------------------------------- P0-4 / P0-5: trainer folds
def _mixup_df(n_per_class=12, n_rows=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for target in (0, 1):
        ids = [f"img{target}_{i}" for i in range(n_per_class)]
        for _ in range(n_rows // 2):
            a, b = rng.choice(ids, 2, replace=False)
            rows.append({"image_id_1": a, "image_id_2": b, "image_path_1": f"{a}.jpg", "image_path_2": f"{b}.jpg",
                         "lambda": 0.5, "target": target})
    return pd.DataFrame(rows)


def test_source_images_and_fold_rows_do_not_leak():
    from trainer import rows_without_images, source_images
    df = _mixup_df()
    images = source_images(df)
    assert images["image_id"].is_unique and len(images) == 24
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    for _, val_idx in skf.split(images, images["target"]):
        val_ids = set(images.iloc[val_idx]["image_id"])
        train = rows_without_images(df, val_ids)
        used = set(train["image_id_1"]) | set(train["image_id_2"])
        assert not used & val_ids, "a validation image is also used in a training mixup pair"
        assert len(train) > 0


def test_repeated_image_rows_grouped():
    """'artificially_inflated' CSVs repeat the same image_id with different 'mod' values."""
    from trainer import rows_without_images, source_images
    df = pd.DataFrame({"image_id": ["a", "a", "b", "b", "c"], "image_path": list("aabbc"),
                       "target": [0, 0, 1, 1, 0], "mod": ["None", "rot1", "None", "rot1", "None"]})
    assert list(source_images(df)["image_id"]) == ["a", "b", "c"]
    assert set(rows_without_images(df, {"a"})["image_id"]) == {"b", "c"}


def test_each_fold_restarts_from_initial_weights(tmp_path, monkeypatch):
    from trainer import Trainer
    model = torch.nn.Linear(4, 2)
    init = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    df = pd.DataFrame({"image_id": [f"i{i}" for i in range(12)], "image_path": ["p"] * 12,
                       "target": [0, 1] * 6})
    config = {"n_fold": 3, "seed": 0, "image_size": 8, "batch_size": 4, "num_epochs": 2,
              "freeze_name_template": str(tmp_path / "m_XYYX.pkl")}
    trainer = Trainer("toy", model, config, None, optimizer, df, scheduler=None, device="cpu")

    starts = []

    def fake_train_step(loader, gradient_accumulation):
        starts.append({k: v.clone() for k, v in model.state_dict().items()})
        with torch.no_grad():
            model.weight += 1.0  # "training" changes the weights
        return 0.0, torch.tensor(0.5)

    monkeypatch.setattr(trainer, "train_step", fake_train_step)
    monkeypatch.setattr(trainer, "eval_step", lambda loader: (torch.tensor(0.1), torch.tensor(0.6)))
    monkeypatch.setattr(trainer, "save_training_log", lambda: None)
    trainer.fit()

    fold_starts = starts[::config["num_epochs"]]  # first epoch of each fold
    assert len(fold_starts) == 3
    for s in fold_starts:
        for k in init:
            torch.testing.assert_close(s[k], init[k])  # previously fold k continued from fold k-1
    assert (tmp_path / "m_1.pkl").exists() and (tmp_path / "m_best.pkl").exists()


# --------------------------------------------------------------------------- P0-14 / P0-16: predictor heads
def test_predictor_logits_can_be_negative():
    from predictor_network import PredictorNet
    torch.manual_seed(0)
    net = PredictorNet(8, 16).eval()
    with torch.no_grad():
        out = net(torch.randn(256, 8))
    assert (out < 0).any(), "logits are still clamped by an output activation"


def test_predictor_trainer_learns_and_saves_best(tmp_path):
    from predictor_network import PredictorNet, PredictorNetTrainer
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 200)                               # class-sorted, like the real CSVs
    X = (rng.normal(size=(400, 8)) + y[:, None] * 2.0).astype(np.float32)
    groups = np.repeat(np.arange(200), 2)                    # two "augmented copies" per image
    torch.manual_seed(0)
    net = PredictorNet(8, 16)
    PredictorNetTrainer(net, X, y, str(tmp_path / "head"), num_epochs=5, lr=1e-2, groups=groups).train()
    assert (tmp_path / "head_best.plk").exists()
    assert not net.training
    with torch.no_grad():
        acc = (net(torch.from_numpy(X)).argmax(1).numpy() == y).mean()
    assert acc > 0.9


def test_predictor_group_split_keeps_copies_together():
    from predictor_network import PredictorNet, PredictorNetTrainer
    groups = np.repeat(np.arange(10), 4)                      # 4 augmented copies per image
    X = np.repeat(groups[:, None], 4, axis=1).astype(np.float32)  # features encode the group id
    y = np.repeat([0, 1], 20)
    t = PredictorNetTrainer(PredictorNet(4, 4), X, y, "unused", val_split=0.3, groups=groups)
    tr_loader, va_loader = t._init_data()
    tr_groups = set(tr_loader.dataset.tensors[0][:, 0].int().tolist())
    va_groups = set(va_loader.dataset.tensors[0][:, 0].int().tolist())
    assert tr_groups and va_groups and not tr_groups & va_groups
    assert len(tr_loader.dataset) + len(va_loader.dataset) == 40
    assert tr_loader.batch_size == 64


# --------------------------------------------------------------------------- P0-15: eval mode after load
def test_foundation_heads_load_in_eval_mode(tmp_path):
    from predictor_network import PredictorNet
    import biomed_clip
    path = tmp_path / "head.plk"
    torch.save(PredictorNet(4, 4).state_dict(), path)

    clip = object.__new__(biomed_clip.MediraiBiomedClip)  # skip loading the real CLIP backbone
    clip.pred_model = PredictorNet(4, 4)
    clip.load_pred_net(str(path))
    assert not clip.pred_model.training  # Dropout(0.5) was active at inference


def test_derm_head_loads_in_eval_mode(tmp_path, monkeypatch):
    from predictor_network import PredictorNet
    import huggingface_hub
    # huggingface_hub >= 0.26 removed from_pretrained_keras (see CODEBASE_TODO P2-1); not needed for this test
    monkeypatch.setattr(huggingface_hub, "from_pretrained_keras", lambda *a, **k: None, raising=False)
    import derm_foundation
    path = tmp_path / "head.plk"
    torch.save(PredictorNet(4, 4).state_dict(), path)
    derm = object.__new__(derm_foundation.MediraiDermFoundation)
    derm.pred_model = PredictorNet(4, 4)
    derm._load_pred_net(str(path))
    assert not derm.pred_model.training


# --------------------------------------------------------------------------- P0-8: OOD leakage
def test_ood_and_biomed_csvs_exclude_test_images():
    csvs = ROOT / "poc-nrc-main" / "train_test_csvs"
    test_ids = set(pd.read_csv(csvs / "test_EQ.csv")["image_id"]) | set(pd.read_csv(csvs / "test_EQ_new_neg.csv")["image_id"])
    for name in ("ood_training.csv", "biomed_50k.csv", "biomed_50k_percentile_bin.csv"):
        assert not set(pd.read_csv(csvs / name)["image_id"]) & test_ids, name


# --------------------------------------------------------------------------- uncertaintyNet P0-11 / P0-12 / P0-13
def _unc_utils():
    import utils  # uncertaintyNet-main/utils.py
    assert hasattr(utils, "expected_calibration_error")
    return utils


def test_ece_counts_saturated_predictions():
    utils = _unc_utils()
    # all predictions say p=1.0 but only half are positive -> ECE must be 0.5 (was 0: p=1.0 fell in no bin)
    target = np.array([1, 0, 1, 0])
    assert utils.expected_calibration_error(target, np.ones(4)) == pytest.approx(0.5)
    probs = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    assert utils.top_label_calibration_error(probs, np.array([1, 0, 0, 1])) == pytest.approx(0.5)


def test_load_checkpoint_raises_on_mismatch(tmp_path):
    utils = _unc_utils()
    torch.save({"network_state_dict": torch.nn.Linear(3, 2).state_dict()}, tmp_path / "ckpt.pth.tar")
    with pytest.raises(RuntimeError, match="does not match"):
        utils.load_checkpoint(torch.nn.Linear(5, 2), str(tmp_path / "ckpt.pth.tar"))


def test_load_checkpoint_loads_matching_weights(tmp_path):
    utils = _unc_utils()
    src = torch.nn.Linear(3, 2)
    torch.save({"network_state_dict": src.state_dict()}, tmp_path / "ckpt.pth.tar")
    net, _ = utils.load_checkpoint(torch.nn.Linear(3, 2), str(tmp_path / "ckpt.pth.tar"))
    torch.testing.assert_close(net.weight, src.weight)


def test_resnet18_config_builds_resnet18(monkeypatch):
    from models import model
    built = {}
    for name in ("ResNet18Hidden", "ResNet50Hidden", "WideResNet50Hidden"):
        monkeypatch.setattr(model, name, lambda num_classes, _n=name: built.setdefault("cls", _n))
    cfg = {"architecture": "resnet18p_pretrained", "output_size": 2, "network_type": "deterministic"}
    model.get_model(cfg)
    assert built["cls"] == "ResNet18Hidden"  # was WideResNet50Hidden
    with pytest.raises(NotImplementedError):
        model.get_model({"architecture": "resunet", "output_size": 2, "network_type": "deterministic"})


# --------------------------------------------------------------------------- P0-1/P0-2 wiring in uncertaintyNet
def test_uncertaintynet_reads_persisted_split(tmp_path):
    import dataset
    df = dataset.load_split()
    assert set(df["split"]) == {"train", "val", "test"} and df["label"].notna().all()
    with pytest.raises(FileNotFoundError, match="missing"):
        dataset.load_split(data_dir=str(tmp_path))  # empty image folder -> clear error, not a silent skip
