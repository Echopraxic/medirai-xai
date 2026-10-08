"""
Shared test setup.

The P0 regression tests only need torch/numpy/pandas/sklearn. Heavy optional dependencies that are
not installed on the machine running the tests (e.g. torchvision, cv2, monai on a laptop) are
replaced by permissive stub modules *only if missing*, so the same tests run unmodified on the GPU VM
against the real libraries.
"""
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
NRC = ROOT / "poc-nrc-main"
UNC = ROOT / "uncertaintyNet-main"

OPTIONAL = [
    "torchvision", "torchvision.transforms", "torchvision.transforms.v2", "torchvision.transforms.functional",
    "torchvision.models", "torchvision.models.feature_extraction", "torchvision.datasets",
    "cv2", "albumentations", "albumentations.pytorch", "albumentations.core",
    "albumentations.core.transforms_interface",
    "monai", "monai.transforms", "monai.data", "lightning", "lightning.fabric", "h5py",
    "open_clip", "open_clip.factory", "huggingface_hub", "tensorflow", "tqdm.gui",
    "pytorch_grad_cam", "pytorch_grad_cam.utils", "pytorch_grad_cam.utils.model_targets",
    "pytorch_grad_cam.utils.image", "shap", "lime", "skimage", "skimage.segmentation",
]
STUBBED = set()


def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


for name in OPTIONAL:
    top = name.split(".")[0]
    if name not in sys.modules and (top in STUBBED or not _installed(top)):
        stub = MagicMock(name=name)
        stub.__spec__ = importlib.machinery.ModuleSpec(name, None)  # find_spec() on a stub must not crash
        sys.modules[name] = stub
        STUBBED.add(top)

# Recent matplotlib asks sys.modules["tensorflow"].is_tensor(x) for everything it plots; a MagicMock answers
# truthy, so every number is "converted" to an array forever (RecursionError in plt.subplots).
if "tensorflow" in STUBBED:
    sys.modules["tensorflow"].is_tensor = lambda x: False

for path in (NRC, UNC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
