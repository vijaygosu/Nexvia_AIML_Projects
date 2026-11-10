import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from medimg.explain.gradcam import GradCAM, blend, cam_to_rgba
from medimg.models.classifier import ChestClassifier


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return ChestClassifier(8, pretrained=False).eval()


def test_cam_shape_range_and_class_selection(model):
    x = torch.randn(2, 3, 224, 224)
    cams, logits, idx = GradCAM(model)(x)
    assert cams.shape == (2, 224, 224) and logits.shape == (2, 8)
    assert float(cams.min()) >= 0 and float(cams.max()) <= 1 + 1e-6
    assert torch.equal(idx, logits.argmax(1))
    cams2, _, idx2 = GradCAM(model)(x, class_idx=3)
    assert (idx2 == 3).all() and cams2.shape == cams.shape


def test_cam_works_with_frozen_params(model):
    for p in model.parameters():
        p.requires_grad_(False)
    cams, _, _ = GradCAM(model)(torch.randn(1, 3, 224, 224))
    assert torch.isfinite(cams).all()
    for p in model.parameters():
        p.requires_grad_(True)


def test_overlay_helpers():
    cam = np.random.rand(32, 32).astype(np.float32)
    gray = (np.random.rand(32, 32) * 255).astype(np.uint8)
    assert cam_to_rgba(cam).shape == (32, 32, 4)
    out = blend(gray, cam)
    assert out.shape == (32, 32, 3) and out.dtype == np.uint8
