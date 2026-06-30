from __future__ import annotations

import pytest


def test_import_and_routes():
    import psl_flow

    assert tuple(psl_flow.ALLOWED_ROUTES) == ("psl_flow", "klvae_sit")


def test_terb_fake_forward(monkeypatch):
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")
    import psl_flow.models.terb.terb as terb_module

    class DummyBackbone(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.out_channels = int(kwargs["out_channels"])

        def forward(self, x):
            logits = torch.zeros(x.shape[0], self.out_channels, x.shape[-2], x.shape[-1], device=x.device)
            return logits, logits, (logits,)

    monkeypatch.setattr(terb_module, "SMPWrapper", DummyBackbone)
    model = terb_module.TeRB()
    out = model(torch.rand(1, 1, 256, 256))
    assert {"e", "T_rad", "R_env", "B_edge", "B_edge_logits", "S_phys", "S_01"}.issubset(out)
    assert out["S_phys"].shape == (1, 1, 256, 256)
    assert chr(65) not in out


def test_psl_vae_factor_stack_and_decode():
    torch = pytest.importorskip("torch")
    from psl_flow.models.psl_vae.transforms import PSLFactorTransform

    transform = PSLFactorTransform()
    x = torch.rand(1, 1, 32, 32)
    teacher = {
        "T_rad": torch.rand(1, 1, 32, 32),
        "e": torch.rand(1, 1, 32, 32),
        "R_env": torch.rand(1, 1, 32, 32),
        "B_edge": torch.rand(1, 1, 32, 32),
    }
    targets = transform.stack_from_teacher(teacher, x)
    assert targets["factor_stack_01"].shape[1] == 5
    decoded = transform.decode_stack(targets["factor_stack_tanh"])
    assert decoded["factor_stack_01"].shape[1] == 5
    assert chr(65) not in decoded


def test_psl_vae_instantiates_small_config():
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from psl_flow.models.psl_vae import PSLVAE

    model = PSLVAE(
        {
            "in_channels": 5,
            "out_channels": 5,
            "block_out_channels": [32],
            "down_block_types": ["DownEncoderBlock2D"],
            "up_block_types": ["UpDecoderBlock2D"],
            "layers_per_block": 1,
            "latent_channels": 8,
            "norm_num_groups": 8,
            "sample_size": 32,
        }
    )
    assert model.latent_channels == 8
    assert model.transform.stack_channels == 5


def test_linear_transport_fake_loss():
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")
    from psl_flow.models.sit.transport import create_transport

    class ZeroModel(nn.Module):
        def forward(self, x, t, **kwargs):
            return torch.zeros_like(x)

    transport = create_transport(path_type="Linear", prediction="velocity", loss_weight=None)
    terms = transport.training_losses(ZeroModel(), torch.randn(2, 4, 8, 8), {"x_RGB": torch.randn(2, 4, 8, 8)})
    assert "loss" in terms
    assert terms["loss"].shape == (2,)
