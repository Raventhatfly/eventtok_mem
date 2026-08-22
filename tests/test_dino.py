"""Pin the DINOv2 path to the SigLIP pipeline it is being compared against.

If any of these drift, the encoder comparison silently becomes a comparison of
encoder *and* preprocessing, and the result means nothing. The reference is
``mme_vla_suite/shared/data_utils.pool_tokens_to_size`` and
``mem_buffer.add_buffer``.
"""

from __future__ import annotations

import numpy as np
import pytest

from eventtok.data import dino


def _pool_reference(tokens: np.ndarray, target: int) -> np.ndarray:
    """The SigLIP rule written out longhand: row-major grid, non-overlapping mean."""
    b, p, d = tokens.shape
    h = w = int(round(p**0.5))
    size = int(round((p / target) ** 0.5))
    grid = tokens.reshape(b, h, w, d)
    side = h // size
    return np.stack(
        [
            np.stack(
                [
                    grid[i, r * size : (r + 1) * size, c * size : (c + 1) * size].mean(
                        axis=(0, 1)
                    )
                    for r in range(side)
                    for c in range(side)
                ]
            )
            for i in range(b)
        ]
    )


@pytest.mark.parametrize("target", [4, 16, 64])
def test_pooling_matches_siglip_rule(target: int) -> None:
    import torch

    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 256, 5)).astype(np.float32)
    got = dino.pool_tokens(torch.from_numpy(x), target).numpy()
    assert got.shape == (3, target, 5)
    assert np.allclose(got, _pool_reference(x, target), atol=1e-5)


def test_pooling_is_identity_at_native_size() -> None:
    import torch

    x = torch.randn(2, 64, 4)
    assert torch.equal(dino.pool_tokens(x, 64), x)


def test_pooling_rejects_indivisible_target() -> None:
    import torch

    with pytest.raises(ValueError):
        dino.pool_tokens(torch.randn(1, 256, 4), 32)   # 256/32 = 8, not a square


def test_patch_grid_divides_every_scale() -> None:
    """224/14 = 16, so the 16x16 grid pools exactly to 8x8, 4x4 and 2x2.

    This is why the resolution is not a free parameter: at 518 the grid is 37x37
    and none of the scales divide it.
    """
    grid = dino.RESOLUTION // dino.PATCH
    assert grid == 16
    for scale, n in dino._N_TOKENS.items():
        size = grid // int(round(n**0.5))
        assert size * int(round(n**0.5)) == grid, scale


def test_cache_paths_separate_camera_and_model() -> None:
    """The wrist camera is a different variable from the encoder swap.

    Both must land in distinct directories, or a "DINO vs SigLIP" row could
    quietly be reading wrist features on one side.
    """
    a = dino.cache_path("SwingXtimes", "dinov2l", "2x2", "image", 1000)
    b = dino.cache_path("SwingXtimes", "dinov2l", "2x2", "wrist_image", 1000)
    c = dino.cache_path("SwingXtimes", "dinov2b", "2x2", "image", 1000)
    assert a != b != c and a != c
    assert a.name == b.name == "episode_1000.npy"
    assert "base" in a.parent.name and "wrist" in b.parent.name


def test_widths_are_declared_correctly() -> None:
    assert dino.MODELS["dinov2l"][1] == 1024
    assert dino.MODELS["dinov2b"][1] == 768
