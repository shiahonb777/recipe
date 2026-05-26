"""Materialize test that does NOT touch HF Hub.

We construct a tiny "base model" (a few weight tensors written as
safetensors), a tiny LoRA adapter, and verify that
`apply_lora_adapter` produces base + (alpha/rank) * (B @ A) bit-exactly
within float32 rounding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlrecipe.recipe import Adapter
from mlrecipe.materialize import (
    _apply_lora_to_tensor,
    _match_lora_targets,
    apply_lora_adapter,
)


def test_apply_lora_to_tensor_matches_formula():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((32, 64)).astype(np.float32)
    rank, alpha = 4, 8.0
    A = rng.standard_normal((rank, 64)).astype(np.float32) * 0.01
    B = rng.standard_normal((32, rank)).astype(np.float32) * 0.01
    out = _apply_lora_to_tensor(base, A, B, alpha=alpha, rank=rank)
    expected = base + (alpha / rank) * (B @ A)
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-6)


def test_apply_lora_dtype_preserved():
    rng = np.random.default_rng(0)
    base = (rng.standard_normal((8, 16)) * 0.1).astype(np.float16)
    A = (rng.standard_normal((2, 16)) * 0.01).astype(np.float16)
    B = (rng.standard_normal((8, 2)) * 0.01).astype(np.float16)
    out = _apply_lora_to_tensor(base, A, B, alpha=2.0, rank=2)
    assert out.dtype == np.float16


def test_match_lora_targets_pairs_a_b():
    base_keys = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ]
    lora_keys = [
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight",
    ]
    pairs = _match_lora_targets(base_keys, lora_keys, ["q_proj", "v_proj"])
    assert "model.layers.0.self_attn.q_proj.weight" in pairs
    assert "model.layers.0.self_attn.v_proj.weight" in pairs
    assert "model.layers.0.mlp.gate_proj.weight" not in pairs


def test_match_lora_targets_filters_modules():
    base_keys = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    ]
    lora_keys = [
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight",
    ]
    pairs = _match_lora_targets(base_keys, lora_keys, ["q_proj"])
    assert len(pairs) == 1
    assert "model.layers.0.self_attn.q_proj.weight" in pairs


def test_apply_lora_adapter_end_to_end(tmp_path: Path):
    """Build a fake base safetensors + a fake LoRA adapter, apply, verify."""
    from safetensors.numpy import save_file

    base_dir = tmp_path / "base"
    base_dir.mkdir()
    rng = np.random.default_rng(7)
    q_w = rng.standard_normal((32, 64)).astype(np.float32)
    v_w = rng.standard_normal((32, 64)).astype(np.float32)
    other_w = rng.standard_normal((16, 16)).astype(np.float32)
    base_state = {
        "model.layers.0.self_attn.q_proj.weight": q_w,
        "model.layers.0.self_attn.v_proj.weight": v_w,
        "model.layers.0.mlp.gate_proj.weight": other_w,
    }
    save_file(base_state, str(base_dir / "model.safetensors"))
    # A non-weight file should be copied verbatim.
    (base_dir / "config.json").write_text('{"model_type": "fake"}')

    rank, alpha = 4, 8.0
    A_q = rng.standard_normal((rank, 64)).astype(np.float32) * 0.01
    B_q = rng.standard_normal((32, rank)).astype(np.float32) * 0.01
    A_v = rng.standard_normal((rank, 64)).astype(np.float32) * 0.01
    B_v = rng.standard_normal((32, rank)).astype(np.float32) * 0.01
    lora_state = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": A_q,
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": B_q,
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": A_v,
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": B_v,
    }
    adapter_file = tmp_path / "adapter.safetensors"
    save_file(lora_state, str(adapter_file))

    out_dir = tmp_path / "merged"
    adapter = Adapter(
        type="lora",
        artifact="sha256:" + "0" * 64,
        target_modules=["q_proj", "v_proj"],
        rank=rank,
        alpha=alpha,
    )
    n = apply_lora_adapter(base_dir, adapter_file, adapter, out_dir)
    assert n == 2  # q_proj + v_proj

    # Verify merged weights match the formula.
    from safetensors import safe_open
    with safe_open(str(out_dir / "model.safetensors"), framework="np") as f:
        merged_q = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
        merged_v = f.get_tensor("model.layers.0.self_attn.v_proj.weight")
        unchanged = f.get_tensor("model.layers.0.mlp.gate_proj.weight")
    expected_q = q_w + (alpha / rank) * (B_q @ A_q)
    expected_v = v_w + (alpha / rank) * (B_v @ A_v)
    np.testing.assert_allclose(merged_q, expected_q, rtol=0, atol=1e-5)
    np.testing.assert_allclose(merged_v, expected_v, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(unchanged, other_w)

    # Non-weight file copied.
    assert (out_dir / "config.json").is_file()


def test_apply_lora_adapter_no_match_raises_via_zero_count(tmp_path: Path):
    """If LoRA targets don't match anything, apply returns 0 modifications.
    The high-level materialize() converts that into an exception, but the
    low-level helper just reports the count."""
    from safetensors.numpy import save_file

    base_dir = tmp_path / "base"
    base_dir.mkdir()
    save_file(
        {"model.foo.weight": np.zeros((4, 4), dtype=np.float32)},
        str(base_dir / "model.safetensors"),
    )

    A = np.zeros((2, 8), dtype=np.float32)
    B = np.zeros((8, 2), dtype=np.float32)
    lora = {
        "base_model.model.unrelated.lora_A.weight": A,
        "base_model.model.unrelated.lora_B.weight": B,
    }
    adapter_file = tmp_path / "lora.safetensors"
    save_file(lora, str(adapter_file))

    out = tmp_path / "out"
    adapter = Adapter(
        type="lora",
        artifact="sha256:" + "0" * 64,
        target_modules=["unrelated"],
        rank=2,
        alpha=2.0,
    )
    n = apply_lora_adapter(base_dir, adapter_file, adapter, out)
    assert n == 0
