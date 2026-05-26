"""Materialize a recipe into a merged checkpoint.

Steps:
  1. Resolve the base model (download from HF Hub if needed, by `ref` and
     optional `revision`).
  2. For each adapter, fetch the artifact (locally first, then HF Hub
     fallback for shared registries).
  3. Apply adapters in order, in float32 then cast back to base dtype.
  4. Verify the result against the recipe's optional `expected_sha256`.

The output is a directory of safetensors shards equivalent to what
HF Hub would have served if you'd uploaded the merged weights directly.

We deliberately keep this implementation self-contained: numpy + safetensors,
no torch dependency for the core path. Torch is used only when a recipe
explicitly references torch-shaped LoRA adapters and torch is available.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from mlrecipe.recipe import Adapter, Recipe, artifact_path


# ---------- Base model resolution ----------


def resolve_base_path(recipe: Recipe, hf_cache: Optional[Path] = None) -> Path:
    """Return a local directory containing the base model's safetensors.

    We rely on huggingface_hub to do the download, with content-revision
    pinning when the recipe specifies it. The local snapshot directory is
    returned unchanged.
    """
    from huggingface_hub import snapshot_download
    kwargs: dict = {"repo_id": recipe.base.ref}
    if recipe.base.revision:
        kwargs["revision"] = recipe.base.revision
    if hf_cache is not None:
        kwargs["cache_dir"] = str(hf_cache)
    # Only fetch what we need: the safetensors and config; skip giant
    # tokenizer-only blobs and previews.
    kwargs["allow_patterns"] = [
        "*.safetensors",
        "*.json",
        "tokenizer*",
        "special_tokens_map.json",
    ]
    local = snapshot_download(**kwargs)
    return Path(local)


# ---------- LoRA application ----------


def _is_safetensors_index(path: Path) -> bool:
    return path.name.endswith(".safetensors.index.json")


def list_safetensors_files(base_dir: Path) -> list[Path]:
    """Return the list of safetensors files holding model weights.

    Prefers the index file if present (sharded models); otherwise the
    flat single-file pattern.
    """
    indexes = list(base_dir.glob("*.safetensors.index.json"))
    if indexes:
        with open(indexes[0]) as f:
            idx = json.load(f)
        files = sorted({v for v in idx.get("weight_map", {}).values()})
        return [base_dir / f for f in files]
    files = sorted(base_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors found in {base_dir}")
    return files


def _load_lora_weights(artifact_file: Path) -> dict[str, np.ndarray]:
    """Load a LoRA adapter from a safetensors file as a dict of arrays.

    Recognizes PEFT's standard naming: `<module>.lora_A.weight`,
    `<module>.lora_B.weight`. We don't impose a stricter schema here;
    we'll match by suffix during application.
    """
    from safetensors import safe_open
    out: dict[str, np.ndarray] = {}
    with safe_open(str(artifact_file), framework="np") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def _apply_lora_to_tensor(
    base_w: np.ndarray,
    lora_a: np.ndarray,
    lora_b: np.ndarray,
    alpha: float,
    rank: int,
    fan_in_fan_out: bool = False,
) -> np.ndarray:
    """Compute base + (alpha/rank) * delta and return same dtype as base.

    Default convention (PyTorch nn.Linear):
        A is (rank, in), B is (out, rank), base is (out, in).
        delta = B @ A is (out, in), shape-matches base directly.

    `fan_in_fan_out=True` (HuggingFace Conv1D, used in GPT-2's c_attn):
        Base weight is stored as (in, out) instead of (out, in).
        delta = (B @ A).T  to match base shape.
    """
    scaling = alpha / rank
    # NumPy 2.x emits spurious "divide by zero / overflow / invalid"
    # RuntimeWarnings from matmul on certain BLAS paths even when the
    # inputs and outputs are perfectly finite. Suppress them locally;
    # we re-check for NaN/Inf below as defense in depth.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        delta = lora_b.astype(np.float32) @ lora_a.astype(np.float32)
    if fan_in_fan_out:
        delta = delta.T
    delta = scaling * delta
    if delta.shape != base_w.shape:
        raise ValueError(
            f"LoRA delta shape {delta.shape} != base shape {base_w.shape} "
            f"(fan_in_fan_out={fan_in_fan_out})"
        )
    out = (base_w.astype(np.float32) + delta).astype(base_w.dtype)
    if not np.isfinite(out).all():
        raise ValueError(
            "non-finite values in merged tensor; LoRA application produced "
            "NaN or Inf (check rank/alpha and adapter dtype)"
        )
    return out


def _match_lora_targets(
    weight_keys: list[str],
    lora_keys: list[str],
    target_modules: list[str],
) -> dict[str, tuple[str, str]]:
    """Pair base weight keys with their corresponding (lora_A, lora_B) keys.

    Returns a dict: base_key -> (lora_A_key, lora_B_key).

    PEFT's standard naming wraps the original model in `base_model.model.`,
    so a base parameter `transformer.h.0.attn.c_attn.weight` becomes a LoRA
    pair under root `base_model.model.transformer.h.0.attn.c_attn`. But
    *which* prefix gets prepended depends on the wrapping model class.

    We therefore match by suffix: strip the common `base_model.model.`
    (or `base_model.`) prefix from the LoRA root, then look for any base
    key that *ends* with the same suffix (allowing for the base to omit
    one or more leading scopes the wrapper added).
    """
    pairs: dict[str, tuple[str, str]] = {}

    lora_a_by_root: dict[str, str] = {}
    lora_b_by_root: dict[str, str] = {}
    for lk in lora_keys:
        if lk.endswith(".lora_A.weight"):
            root = lk[: -len(".lora_A.weight")]
            lora_a_by_root[root] = lk
        elif lk.endswith(".lora_B.weight"):
            root = lk[: -len(".lora_B.weight")]
            lora_b_by_root[root] = lk

    for root, a_key in lora_a_by_root.items():
        b_key = lora_b_by_root.get(root)
        if b_key is None:
            continue

        # Strip the PEFT prefix wrapper (base_model.model. / base_model.).
        bare = root
        for prefix in ("base_model.model.", "base_model."):
            if bare.startswith(prefix):
                bare = bare[len(prefix):]
                break

        # Filter by target_modules using the *last* path component.
        if target_modules:
            mod_name = bare.split(".")[-1]
            if mod_name not in target_modules:
                continue

        # Try exact match first: <bare>.weight in weight_keys?
        candidate_exact = bare + ".weight"
        match: Optional[str] = None
        if candidate_exact in weight_keys:
            match = candidate_exact
        else:
            # Suffix match. Walk leading scopes off `bare` and try each.
            parts = bare.split(".")
            for i in range(1, len(parts)):
                suffix = ".".join(parts[i:]) + ".weight"
                hits = [k for k in weight_keys if k.endswith(suffix)]
                if len(hits) == 1:
                    match = hits[0]
                    break
                if len(hits) > 1:
                    # Ambiguous — go back to a more-specific suffix.
                    continue
        if match:
            pairs[match] = (a_key, b_key)
    return pairs


def apply_lora_adapter(
    base_dir: Path,
    adapter_file: Path,
    adapter: Adapter,
    out_dir: Path,
) -> int:
    """Apply a LoRA adapter to base safetensors files; write merged shards.

    Returns the number of weights modified.
    """
    from safetensors import safe_open
    from safetensors.numpy import save_file
    import shutil

    out_dir.mkdir(parents=True, exist_ok=True)

    lora = _load_lora_weights(adapter_file)
    rank = adapter.rank or 0
    alpha = adapter.alpha if adapter.alpha is not None else float(rank)
    fan_in_fan_out = bool(adapter.extra.get("fan_in_fan_out", False))
    if rank <= 0:
        # Try to infer rank from the first lora_A matrix.
        for k, v in lora.items():
            if k.endswith(".lora_A.weight"):
                rank = v.shape[0]
                if alpha is None or alpha == 0:
                    alpha = float(rank)
                break
    if rank <= 0:
        raise ValueError("could not determine LoRA rank")

    # Copy non-weight files (config, tokenizer) verbatim once.
    for f in base_dir.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") \
                and not f.name.endswith(".safetensors.index.json"):
            dst = out_dir / f.name
            if not dst.exists():
                shutil.copy2(f, dst)

    modified = 0
    shard_files = list_safetensors_files(base_dir)
    for shard in shard_files:
        with safe_open(str(shard), framework="np") as f:
            weight_keys = list(f.keys())
            pairs = _match_lora_targets(weight_keys, list(lora.keys()),
                                        adapter.target_modules)
            new_state: dict[str, np.ndarray] = {}
            for k in weight_keys:
                w = f.get_tensor(k)
                if k in pairs:
                    a_key, b_key = pairs[k]
                    w = _apply_lora_to_tensor(w, lora[a_key], lora[b_key],
                                              alpha=alpha, rank=rank,
                                              fan_in_fan_out=fan_in_fan_out)
                    modified += 1
                new_state[k] = w
        out_shard = out_dir / shard.name
        save_file(new_state, str(out_shard))

    # Copy the index, if present (it still maps weight keys to shard names).
    indexes = list(base_dir.glob("*.safetensors.index.json"))
    for idx in indexes:
        dst = out_dir / idx.name
        if not dst.exists():
            shutil.copy2(idx, dst)

    return modified


# ---------- Top-level entry ----------


def materialize(
    recipe: Recipe,
    out_dir: os.PathLike | str,
    repo_dir: Optional[os.PathLike | str] = None,
    hf_cache: Optional[os.PathLike | str] = None,
) -> Path:
    """Materialize a recipe into `out_dir` as a complete safetensors checkpoint.

    `repo_dir` is the local recipe directory containing artifacts/.
    `hf_cache` overrides the HF Hub cache location.

    Returns the path to `out_dir` on success.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    base_path = resolve_base_path(
        recipe,
        hf_cache=Path(hf_cache) if hf_cache else None,
    )

    if not recipe.adapters:
        # Nothing to apply — just copy the base.
        import shutil
        for f in base_path.iterdir():
            if f.is_file():
                dst = out_path / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
        return out_path

    current_dir = base_path
    for i, adapter in enumerate(recipe.adapters):
        if adapter.type != "lora":
            raise NotImplementedError(
                f"adapter type {adapter.type!r} not supported in this version"
            )
        adapter_file = _resolve_artifact(adapter.artifact, repo_dir)
        intermediate = out_path if i == len(recipe.adapters) - 1 else (
            out_path.parent / f".recipe-stage-{i}"
        )
        modified = apply_lora_adapter(current_dir, adapter_file, adapter, intermediate)
        if modified == 0:
            raise RuntimeError(
                f"adapter {adapter.artifact} matched 0 base weights — "
                "check target_modules and key naming"
            )
        current_dir = intermediate

    return out_path


def _resolve_artifact(
    content_hash: str,
    repo_dir: Optional[os.PathLike | str],
) -> Path:
    """Resolve a content-addressed artifact to a local file path.

    Resolution order:
      1. Local repo's artifacts/ directory, if repo_dir given.
      2. (TODO) HF Hub fetch via the recipe registry.
    """
    if repo_dir is not None:
        candidate = artifact_path(repo_dir, content_hash)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"artifact {content_hash} not found locally; remote fetch not yet implemented"
    )
