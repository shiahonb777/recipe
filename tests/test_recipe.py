"""Recipe data structure + I/O tests. Pure-local, no network."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import pytest

from mlrecipe import load_recipe, save_recipe
from mlrecipe.recipe import (
    Adapter,
    BaseRef,
    Recipe,
    TrainingMetadata,
    artifact_path,
    hash_bytes,
    hash_file,
    store_artifact,
)


def test_recipe_round_trip(tmp_path: Path):
    r = Recipe(
        name="test",
        base=BaseRef(ref="meta-llama/Llama-3-8B", revision="0123abc"),
        adapters=[
            Adapter(
                type="lora",
                artifact="sha256:" + "a" * 64,
                target_modules=["q_proj", "v_proj"],
                rank=16,
                alpha=32.0,
            )
        ],
        training=TrainingMetadata(
            method="lora", seed=42, steps=1000, learning_rate=2e-4,
        ),
    )
    save_recipe(r, tmp_path)
    assert (tmp_path / "recipe.toml").is_file()

    r2 = load_recipe(tmp_path)
    assert r2.name == r.name
    assert r2.base.ref == r.base.ref
    assert r2.base.revision == r.base.revision
    assert len(r2.adapters) == 1
    a = r2.adapters[0]
    assert a.type == "lora"
    assert a.rank == 16
    assert a.alpha == 32.0
    assert a.target_modules == ["q_proj", "v_proj"]
    assert r2.training.seed == 42
    assert r2.training.steps == 1000


def test_recipe_minimal(tmp_path: Path):
    """A recipe with only a base reference, no adapters, no training metadata."""
    r = Recipe(name="bare", base=BaseRef(ref="some/model"))
    save_recipe(r, tmp_path)
    r2 = load_recipe(tmp_path)
    assert r2.name == "bare"
    assert r2.base.ref == "some/model"
    assert r2.base.revision is None
    assert r2.adapters == []


def test_recipe_with_parent(tmp_path: Path):
    r = Recipe(
        name="child",
        base=BaseRef(ref="meta-llama/Llama-3-8B"),
        parent="alice/llama-medical@v1",
    )
    save_recipe(r, tmp_path)
    r2 = load_recipe(tmp_path)
    assert r2.parent == "alice/llama-medical@v1"


def test_hash_bytes_is_stable():
    h1 = hash_bytes(b"hello world")
    h2 = hash_bytes(b"hello world")
    h3 = hash_bytes(b"hello worlz")
    assert h1 == h2
    assert h1 != h3
    expected = "sha256:" + hashlib.sha256(b"hello world").hexdigest()
    assert h1 == expected


def test_hash_file_matches_hash_bytes(tmp_path: Path):
    data = b"some bytes" * 100
    f = tmp_path / "a.bin"
    f.write_bytes(data)
    assert hash_file(f) == hash_bytes(data)


def test_artifact_path_validates(tmp_path: Path):
    repo = tmp_path / ".recipe"
    repo.mkdir()
    h = "sha256:" + "f" * 64
    p = artifact_path(repo, h)
    assert p.parent.name == "ff"
    assert p.name == "f" * 64

    with pytest.raises(ValueError):
        artifact_path(repo, "md5:abc")
    with pytest.raises(ValueError):
        artifact_path(repo, "sha256:tooshort")
    with pytest.raises(ValueError):
        artifact_path(repo, "no-prefix")


def test_store_artifact_dedups(tmp_path: Path):
    repo = tmp_path / ".recipe"
    repo.mkdir()
    src = tmp_path / "a.bin"
    src.write_bytes(b"some payload" * 1024)

    h1 = store_artifact(repo, src)
    p1 = artifact_path(repo, h1)
    assert p1.exists()
    mtime1 = p1.stat().st_mtime

    # Second store of same content: should be a no-op (no rewrite).
    h2 = store_artifact(repo, src)
    assert h1 == h2
    assert p1.stat().st_mtime == mtime1


def test_unknown_keys_round_trip(tmp_path: Path):
    """Future format extensions in [training] or [[adapters]] should survive
    a load+save cycle in `extra`."""
    r = Recipe(
        name="t",
        base=BaseRef(ref="m"),
        training=TrainingMetadata(method="lora", extra={"future_key": "x"}),
        adapters=[Adapter(type="lora", artifact="sha256:" + "1" * 64,
                          extra={"future_param": 7})],
    )
    save_recipe(r, tmp_path)
    r2 = load_recipe(tmp_path)
    assert r2.training.extra.get("future_key") == "x"
    assert r2.adapters[0].extra.get("future_param") == 7
