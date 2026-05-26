"""CLI smoke tests. We invoke `recipe.cli.main(...)` directly to avoid
spawning a subprocess, which keeps tests fast and lets pytest see coverage.

Network / HF / gh-CLI dependent commands (push, clone, materialize) are
not exercised here — those need their own integration tests with mocks
or fixtures.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mlrecipe import load_recipe
from mlrecipe.cli import main


def _run(monkeypatch, tmp_path, argv):
    monkeypatch.chdir(tmp_path)
    return main(argv)


def test_init_creates_repo(monkeypatch, tmp_path: Path):
    rc = _run(monkeypatch, tmp_path, ["init"])
    assert rc == 0
    assert (tmp_path / ".recipe").is_dir()
    assert (tmp_path / ".recipe" / "artifacts").is_dir()
    assert (tmp_path / ".recipe" / "HEAD").is_file()


def test_init_refuses_overwrite(monkeypatch, tmp_path: Path):
    _run(monkeypatch, tmp_path, ["init"])
    rc = _run(monkeypatch, tmp_path, ["init"])
    assert rc != 0


def test_commit_with_adapter(monkeypatch, tmp_path: Path, capsys):
    _run(monkeypatch, tmp_path, ["init"])

    # Fake adapter file (just bytes; recipe doesn't validate format at commit time).
    adapter = tmp_path / "lora.safetensors"
    adapter.write_bytes(b"fake lora payload" * 1000)

    rc = _run(monkeypatch, tmp_path, [
        "commit",
        "--name", "my-finetune",
        "--base", "meta-llama/Llama-3-8B",
        "--revision", "abc123",
        "--adapter", str(adapter),
        "--target-modules", "q_proj", "v_proj",
        "--rank", "16",
        "--alpha", "32",
        "--seed", "42",
        "--steps", "1000",
        "--lr", "0.0002",
    ])
    assert rc == 0

    r = load_recipe(tmp_path / ".recipe")
    assert r.name == "my-finetune"
    assert r.base.ref == "meta-llama/Llama-3-8B"
    assert r.base.revision == "abc123"
    assert len(r.adapters) == 1
    a = r.adapters[0]
    assert a.type == "lora"
    assert a.rank == 16
    assert a.alpha == 32.0
    assert a.target_modules == ["q_proj", "v_proj"]
    assert r.training.seed == 42

    # The artifact must be stored under .recipe/artifacts/<sha-prefix>/
    assert a.artifact.startswith("sha256:")
    sha = a.artifact.split(":", 1)[1]
    assert (tmp_path / ".recipe" / "artifacts" / sha[:2] / sha).is_file()


def test_commit_requires_adapter(monkeypatch, tmp_path: Path):
    _run(monkeypatch, tmp_path, ["init"])
    rc = _run(monkeypatch, tmp_path, [
        "commit", "--base", "x/y",
    ])
    assert rc == 2  # missing required arg


def test_commit_allow_empty(monkeypatch, tmp_path: Path):
    _run(monkeypatch, tmp_path, ["init"])
    rc = _run(monkeypatch, tmp_path, [
        "commit", "--base", "x/y", "--allow-empty",
    ])
    assert rc == 0
    r = load_recipe(tmp_path / ".recipe")
    assert r.adapters == []


def test_show_after_commit(monkeypatch, tmp_path: Path, capsys):
    _run(monkeypatch, tmp_path, ["init"])
    adapter = tmp_path / "lora.safetensors"
    adapter.write_bytes(b"x" * 4096)
    _run(monkeypatch, tmp_path, [
        "commit", "--name", "t", "--base", "x/y",
        "--adapter", str(adapter),
        "--rank", "8", "--alpha", "16",
    ])
    capsys.readouterr()  # discard prior output
    rc = _run(monkeypatch, tmp_path, ["show"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "x/y" in out
    assert "rank=8" in out
    assert "alpha=16" in out


def test_commit_outside_repo_fails(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    rc = main([
        "commit", "--base", "x/y", "--allow-empty",
    ])
    assert rc != 0
