"""The Recipe data structure and on-disk format.

A recipe is a small TOML file plus zero or more "artifact" files referenced
by content hash. The TOML is the source of truth; artifacts (LoRA weights,
training data hashes, tokenizer extras) are content-addressable blobs.

On-disk layout in a `.recipe/` directory:

    .recipe/
      recipe.toml           # the recipe (this file)
      artifacts/
        <sha256>.bin        # content-addressed binary blobs (LoRA, etc.)

Format (recipe.toml):

    [recipe]
    version = "0.1"
    name = "my-medical-finetune"

    [base]
    # The base model the recipe is derived from. Two ways to identify:
    # 1. HF Hub reference (preferred): "meta-llama/Llama-3-8B@<commit-sha>"
    # 2. Local path with content hash
    ref = "meta-llama/Llama-3-8B"
    revision = "0d0b1d6f10dc6c9e0f9c5af0e2c9fa3df8e90f7a"  # HF commit SHA
    sha256 = "..."  # SHA256 of the safetensors file(s); optional but recommended

    [training]
    # Metadata about how this fine-tune was produced. Pure provenance;
    # the materializer doesn't need it but auditors do.
    method = "lora"
    seed = 42
    steps = 10000
    learning_rate = 0.0002
    dataset_hash = "sha256:abc..."

    [[adapters]]
    # Each adapter is a content-addressed blob applied on top of the base.
    # Adapters are applied in the order they appear.
    type = "lora"
    artifact = "sha256:9c2bc..."
    target_modules = ["q_proj", "v_proj"]
    rank = 16
    alpha = 32

    [parents]
    # Optional: this recipe is derived from another recipe. The parent's
    # materialized weights are the "base" for this recipe's adapters.
    # This is what makes lineage / fork-and-merge possible.
    # parent = "user/repo@v1.2"
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]
import tomli_w


RECIPE_FORMAT_VERSION = "0.1"


# ---------- Data classes ----------


@dataclass
class BaseRef:
    """Reference to the base model this recipe is derived from."""
    ref: str
    revision: Optional[str] = None
    sha256: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"ref": self.ref}
        if self.revision is not None:
            d["revision"] = self.revision
        if self.sha256 is not None:
            d["sha256"] = self.sha256
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseRef":
        return cls(
            ref=d["ref"],
            revision=d.get("revision"),
            sha256=d.get("sha256"),
        )


@dataclass
class TrainingMetadata:
    """Provenance only — not consumed by the materializer."""
    method: str = "unknown"
    seed: Optional[int] = None
    steps: Optional[int] = None
    learning_rate: Optional[float] = None
    dataset_hash: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {"method": self.method}
        for k in ("seed", "steps", "learning_rate", "dataset_hash"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingMetadata":
        known = {"method", "seed", "steps", "learning_rate", "dataset_hash"}
        kwargs = {k: d.get(k) for k in known if k in d}
        kwargs.setdefault("method", "unknown")
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(**kwargs, extra=extra)


@dataclass
class Adapter:
    """A delta applied on top of the (possibly merged) base."""
    type: str            # "lora" today; future: "ia3", "sparse_diff", "full_diff"
    artifact: str        # "sha256:<hex>" reference to a blob
    target_modules: list[str] = field(default_factory=list)
    rank: Optional[int] = None
    alpha: Optional[float] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {"type": self.type, "artifact": self.artifact}
        if self.target_modules:
            d["target_modules"] = list(self.target_modules)
        if self.rank is not None:
            d["rank"] = self.rank
        if self.alpha is not None:
            d["alpha"] = self.alpha
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Adapter":
        known = {"type", "artifact", "target_modules", "rank", "alpha"}
        kwargs = {
            "type": d["type"],
            "artifact": d["artifact"],
            "target_modules": list(d.get("target_modules", [])),
            "rank": d.get("rank"),
            "alpha": d.get("alpha"),
        }
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(**kwargs, extra=extra)


@dataclass
class Recipe:
    """A complete fine-tune recipe."""
    name: str
    base: BaseRef
    adapters: list[Adapter] = field(default_factory=list)
    training: TrainingMetadata = field(default_factory=TrainingMetadata)
    parent: Optional[str] = None
    format_version: str = RECIPE_FORMAT_VERSION

    def to_dict(self) -> dict:
        d: dict = {
            "recipe": {
                "version": self.format_version,
                "name": self.name,
            },
            "base": self.base.to_dict(),
            "training": self.training.to_dict(),
        }
        if self.adapters:
            d["adapters"] = [a.to_dict() for a in self.adapters]
        if self.parent is not None:
            d["parents"] = {"parent": self.parent}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Recipe":
        recipe_meta = d.get("recipe", {})
        version = recipe_meta.get("version", RECIPE_FORMAT_VERSION)
        if version != RECIPE_FORMAT_VERSION:
            # We accept older versions for read; a warning would go here.
            pass
        return cls(
            name=recipe_meta.get("name", "unnamed"),
            base=BaseRef.from_dict(d["base"]),
            adapters=[Adapter.from_dict(a) for a in d.get("adapters", [])],
            training=TrainingMetadata.from_dict(d.get("training", {})),
            parent=d.get("parents", {}).get("parent"),
            format_version=version,
        )


# ---------- I/O ----------


def load_recipe(path: os.PathLike | str) -> Recipe:
    """Load a recipe from a directory (containing recipe.toml) or a TOML file."""
    p = Path(path)
    if p.is_dir():
        p = p / "recipe.toml"
    with open(p, "rb") as f:
        d = tomllib.load(f)
    return Recipe.from_dict(d)


def save_recipe(recipe: Recipe, path: os.PathLike | str) -> Path:
    """Write a recipe to a directory. Creates the directory and recipe.toml."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    out = p / "recipe.toml"
    with open(out, "wb") as f:
        tomli_w.dump(recipe.to_dict(), f)
    return out


# ---------- Content addressing helpers ----------


def hash_file(path: os.PathLike | str, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA256 of a file as 'sha256:<hex>'."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def hash_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def artifact_path(repo_dir: os.PathLike | str, content_hash: str) -> Path:
    """Compute the local path for a content-addressed artifact.

    The hash is given as 'sha256:<hex>'. We strip the prefix and use the
    hex string as the filename, sharded by the first two chars to keep
    directory sizes manageable.
    """
    if ":" not in content_hash:
        raise ValueError(f"expected 'sha256:<hex>', got {content_hash!r}")
    algo, hexdigest = content_hash.split(":", 1)
    if algo != "sha256":
        raise ValueError(f"unsupported hash algo: {algo}")
    if len(hexdigest) != 64 or not all(c in "0123456789abcdef" for c in hexdigest):
        raise ValueError(f"invalid sha256 hex: {hexdigest!r}")
    return Path(repo_dir) / "artifacts" / hexdigest[:2] / hexdigest


def store_artifact(repo_dir: os.PathLike | str, src: os.PathLike | str) -> str:
    """Copy a file into the artifacts/ store, return its content hash."""
    h = hash_file(src)
    dst = artifact_path(repo_dir, h)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        # Atomic move via temp + rename.
        import shutil
        tmp = dst.with_suffix(".tmp")
        shutil.copy2(src, tmp)
        tmp.replace(dst)
    return h
