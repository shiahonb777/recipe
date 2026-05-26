"""`recipe` — command-line entry point.

Subcommands:
  recipe init                Create a .recipe/ directory in cwd.
  recipe commit              Create a recipe from --base / --adapter inputs.
  recipe show                Pretty-print a recipe.
  recipe materialize         Apply a recipe to produce a merged checkpoint.
  recipe push                Push a recipe to a GitHub Release.
  recipe clone               Pull a recipe from a GitHub Release.

Design constraints:
  - No required dependencies beyond what `recipe` itself needs.
  - Friendly errors. The first time someone runs the wrong command we want
    them to know exactly what to fix.
  - Subcommands are flat; we don't nest beyond one level.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from recipe import (
    Recipe,
    load_recipe,
    save_recipe,
)
from recipe.recipe import (
    Adapter,
    BaseRef,
    TrainingMetadata,
    artifact_path,
    hash_file,
    store_artifact,
)


# ---------- helpers ----------


def _err(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)


def _find_repo(start: Path) -> Path:
    """Walk up from `start` to find a `.recipe` directory."""
    p = start.resolve()
    while True:
        candidate = p / ".recipe"
        if candidate.is_dir():
            return candidate
        if p.parent == p:
            raise FileNotFoundError(
                "not a recipe repo (or any parent up to filesystem root). "
                "run `recipe init` first."
            )
        p = p.parent


# ---------- subcommands ----------


def cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.path).resolve()
    repo_dir = target / ".recipe"
    if repo_dir.exists():
        _err(f"{repo_dir} already exists")
        return 1
    repo_dir.mkdir(parents=True)
    (repo_dir / "artifacts").mkdir()
    (repo_dir / "HEAD").write_text("draft\n")
    print(f"initialized empty recipe repo in {repo_dir}")
    return 0


def cmd_commit(args: argparse.Namespace) -> int:
    try:
        repo_dir = _find_repo(Path.cwd())
    except FileNotFoundError as e:
        _err(str(e))
        return 1
    if not args.base:
        _err("--base is required")
        return 2
    if not args.adapter and not args.allow_empty:
        _err("--adapter is required (or pass --allow-empty for a base-only recipe)")
        return 2

    base = BaseRef(ref=args.base, revision=args.revision)
    adapters: list[Adapter] = []

    if args.adapter:
        adapter_path = Path(args.adapter)
        if not adapter_path.exists():
            _err(f"adapter file not found: {adapter_path}")
            return 1
        h = store_artifact(repo_dir, adapter_path)
        adapter = Adapter(
            type="lora",
            artifact=h,
            target_modules=args.target_modules or [],
            rank=args.rank,
            alpha=args.alpha,
        )
        adapters.append(adapter)

    training = TrainingMetadata(
        method="lora" if adapters else "none",
        seed=args.seed,
        steps=args.steps,
        learning_rate=args.lr,
        dataset_hash=args.dataset_hash,
    )

    name = args.name or "draft"
    recipe = Recipe(name=name, base=base, adapters=adapters, training=training)
    out = save_recipe(recipe, repo_dir)
    print(f"recipe saved to {out}")
    if adapters:
        print(f"  base       : {base.ref}" + (f"@{base.revision}" if base.revision else ""))
        for a in adapters:
            size = artifact_path(repo_dir, a.artifact).stat().st_size
            print(f"  adapter    : {a.artifact[:24]}... ({size:,} bytes, "
                  f"rank={a.rank}, alpha={a.alpha})")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    if args.path:
        repo_dir = Path(args.path)
    else:
        try:
            repo_dir = _find_repo(Path.cwd())
        except FileNotFoundError as e:
            _err(str(e))
            return 1
    recipe = load_recipe(repo_dir)
    print(f"# {recipe.name}")
    print(f"format    {recipe.format_version}")
    print(f"base      {recipe.base.ref}"
          + (f"@{recipe.base.revision}" if recipe.base.revision else ""))
    if recipe.parent:
        print(f"parent    {recipe.parent}")
    if recipe.training.method != "unknown":
        t = recipe.training
        print("training:")
        print(f"  method  {t.method}")
        for k in ("seed", "steps", "learning_rate", "dataset_hash"):
            v = getattr(t, k)
            if v is not None:
                print(f"  {k:8s}{v}")
    if recipe.adapters:
        print("adapters:")
        for i, a in enumerate(recipe.adapters):
            ap = artifact_path(repo_dir, a.artifact)
            size = ap.stat().st_size if ap.exists() else None
            size_s = f"{size:,} B" if size is not None else "(missing)"
            print(f"  [{i}] {a.type} {a.artifact[:32]}...  {size_s}")
            if a.rank is not None:
                print(f"      rank={a.rank} alpha={a.alpha} targets={a.target_modules}")
    return 0


def cmd_materialize(args: argparse.Namespace) -> int:
    from recipe.materialize import materialize
    if args.repo:
        repo_dir = Path(args.repo)
    else:
        try:
            repo_dir = _find_repo(Path.cwd())
        except FileNotFoundError as e:
            _err(str(e))
            return 1
    recipe = load_recipe(repo_dir)
    out = Path(args.out)
    print(f"materializing {recipe.name} -> {out}")
    materialize(recipe, out, repo_dir=repo_dir)
    print(f"done. checkpoint at {out}")
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    """Push the current recipe to a GitHub Release.

    We bundle .recipe/recipe.toml + artifacts/ into a tar.gz and attach it
    as a release asset on the user's GitHub repo. This keeps the operator
    cost zero (GitHub hosts the file) and the trust story simple (the user
    sees the file under their own account).
    """
    import subprocess
    try:
        repo_dir = _find_repo(Path.cwd())
    except FileNotFoundError as e:
        _err(str(e))
        return 1
    recipe = load_recipe(repo_dir)
    target = args.target  # "user/repo" or "user/repo@tag"
    if "@" in target:
        repo, tag = target.split("@", 1)
    else:
        repo = target
        tag = recipe.name

    # Bundle.
    bundle = repo_dir.parent / f".recipe-bundle-{tag}.tar.gz"
    print(f"bundling -> {bundle}")
    import tarfile
    with tarfile.open(bundle, "w:gz") as tf:
        tf.add(repo_dir, arcname=".recipe")
    size = bundle.stat().st_size
    print(f"bundle size: {size:,} bytes")

    # Use `gh` to create / attach.
    print(f"creating release {tag} on {repo}")
    create = subprocess.run(
        ["gh", "release", "create", tag, str(bundle),
         "--repo", repo, "--title", tag,
         "--notes", f"recipe `{recipe.name}` (format {recipe.format_version})"],
        capture_output=True, text=True,
    )
    if create.returncode != 0:
        # Maybe the release already exists; try uploading the asset.
        if "already exists" in (create.stderr or ""):
            up = subprocess.run(
                ["gh", "release", "upload", tag, str(bundle),
                 "--repo", repo, "--clobber"],
                capture_output=True, text=True,
            )
            if up.returncode != 0:
                _err(f"gh release upload failed: {up.stderr.strip()}")
                return 1
        else:
            _err(f"gh release create failed: {create.stderr.strip()}")
            return 1
    print(f"pushed: https://github.com/{repo}/releases/tag/{tag}")
    return 0


def cmd_clone(args: argparse.Namespace) -> int:
    """Pull a recipe from a GitHub Release into a fresh directory."""
    import subprocess
    target = args.target
    if "@" in target:
        repo, tag = target.split("@", 1)
    else:
        repo = target
        tag = "latest"
    out_dir = Path(args.out or target.split("/")[-1].split("@")[0])
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = out_dir / "bundle.tar.gz"
    print(f"fetching {repo}@{tag} -> {bundle}")
    asset_pattern = "*.tar.gz"
    download = subprocess.run(
        ["gh", "release", "download", tag,
         "--repo", repo,
         "--pattern", asset_pattern,
         "--output", str(bundle),
         "--clobber"],
        capture_output=True, text=True,
    )
    if download.returncode != 0:
        _err(f"gh release download failed: {download.stderr.strip()}")
        return 1

    print("unpacking...")
    import tarfile
    with tarfile.open(bundle) as tf:
        tf.extractall(out_dir)
    bundle.unlink()
    recipe = load_recipe(out_dir / ".recipe")
    print(f"cloned recipe `{recipe.name}` into {out_dir}/")
    print(f"to materialize: cd {out_dir} && recipe materialize ./merged")
    return 0


# ---------- main ----------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="recipe",
        description="Ship model recipes, not weights.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="initialize a recipe repo here")
    p_init.add_argument("path", nargs="?", default=".")
    p_init.set_defaults(func=cmd_init)

    p_commit = sub.add_parser("commit", help="record a recipe in this repo")
    p_commit.add_argument("--name")
    p_commit.add_argument("--base", required=True,
                          help="HF Hub ref (e.g. meta-llama/Llama-3-8B)")
    p_commit.add_argument("--revision",
                          help="optional HF commit SHA to pin the base")
    p_commit.add_argument("--adapter",
                          help="path to a LoRA adapter (.safetensors or directory)")
    p_commit.add_argument("--target-modules", nargs="*",
                          help="modules the LoRA was applied to")
    p_commit.add_argument("--rank", type=int)
    p_commit.add_argument("--alpha", type=float)
    p_commit.add_argument("--seed", type=int)
    p_commit.add_argument("--steps", type=int)
    p_commit.add_argument("--lr", type=float)
    p_commit.add_argument("--dataset-hash")
    p_commit.add_argument("--allow-empty", action="store_true",
                          help="allow a recipe with no adapter")
    p_commit.set_defaults(func=cmd_commit)

    p_show = sub.add_parser("show", help="display the current recipe")
    p_show.add_argument("path", nargs="?")
    p_show.set_defaults(func=cmd_show)

    p_mat = sub.add_parser("materialize",
                           help="rebuild merged weights from the recipe")
    p_mat.add_argument("out", help="output directory")
    p_mat.add_argument("--repo",
                       help="recipe directory (default: search upward for .recipe)")
    p_mat.set_defaults(func=cmd_materialize)

    p_push = sub.add_parser("push",
                            help="push the recipe to a GitHub Release")
    p_push.add_argument("target", help="user/repo or user/repo@tag")
    p_push.set_defaults(func=cmd_push)

    p_clone = sub.add_parser("clone", help="pull a recipe from a GitHub Release")
    p_clone.add_argument("target", help="user/repo or user/repo@tag")
    p_clone.add_argument("out", nargs="?", help="output directory")
    p_clone.set_defaults(func=cmd_clone)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
