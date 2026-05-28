# mlrecipe

A content-addressed format for LoRA fine-tunes. The recipe — a small TOML file
plus a hashed adapter — points at a base model on Hugging Face and rebuilds a
merged checkpoint that is bit-identical to `peft.merge_and_unload()`. Storage
and bandwidth drop by two to three orders of magnitude. Distribution is
GitHub releases. There is no new server, no new registry, no new account.

```bash
$ mlrecipe push alice/llama3-medical
bundling -> .recipe-bundle-v1.tar.gz
bundle size: 53,712 bytes        # not 14 GB
pushed: https://github.com/alice/llama3-medical/releases/tag/v1

$ mlrecipe clone alice/llama3-medical
fetching alice/llama3-medical@latest
unpacking…
cloned recipe `llama3-medical` into llama3-medical/

$ mlrecipe materialize ./merged
materializing llama3-medical -> merged
done. checkpoint at merged/
```

## The web explorer

[shiahonb777.github.io/mlrecipe](https://shiahonb777.github.io/mlrecipe/) reads any
recipe published as a GitHub release and renders its base, adapters, sizes, and
the four commands that rebuild it. Three panes:

- *Browse* a known recipe by `user/repo@tag`.
- *Search* the public web for `recipe.toml` files via the GitHub search API.
- *Publish* — the four commands that turn a PEFT adapter directory into a
  recipe and push it as a GitHub release. The adapter never leaves your
  machine before that `git push`.

The page is a static site on GitHub Pages. The data is whatever you publish on
your own GitHub account. There is no central index; the explorer queries the
GitHub API on every visit.

## Why a recipe is small

A LoRA fine-tune isn't a 14 GB blob. It's a 50 MB delta on top of a base model
the world already has. Today's tools — Hugging Face Hub, `git-lfs` — store and
transfer the merged result, paying the full cost every time.

`mlrecipe` records what was done: `base + adapter + training metadata`. The
receiver runs the recipe locally and gets the same weights back.

| Format                        | LoRA fine-tune on disk |
|-------------------------------|-----------------------:|
| Hugging Face Hub (merged)     | 14,000 MB              |
| `git-lfs` chunked dedup       | ~2,000 MB              |
| `mlrecipe` (LoRA-aware)       | ~50 MB                 |

## Worked examples

[`examples/gpt2_alpaca/`](examples/gpt2_alpaca/) downloads a real PEFT LoRA
(`monsterapi/gpt2_alpaca-lora`), packages it as a recipe (1.1 MB instead of
500 MB), materializes it, and verifies the result against PEFT's official
`merge_and_unload` — 148 of 148 tensors identical, max element-wise difference
zero.

```bash
bash examples/gpt2_alpaca/run.sh
```

The published copy is at
[`shiahonb777/gpt2-alpaca-recipe@v1`](https://github.com/shiahonb777/gpt2-alpaca-recipe/releases/tag/v1).
Cloning and materializing it from a fresh machine:

```bash
mlrecipe clone shiahonb777/gpt2-alpaca-recipe@v1
cd gpt2-alpaca-recipe
mlrecipe materialize ./merged
```

A larger example, [`examples/qwen_oasst/`](examples/qwen_oasst/), runs the
same path against Qwen 2.5 1.5B with a seven-module rank-16 LoRA. The recipe
is 74 MB; the merged checkpoint is roughly 3 GB. Published as
[`shiahonb777/qwen2.5-1.5b-oasst-recipe@v1`](https://github.com/shiahonb777/qwen2.5-1.5b-oasst-recipe/releases/tag/v1).

## Install

```bash
pip install git+https://github.com/shiahonb777/mlrecipe.git
```

Python 3.9 or newer. The optional `[torch]` extra enables a torch-native fast
path; the default pure-NumPy path is what runs today.

## Usage

If you already have a PEFT adapter directory, one command produces a recipe.

```bash
mlrecipe from-peft ./my_lora_dir --name medical-v1
```

That reads `adapter_config.json`, picks up `rank`, `lora_alpha`,
`target_modules`, `fan_in_fan_out`, and the base model reference
automatically, and stores everything in `.recipe/`.

For finer control:

```bash
mlrecipe init                       # initialize a recipe repo in cwd
mlrecipe commit \
    --name medical-v1 \
    --base meta-llama/Llama-3-8B \
    --revision <hf-commit-sha> \
    --adapter ./output/adapter_model.safetensors \
    --target-modules q_proj v_proj \
    --rank 16 --alpha 32 \
    --seed 42 --steps 10000
mlrecipe show
mlrecipe push alice/llama3-medical@v1     # uses the gh CLI
```

On any other machine:

```bash
mlrecipe clone alice/llama3-medical@v1
cd llama3-medical
mlrecipe materialize ./merged
```

### Python API

```python
from mlrecipe import from_peft, commit_from_peft, materialize, load_recipe

# Build a recipe from a live PEFT model.
recipe, adapter_bytes = from_peft(peft_model, base_ref="gpt2")

# Or from a saved adapter directory.
recipe = commit_from_peft("./my_lora_dir", repo_dir=".", name="medical-v1")
```

## Format

A `recipe.toml` records:

- `base` — Hugging Face model ref plus commit SHA (an immutable pin).
- `adapters` — ordered list of LoRA adapters, each referenced by SHA-256 of
  the adapter bytes.
- `training` — provenance (seed, steps, dataset hash). Auditable, not
  required for materialization.
- `parents` — optional pointer to another recipe, for lineage.

`mlrecipe materialize` downloads the base, loads each adapter from the local
artifact store, applies them in order via `B @ A` matrix multiplication, and
writes a complete safetensors checkpoint identical to what Hugging Face Hub
would have served.

Storage layout:

```
.recipe/
  recipe.toml          # ~50 KB of TOML
  artifacts/
    9c/9c2bc...        # content-addressed LoRA adapters
  HEAD                 # current commit / draft marker
```

GitHub releases host the bundles. The recipient downloads exactly the file
you uploaded, under your GitHub account.

## Status

Alpha. The format version is `0.1`; the version string in every recipe gets
bumped on any breaking change. The core path (commit / show / materialize /
push / clone) is tested against synthetic models, plus the two real PEFT
adapters above. PEFT, Axolotl, and unsloth-style adapter naming conventions
are matched conservatively in `materialize._match_lora_targets`; extending it
for further toolchains is the next milestone.

## Roadmap

Done:

- Recipe format v0.1, content-addressed artifacts.
- LoRA application with PEFT-style key conventions.
- `init` / `commit` / `show` / `materialize`.
- Bit-identical match against `peft.merge_and_unload` on real PEFT adapters.
- `fan_in_fan_out` / Conv1D layouts.
- bf16 / fp16 / fp32 base weights via the torch read path.
- `push` / `clone` via GitHub releases.
- PEFT integration: `mlrecipe from-peft <dir>` and the equivalent Python API.
- Web explorer: browse, search, publish.
- Browser-side materialization for small bases (GPT-2 family, Qwen 2.5 0.5B). The merge runs in JavaScript and matches PEFT's reference output to fp32 precision (max element-wise difference ~10⁻⁷).

Not done:

- Multi-shard base models in the browser path. Anything sharded across files (Qwen 1.5B and up) requires the CLI.
- Axolotl and unsloth presets.
- Recipe lineage: `mlrecipe log`, `mlrecipe parent`, `mlrecipe diff`.
- Quantized LoRA support.
- Adapter types beyond LoRA: IA³, sparse delta, full-FT delta with quant.

## License

MIT.
