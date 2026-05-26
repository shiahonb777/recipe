# recipe

**Ship model recipes, not weights.**

A 14 GB fine-tune becomes a 50 KB recipe + a tiny LoRA adapter. The
receiver re-derives the merged checkpoint locally, bit-exactly.

```
$ recipe push alice/llama3-medical
bundling -> .recipe-bundle-v1.tar.gz
bundle size: 53,712 bytes        ← not 14 GB
pushed: https://github.com/alice/llama3-medical/releases/tag/v1

$ recipe clone alice/llama3-medical
fetching alice/llama3-medical@latest
unpacking...
cloned recipe `llama3-medical` into llama3-medical/

$ recipe materialize ./merged
materializing llama3-medical -> merged
done. checkpoint at merged
```

## Why

A LoRA fine-tune isn't a 14 GB blob — it's a 50 MB delta on top of a base
model the world already has. Today's tools (HF Hub, git-lfs) don't know
that. They store and transfer the merged result, paying the full cost
every time.

`recipe` treats a fine-tune as a small program: `base + adapter + training
metadata`. The receiver runs the program locally and gets back the same
weights. Storage and bandwidth drop ~1000x for LoRA-style fine-tunes.

| Format | LoRA fine-tune size on disk |
|---|---|
| HF Hub (merged safetensors) | 14,000 MB |
| `git-lfs` chunked dedup | ~2,000 MB |
| `recipe` (LoRA-aware) | **~50 MB** |

(numbers from `research_pocs/apex2_weights_program/` — measured against
toy models with realistic spectra; full-model fine-tunes still get ~5x via
tensor-aware delta.)

## Install

> Note: Not yet published to PyPI. The install command below will work
> once 0.1.0 is released. Until then, install from source.

```bash
# From PyPI (planned):
pip install mlrecipe

# From source (today):
git clone https://github.com/shiahonb777/mlrecipe
cd mlrecipe
pip install -e .
```

The CLI is called `recipe`. (The PyPI package is `mlrecipe` because the
shorter name is taken; the CLI is the user-facing brand.)

```bash
recipe --help
```

Requires Python 3.9+. The optional `[torch]` extra enables a future
torch-native fast path; the default pure-NumPy path is what runs today.

## Usage

```bash
# Initialize a recipe repo in your fine-tune directory
recipe init

# Record a recipe pointing to a HF base model and a local LoRA adapter
recipe commit \
    --name medical-v1 \
    --base meta-llama/Llama-3-8B \
    --revision <hf-commit-sha> \
    --adapter ./output/adapter_model.safetensors \
    --target-modules q_proj v_proj \
    --rank 16 --alpha 32 \
    --seed 42 --steps 10000

# Inspect
recipe show

# Push to a GitHub Release (uses gh CLI; requires SSH or HTTPS auth)
recipe push alice/llama3-medical@v1

# On any other machine:
recipe clone alice/llama3-medical@v1
cd llama3-medical
recipe materialize ./merged
```

## How it works

A `recipe.toml` records:

- `base`: HF Hub model ref + commit SHA (immutable pin)
- `adapters`: ordered list of LoRA adapters (referenced by SHA-256)
- `training`: provenance (seed, steps, dataset hash) — auditable, not
  required for materialization
- `parents`: optional reference to another recipe (for lineage)

`recipe materialize` downloads the base, fetches each adapter from the
local artifact store, applies them in order via `B @ A` matrix
multiplication, and writes a complete safetensors checkpoint identical
to what HF Hub would have served.

Storage layout:

```
.recipe/
  recipe.toml          # the recipe (50KB)
  artifacts/
    9c/9c2bc...        # content-addressed LoRA adapters
  HEAD                 # current commit / draft marker
```

GitHub Releases host the bundles. No proprietary registry. No new
servers to trust. The recipient sees exactly the file you uploaded under
your own GitHub account.

## Status

Alpha. The format is `0.1`; we'll bump it on any breaking change. The
core path (commit / show / materialize / push / clone) is tested with
synthetic models. Real-world LoRA fine-tunes from PEFT / Axolotl /
unsloth are the next milestone — the matching logic in
`materialize._match_lora_targets` is conservative and may need extension
for non-default naming conventions.

## Roadmap

- [x] Recipe format v0.1, content-addressed artifacts
- [x] LoRA application (PEFT-style key conventions)
- [x] `init` / `commit` / `show` / `materialize`
- [x] `push` / `clone` via GitHub Releases
- [ ] PEFT integration: `recipe.from_peft(model)` one-liner
- [ ] Axolotl / unsloth presets
- [ ] Recipe lineage: `recipe log`, `recipe parent`, `recipe diff`
- [ ] Quantized LoRA (QLoRA) support
- [ ] Adapter types beyond LoRA: IA³, sparse delta, full-FT delta+quant
- [ ] Web "Recipe Explorer" — paste a recipe URL, see lineage tree

## License

MIT.
