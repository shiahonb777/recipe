# recipe

**Ship model recipes, not weights.**

A 14 GB fine-tune becomes a 50 KB recipe + a tiny LoRA adapter. The
receiver re-derives the merged checkpoint locally, bit-exactly.

```
$ mlrecipe push alice/llama3-medical
bundling -> .recipe-bundle-v1.tar.gz
bundle size: 53,712 bytes        ← not 14 GB
pushed: https://github.com/alice/llama3-medical/releases/tag/v1

$ mlrecipe clone alice/llama3-medical
fetching alice/llama3-medical@latest
unpacking...
cloned recipe `llama3-medical` into llama3-medical/

$ mlrecipe materialize ./merged
materializing llama3-medical -> merged
done. checkpoint at merged
```

## Why

A LoRA fine-tune isn't a 14 GB blob — it's a 50 MB delta on top of a base
model the world already has. Today's tools (HF Hub, git-lfs) don't know
that. They store and transfer the merged result, paying the full cost
every time.

`mlrecipe` treats a fine-tune as a small program: `base + adapter + training
metadata`. The receiver runs the program locally and gets back the same
weights. Storage and bandwidth drop ~1000x for LoRA-style fine-tunes.

| Format | LoRA fine-tune size on disk |
|---|---|
| HF Hub (merged safetensors) | 14,000 MB |
| `git-lfs` chunked dedup | ~2,000 MB |
| `mlrecipe` (LoRA-aware) | **~50 MB** |

(numbers from `research_pocs/apex2_weights_program/` — measured against
toy models with realistic spectra; full-model fine-tunes still get ~5x via
tensor-aware delta.)

## Real example

There's a working end-to-end demo in
[`examples/gpt2_alpaca/`](examples/gpt2_alpaca/) that:

- Downloads a real PEFT LoRA from HF Hub (`monsterapi/gpt2_alpaca-lora`)
- Packages it as an `mlrecipe` (1.1 MB instead of 500 MB)
- Materializes it into a merged GPT-2 checkpoint
- Verifies the result is **bit-identical** to PEFT's official
  `merge_and_unload` (148/148 tensors, max abs diff = 0.0)

```bash
bash examples/gpt2_alpaca/run.sh
```

A live, distributable copy is at
[`shiahonb777/gpt2-alpaca-recipe@v1`](https://github.com/shiahonb777/gpt2-alpaca-recipe/releases/tag/v1).
You can clone and materialize it from any machine:

```bash
mlrecipe clone shiahonb777/gpt2-alpaca-recipe@v1
cd gpt2-alpaca-recipe
mlrecipe materialize ./merged
```

The recipe is **1.1 MB**. The merged checkpoint is **~500 MB**.

## Install

```bash
pip install git+https://github.com/shiahonb777/mlrecipe.git
```

Verify:

```bash
mlrecipe --help
```

Requires Python 3.9+. The optional `[torch]` extra enables a future
torch-native fast path; the default pure-NumPy path is what runs today.

## Usage

If you have an existing PEFT adapter (saved by `peft.PeftModel.save_pretrained()`),
the fastest path is one command:

```bash
mlrecipe from-peft ./my_lora_dir --name medical-v1
```

That reads `adapter_config.json`, picks up `rank`, `lora_alpha`,
`target_modules`, `fan_in_fan_out`, and the base model ref
automatically, and stores everything in `.recipe/`.

If you'd rather drive it explicitly:

```bash
# Initialize a recipe repo in your fine-tune directory
mlrecipe init

# Record a recipe pointing to a HF base model and a local LoRA adapter
mlrecipe commit \
    --name medical-v1 \
    --base meta-llama/Llama-3-8B \
    --revision <hf-commit-sha> \
    --adapter ./output/adapter_model.safetensors \
    --target-modules q_proj v_proj \
    --rank 16 --alpha 32 \
    --seed 42 --steps 10000

# Inspect
mlrecipe show

# Push to a GitHub Release (uses gh CLI; requires SSH or HTTPS auth)
mlrecipe push alice/llama3-medical@v1

# On any other machine:
mlrecipe clone alice/llama3-medical@v1
cd llama3-medical
mlrecipe materialize ./merged
```

### Python API

```python
from mlrecipe import from_peft, commit_from_peft, materialize, load_recipe

# Build a recipe from a PEFT model in memory:
recipe, adapter_bytes = from_peft(peft_model, base_ref="gpt2")

# Or from a saved adapter directory, all in one call:
recipe = commit_from_peft("./my_lora_dir", repo_dir=".", name="medical-v1")
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
- [x] Real-world LoRA verification: bit-identical to PEFT
      `merge_and_unload` (see `examples/gpt2_alpaca/`)
- [x] `fan_in_fan_out` / Conv1D layouts (GPT-2 etc.)
- [x] `push` / `clone` via GitHub Releases
- [x] PEFT integration: `mlrecipe from-peft <dir>` and
      `mlrecipe.from_peft()` / `mlrecipe.commit_from_peft()` Python API
- [ ] Axolotl / unsloth presets
- [ ] Recipe lineage: `mlrecipe log`, `mlrecipe parent`, `mlrecipe diff`
- [ ] Quantized LoRA (QLoRA) support
- [ ] Adapter types beyond LoRA: IA³, sparse delta, full-FT delta+quant
- [ ] Web "Recipe Explorer" — paste a recipe URL, see lineage tree

## License

MIT.
