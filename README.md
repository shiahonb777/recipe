# mlrecipe

A LoRA fine-tune is a 50 MB delta on top of a base model the world already has.
But today we ship it as a 14 GB merged blob, every time, and pay full cost on
every download. `mlrecipe` is a small format that ships the delta — a 200-byte
recipe pointing at the base + the SHA-256 of the LoRA — and rebuilds the merged
checkpoint locally.

The rebuilt checkpoint is bit-identical to `peft.merge_and_unload()`. Storage
and bandwidth drop by two to three orders of magnitude. Distribution is GitHub
releases or any object store. No new registry, no new server, no new account.

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

> **Project status (2026-05).** `mlrecipe` is a working reference
> implementation. It is not pitching for stars or aiming to replace Hugging
> Face Hub. The intended outcome is a discussion with `peft`/`huggingface_hub`
> maintainers about whether content-addressed adapter recipes belong in the
> standard. Browser demo and CLI both work; the question this README is here
> to answer is whether the design is right, not whether the package is
> ready to install at scale. Feedback on the recipe schema, the `materialize`
> contract, and lineage semantics is more useful right now than bug reports
> on the CLI.

## What a recipe looks like

A real one, from `examples/gpt2_alpaca/`:

```toml
[recipe]
version = "0.1"
name    = "gpt2-alpaca"

[base]
ref      = "gpt2"
revision = "11c5a3d5811f50298f278a704980280950aedb10"

[[adapters]]
type            = "lora"
artifact        = "sha256:ad2da5c75adc880818cca1692ecff1ccc5e8259a22a279ee664071fc9cb69bb4"
target_modules  = ["c_attn"]
rank            = 8
alpha           = 16.0
fan_in_fan_out  = true

[training]
method = "lora"
```

The TOML is 290 bytes. The adapter it references is 1.1 MB. Together they
materialize a 500 MB GPT-2 checkpoint that is byte-identical to PEFT's own
output, verified across 148 of 148 tensors with maximum element-wise
difference 0.0.

## Why this might be worth standardizing

Three properties the merged-checkpoint approach doesn't have:

- **Content-addressed.** Every adapter has a single canonical name (its
  SHA-256). Two users uploading the same fine-tune get the same hash. The
  recipe TOML names the base by Hugging Face commit SHA, so "the base I
  trained on" is unambiguous in a way `base_model.name_or_path` is not.

- **Auditable.** A reader can verify the merged checkpoint matches the
  recipe without trusting the publisher. Pin the base by commit SHA, hash
  the adapter, recompute locally — same bytes or a clear failure.

- **Composable.** Multiple adapters in one recipe, applied in order,
  capture stacked fine-tunes (a domain LoRA on top of an instruction LoRA,
  for example) as a deterministic build graph rather than a flattened
  blob. `parents = "..."` records lineage between recipes.

These match what content-addressed storage gave Git over CVS, and what Nix
gave package management over apt. The same shift hasn't happened for
fine-tunes yet.

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

[`run.html`](https://shiahonb777.github.io/mlrecipe/run.html) goes one step
further: it materializes a recipe entirely in the browser. JavaScript fetches
the base model from Hugging Face, fetches the adapter from the recipe repo's
raw URL, runs the LoRA merge in pure JS (`B @ A · α/r`), and offers the merged
`safetensors` file for download. The merge is verified bit-identical to PEFT
on the GPT-2 alpaca recipe. Limited to bases that fit in browser memory
(~2 GB).

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
bundle is 74 MB; the merged checkpoint is roughly 3 GB. Published as
[`shiahonb777/qwen2.5-1.5b-oasst-recipe@v1`](https://github.com/shiahonb777/qwen2.5-1.5b-oasst-recipe/releases/tag/v1).

For comparison against current practice:

| Format                        | Same fine-tune on disk |
|-------------------------------|-----------------------:|
| Hugging Face Hub (merged)     | 14,000 MB              |
| `git-lfs` chunked dedup       | ~2,000 MB              |
| `mlrecipe` (LoRA-aware)       | ~50 MB                 |

## Open design questions

These are the parts the RFC discussion should focus on, not the implementation:

- **Schema location.** Should `recipe.toml` live alongside the existing
  `adapter_config.json`, replace it, or extend it? Current `mlrecipe` reads
  `adapter_config.json` and emits `recipe.toml` separately, which is the
  most conservative choice but also the most redundant.

- **Lineage primitives.** A recipe with `parents = "alice/recipe-v1"` is
  enough to record one parent, but multi-parent merges (a la Git) and
  multi-adapter stacking (LoRA on LoRA) need a richer model. `mlrecipe`
  treats a recipe as an ordered list of adapters; whether that's right
  for adapter types beyond LoRA (IA³, sparse delta, full-FT delta) is open.

- **Hash domain.** SHA-256 of the safetensors file works, but doesn't cover
  semantically equivalent re-orderings of tensors or differing safetensors
  metadata. A canonical-form hash (sorted tensors + canonical metadata)
  would be more robust at the cost of a small canonicalization step.

- **Distribution channel.** GitHub releases work today and need no
  cooperation from anyone. But a Hub-native form — recipe as a first-class
  Hub artifact, browsable from `huggingface.co/<user>/<repo>` — would
  reduce friction for the typical user and is what makes this feel
  "standard" instead of "external."

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

## Implementation status

The format version is `0.1`. Every recipe carries the version string; a
breaking schema change increments it.

Verified:

- `init` / `commit` / `show` / `materialize` / `push` / `clone` against
  synthetic models and the two real PEFT adapters above.
- Bit-identical match against `peft.merge_and_unload` on real adapters,
  148/148 tensors, max diff 0.0.
- `fan_in_fan_out` / Conv1D layouts (GPT-2 family).
- `bf16` / `fp16` / `fp32` base weights through the torch read path.
- PEFT integration: `mlrecipe from-peft <dir>` and the equivalent Python
  API read `adapter_config.json` + `adapter_model.safetensors` directly.
- Browser materialization for single-shard bases that fit in browser
  memory (GPT-2 family, Qwen 2.5 0.5B). The merge runs in JavaScript and
  matches PEFT's reference output to fp32 precision (max element-wise
  difference ~10⁻⁷).

Not implemented:

- Multi-shard base models in the browser path. Anything sharded across
  files (Qwen 1.5B and up) needs the CLI.
- Axolotl and unsloth-style adapter naming conventions beyond what
  `_match_lora_targets` already handles.
- `mlrecipe log` / `mlrecipe parent` / `mlrecipe diff` (lineage commands;
  the `parents` field exists but no UX yet).
- Quantized LoRA / QLoRA.
- Adapter types beyond LoRA: IA³, sparse delta, full-FT delta with
  quantization.

## License

MIT.
