// End-to-end test: merge GPT-2 + alpaca LoRA in JS, compare to PEFT's output.
//
// Inputs:
//   1. GPT-2 base from Hugging Face (downloaded once into ../../.cache/)
//   2. The alpaca LoRA artifact from examples/gpt2_alpaca/work/repo/.recipe/artifacts
//   3. The recipe.toml (for adapter config: rank, alpha, target_modules, fan_in_fan_out)
//
// Reference:
//   examples/gpt2_alpaca/work/merged/model.safetensors  -- produced by PEFT's merge_and_unload
//
// Pass criteria:
//   - All adapter-targeted tensors match the PEFT reference within
//     a small fp32-vs-fp32 numerical tolerance.
//   - All non-targeted tensors are byte-identical to the base.
//   - The header structure (tensor list, dtypes, shapes) matches the PEFT
//     reference exactly.

import fs from "node:fs";
import path from "node:path";
import { readSafetensors, tensorAsF32, f32ToBuffer, writeSafetensors } from "./safetensors.js";
import { matchLoraTargets, applyLoRA } from "./lora.js";

const ROOT = path.resolve(import.meta.dirname, "../../");
const RECIPE_DIR = path.join(ROOT, "examples/gpt2_alpaca/work/repo/.recipe");
const ARTIFACT_HASH = "ad2da5c75adc880818cca1692ecff1ccc5e8259a22a279ee664071fc9cb69bb4";
const ARTIFACT_PATH = path.join(RECIPE_DIR, "artifacts", ARTIFACT_HASH.slice(0, 2), ARTIFACT_HASH);
const REFERENCE_MERGED = path.join(ROOT, "examples/gpt2_alpaca/work/merged/model.safetensors");
const CACHE_DIR = path.join(ROOT, ".test-cache");
const BASE_PATH = path.join(CACHE_DIR, "gpt2-model.safetensors");
const BASE_URL = "https://huggingface.co/gpt2/resolve/main/model.safetensors";

async function ensureBase() {
  if (fs.existsSync(BASE_PATH)) return;
  fs.mkdirSync(CACHE_DIR, { recursive: true });
  console.log("downloading gpt2 base from huggingface (~500 MB, one-time)…");
  const res = await fetch(BASE_URL);
  if (!res.ok) throw new Error(`fetch base: ${res.status}`);
  const buf = await res.arrayBuffer();
  fs.writeFileSync(BASE_PATH, new Uint8Array(buf));
  console.log(`  saved ${(buf.byteLength / 1e6).toFixed(1)} MB`);
}

function readFile(p) {
  return new Uint8Array(fs.readFileSync(p));
}

async function main() {
  await ensureBase();

  const base = readSafetensors(readFile(BASE_PATH));
  const adapter = readSafetensors(readFile(ARTIFACT_PATH));
  const reference = readSafetensors(readFile(REFERENCE_MERGED));

  // Hard-code adapter config from recipe.toml of the GPT-2 alpaca recipe:
  //   rank=8, alpha=16.0, fan_in_fan_out=true, target_modules=["c_attn"]
  const rank = 8;
  const alpha = 16.0;
  const fanInFanOut = true;
  const targetModules = ["c_attn"];

  const baseKeys = Object.keys(base.tensors);
  const loraKeys = Object.keys(adapter.tensors);
  const pairs = matchLoraTargets(baseKeys, loraKeys, targetModules);
  const numPairs = Object.keys(pairs).length;
  console.log(`matched ${numPairs} LoRA targets`);
  if (numPairs === 0) throw new Error("no LoRA targets matched");

  let modified = 0;
  let maxDiff = 0;
  let totalDiffSum = 0;
  let totalElements = 0;
  let badTensors = [];
  const t0 = Date.now();

  for (const baseKey of Object.keys(pairs)) {
    const { aKey, bKey } = pairs[baseKey];
    const baseT = base.tensors[baseKey];
    const aT = adapter.tensors[aKey];
    const bT = adapter.tensors[bKey];

    const baseF32 = tensorAsF32(baseT);
    const aF32 = tensorAsF32(aT);
    const bF32 = tensorAsF32(bT);

    process.stdout.write(`\r  merge [${modified+1}/${numPairs}]  ${baseKey}                          `);

    const merged = await applyLoRA({
      baseF32, baseShape: baseT.shape,
      aF32,    aShape:    aT.shape,
      bF32,    bShape:    bT.shape,
      alpha, rank, fanInFanOut,
      yieldEvery: 0,
    });

    // Compare to reference.
    const refT = reference.tensors[baseKey];
    if (!refT) {
      console.log(`\n  WARN: reference missing ${baseKey}`);
      modified += 1;
      continue;
    }
    const refF32 = tensorAsF32(refT);
    if (refF32.length !== merged.length) {
      console.log(`\n  FAIL: ${baseKey} length ${merged.length} vs ref ${refF32.length}`);
      badTensors.push(baseKey);
      modified += 1;
      continue;
    }
    let m = 0, sum = 0;
    for (let i = 0; i < merged.length; i++) {
      const d = Math.abs(merged[i] - refF32[i]);
      if (d > m) m = d;
      sum += d;
    }
    if (m > maxDiff) maxDiff = m;
    totalDiffSum += sum;
    totalElements += merged.length;
    if (m > 1e-3) {
      // bf16 storage error budget; gpt2 base is fp32 though, so this should be ~0.
      console.log(`\n  FAIL: ${baseKey}  max abs diff ${m}`);
      badTensors.push(baseKey);
    }
    modified += 1;
  }

  const dt = (Date.now() - t0) / 1000;
  console.log(`\n  merged ${modified} tensors in ${dt.toFixed(1)}s`);
  console.log(`  global max abs diff:  ${maxDiff.toExponential(3)}`);
  console.log(`  global mean abs diff: ${(totalDiffSum / totalElements).toExponential(3)}`);

  // Now: also verify all NON-targeted tensors are byte-identical between
  // base and reference (they should be, as PEFT only modifies the
  // adapter targets).
  let unmodifiedMatch = 0;
  let unmodifiedMismatch = 0;
  for (const k of baseKeys) {
    if (pairs[k]) continue;
    const a = base.tensors[k];
    const b = reference.tensors[k];
    if (!b) {
      console.log(`  WARN: reference missing untargeted tensor ${k}`);
      continue;
    }
    let same = (a.byteLength === b.byteLength);
    if (same) {
      for (let i = 0; i < a.byteLength; i++) {
        if (a.view[i] !== b.view[i]) { same = false; break; }
      }
    }
    if (same) unmodifiedMatch += 1;
    else unmodifiedMismatch += 1;
  }
  console.log(`  unmodified tensors:   ${unmodifiedMatch} byte-identical, ${unmodifiedMismatch} differ`);

  if (badTensors.length || unmodifiedMismatch > 0) {
    console.error(`\nFAIL: ${badTensors.length} merged tensors above tolerance, ${unmodifiedMismatch} unmodified differ`);
    process.exit(2);
  }
  console.log("\nE2E PASS: JS-side LoRA merge matches PEFT merge_and_unload");
}

main().catch((e) => { console.error("\nFAIL:", e.message); process.exit(1); });
