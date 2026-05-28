// Cross-check applyLoRA against a known-good NumPy/PyTorch result.
//
// We don't shell to Python; instead, we hand-build a tiny case where the
// math is verifiable by inspection, and compare to a pre-computed reference.

import { applyLoRA, matchLoraTargets } from "./lora.js";

// Test 1: Identity check.
//
// base = [[1,2,3],[4,5,6]]  (out=2, in=3)
// A = [[1,0,0]]            (rank=1, in=3)
// B = [[2],[3]]            (out=2, rank=1)
// alpha=2, rank=1 -> scale=2
// delta = B@A = [[2,0,0],[3,0,0]]
// scaled = [[4,0,0],[6,0,0]]
// merged = [[5,2,3],[10,5,6]]

async function test1() {
  const base = new Float32Array([1, 2, 3, 4, 5, 6]);
  const A = new Float32Array([1, 0, 0]);
  const B = new Float32Array([2, 3]);
  const out = await applyLoRA({
    baseF32: base, baseShape: [2, 3],
    aF32: A, aShape: [1, 3],
    bF32: B, bShape: [2, 1],
    alpha: 2, rank: 1, fanInFanOut: false,
    yieldEvery: 0,
  });
  const expected = [5, 2, 3, 10, 5, 6];
  for (let i = 0; i < expected.length; i++) {
    if (Math.abs(out[i] - expected[i]) > 1e-6) {
      throw new Error(`test1: out[${i}]=${out[i]} expected ${expected[i]}`);
    }
  }
  console.log("test1 (basic merge) OK");
}

// Test 2: fan_in_fan_out (Conv1D layout).
//
// base = (in=3, out=2) = [[1,2],[3,4],[5,6]]
// A    = (rank=1, in=3) = [[1,0,0]]
// B    = (out=2, rank=1) = [[2],[3]]
// delta = B@A = [[2,0,0],[3,0,0]]  shape (out=2, in=3)
// delta.T   = [[2,3],[0,0],[0,0]]   shape (in=3, out=2)  matches base
// alpha=2, rank=1 -> scale=2
// scaled = [[4,6],[0,0],[0,0]]
// merged = [[5,8],[3,4],[5,6]]

async function test2() {
  const base = new Float32Array([1, 2, 3, 4, 5, 6]);
  const A = new Float32Array([1, 0, 0]);
  const B = new Float32Array([2, 3]);
  const out = await applyLoRA({
    baseF32: base, baseShape: [3, 2],
    aF32: A, aShape: [1, 3],
    bF32: B, bShape: [2, 1],
    alpha: 2, rank: 1, fanInFanOut: true,
    yieldEvery: 0,
  });
  const expected = [5, 8, 3, 4, 5, 6];
  for (let i = 0; i < expected.length; i++) {
    if (Math.abs(out[i] - expected[i]) > 1e-6) {
      throw new Error(`test2 (Conv1D): out[${i}]=${out[i]} expected ${expected[i]}`);
    }
  }
  console.log("test2 (Conv1D fan_in_fan_out) OK");
}

// Test 3: matchLoraTargets with PEFT's typical wrapping.
function test3() {
  const weights = [
    "transformer.h.0.attn.c_attn.weight",
    "transformer.h.0.attn.c_proj.weight",
    "transformer.h.1.attn.c_attn.weight",
    "lm_head.weight",
  ];
  const lora = [
    "base_model.model.transformer.h.0.attn.c_attn.lora_A.weight",
    "base_model.model.transformer.h.0.attn.c_attn.lora_B.weight",
    "base_model.model.transformer.h.1.attn.c_attn.lora_A.weight",
    "base_model.model.transformer.h.1.attn.c_attn.lora_B.weight",
  ];
  const pairs = matchLoraTargets(weights, lora, ["c_attn"]);
  if (Object.keys(pairs).length !== 2) {
    throw new Error(`test3: expected 2 pairs, got ${Object.keys(pairs).length}`);
  }
  const k = "transformer.h.0.attn.c_attn.weight";
  if (!pairs[k]) throw new Error(`test3: missing ${k}`);
  if (!pairs[k].aKey.endsWith(".lora_A.weight")) throw new Error("test3: aKey wrong");
  if (!pairs[k].bKey.endsWith(".lora_B.weight")) throw new Error("test3: bKey wrong");
  console.log("test3 (matchLoraTargets) OK");
}

// Test 4: rank=2 sanity check.
//
// base = [[0,0],[0,0]]  (2x2)
// A = [[1,0],[0,1]]     (rank=2, in=2)
// B = [[3,4],[5,6]]     (out=2, rank=2)
// delta = B@A = [[3,4],[5,6]]
// alpha=4, rank=2 -> scale=2
// merged = [[6,8],[10,12]]
async function test4() {
  const base = new Float32Array([0, 0, 0, 0]);
  const A = new Float32Array([1, 0, 0, 1]);
  const B = new Float32Array([3, 4, 5, 6]);
  const out = await applyLoRA({
    baseF32: base, baseShape: [2, 2],
    aF32: A, aShape: [2, 2],
    bF32: B, bShape: [2, 2],
    alpha: 4, rank: 2, fanInFanOut: false,
    yieldEvery: 0,
  });
  const expected = [6, 8, 10, 12];
  for (let i = 0; i < expected.length; i++) {
    if (Math.abs(out[i] - expected[i]) > 1e-6) {
      throw new Error(`test4: out[${i}]=${out[i]} expected ${expected[i]}`);
    }
  }
  console.log("test4 (rank>1) OK");
}

await test1();
await test2();
test3();
await test4();
console.log("all LoRA tests OK");
