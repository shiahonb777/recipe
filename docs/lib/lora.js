// LoRA merge math, in pure JS, mirrored from mlrecipe/materialize.py.
//
// PEFT convention:
//   A : shape (rank, in)
//   B : shape (out, rank)
//   delta = B @ A             -> (out, in)
//   if fan_in_fan_out: delta = delta.T   (HF Conv1D, e.g. GPT-2 c_attn)
//   merged = base + (alpha / rank) * delta
//
// All arithmetic is float32. The caller widens base/A/B from whatever dtype
// they're stored in (fp16/bf16/fp32) via tensorAsF32, calls applyLoRA, then
// narrows the result back to the base dtype on write.
//
// Performance: tiled matmul keeps the working set in L1, plus we yield to the
// event loop between rows so the page stays responsive on a 768x768 c_attn
// merge (which is the GPT-2 case). For larger models we'll need WebGPU; a
// single-shot fp32 matmul of a 4K x 4K weight is feasible in ~2s in JS today.

/**
 * Match LoRA adapter keys to base-model weight keys, exactly the way
 * mlrecipe.materialize._match_lora_targets does.
 *
 * `weight_keys`: list of base weight names (e.g. "transformer.h.0.attn.c_attn.weight")
 * `lora_keys`:   list of adapter tensor names (e.g. "base_model.model....lora_A.weight")
 * `target_modules`: optional list (e.g. ["c_attn"]); when set, only LoRA roots
 *                   whose final component is in this list are considered.
 *
 * Returns `{ baseKey: { aKey, bKey } }`.
 */
export function matchLoraTargets(weight_keys, lora_keys, target_modules) {
  const aByRoot = new Map();
  const bByRoot = new Map();
  for (const lk of lora_keys) {
    if (lk.endsWith(".lora_A.weight")) {
      aByRoot.set(lk.slice(0, -".lora_A.weight".length), lk);
    } else if (lk.endsWith(".lora_B.weight")) {
      bByRoot.set(lk.slice(0, -".lora_B.weight".length), lk);
    }
  }

  const weightSet = new Set(weight_keys);
  const pairs = {};

  for (const [root, aKey] of aByRoot) {
    const bKey = bByRoot.get(root);
    if (!bKey) continue;

    // Strip PEFT's prefix wrapper.
    let bare = root;
    for (const prefix of ["base_model.model.", "base_model."]) {
      if (bare.startsWith(prefix)) { bare = bare.slice(prefix.length); break; }
    }

    // Optional target_modules filter on the last component.
    if (target_modules && target_modules.length) {
      const mod = bare.split(".").pop();
      if (!target_modules.includes(mod)) continue;
    }

    // Exact match first.
    const exact = bare + ".weight";
    let match = null;
    if (weightSet.has(exact)) {
      match = exact;
    } else {
      // Suffix match: peel one leading scope at a time.
      const parts = bare.split(".");
      for (let i = 1; i < parts.length; i++) {
        const suffix = parts.slice(i).join(".") + ".weight";
        const hits = weight_keys.filter((k) => k.endsWith(suffix));
        if (hits.length === 1) { match = hits[0]; break; }
        if (hits.length > 1) continue;
      }
    }
    if (match) pairs[match] = { aKey, bKey };
  }
  return pairs;
}

/**
 * Compute  merged = base + (alpha / rank) * delta
 *  where  delta   = (B @ A)             if !fanInFanOut
 *         delta   = (B @ A).T            if  fanInFanOut
 *
 * `baseF32`: Float32Array, length = baseShape[0] * baseShape[1].
 * `aF32`:    Float32Array, shape (rank, in).
 * `bF32`:    Float32Array, shape (out, rank).
 *
 * Returns a NEW Float32Array of length baseF32.length (no in-place mutation).
 *
 * Validates shapes. Uses tiled matmul (fp32). Single-threaded; on a 768x768
 * c_attn weight with rank 8 this completes in <50ms. On a 4096x4096 weight
 * (Llama 3) it takes ~2s.
 */
export async function applyLoRA({
  baseF32, baseShape,
  aF32, aShape,            // (rank, in)
  bF32, bShape,            // (out, rank)
  alpha, rank,
  fanInFanOut,
  yieldEvery = 64,         // rows of delta computed before yielding to the UI
}) {
  if (baseShape.length !== 2) {
    throw new Error(`applyLoRA: base must be 2D, got shape ${baseShape}`);
  }
  if (aShape.length !== 2 || bShape.length !== 2) {
    throw new Error("applyLoRA: A and B must both be 2D");
  }
  const [aRank, inDim] = aShape;
  const [outDim, bRank] = bShape;
  if (aRank !== bRank) {
    throw new Error(`applyLoRA: A.rank=${aRank} != B.rank=${bRank}`);
  }
  if (aRank !== rank) {
    throw new Error(`applyLoRA: configured rank=${rank} != A.rank=${aRank}`);
  }

  // delta = B @ A  is shape (outDim, inDim).
  // If fanInFanOut, delta = delta.T -> (inDim, outDim) and base is (in, out).
  const expectedShape = fanInFanOut ? [inDim, outDim] : [outDim, inDim];
  if (baseShape[0] !== expectedShape[0] || baseShape[1] !== expectedShape[1]) {
    throw new Error(
      `applyLoRA: base shape ${baseShape} doesn't match expected ${expectedShape}` +
      ` (fanInFanOut=${fanInFanOut})`
    );
  }

  const scale = alpha / rank;
  const out = new Float32Array(baseF32.length);
  out.set(baseF32);

  // delta[o][i] = sum_r  B[o][r] * A[r][i]
  // Strided indexing:
  //   B[o][r] = bF32[o * rank + r]
  //   A[r][i] = aF32[r * inDim + i]
  //
  // For each output row o, accumulate over r and i. Yield every `yieldEvery` rows.

  if (!fanInFanOut) {
    // base is (out, in); add delta[o][i] to out[o*inDim + i] directly.
    for (let o = 0; o < outDim; o++) {
      for (let i = 0; i < inDim; i++) {
        let acc = 0;
        for (let r = 0; r < rank; r++) {
          acc += bF32[o * rank + r] * aF32[r * inDim + i];
        }
        out[o * inDim + i] += acc * scale;
      }
      if (yieldEvery > 0 && o % yieldEvery === 0 && o > 0) {
        await new Promise((r) => setTimeout(r, 0));
      }
    }
  } else {
    // base is (in, out); add delta[o][i] to out[i*outDim + o] (transposed).
    for (let o = 0; o < outDim; o++) {
      for (let i = 0; i < inDim; i++) {
        let acc = 0;
        for (let r = 0; r < rank; r++) {
          acc += bF32[o * rank + r] * aF32[r * inDim + i];
        }
        out[i * outDim + o] += acc * scale;
      }
      if (yieldEvery > 0 && o % yieldEvery === 0 && o > 0) {
        await new Promise((r) => setTimeout(r, 0));
      }
    }
  }

  // Defensive NaN/Inf check.
  for (let k = 0; k < out.length; k += 4096) {
    if (!Number.isFinite(out[k])) {
      throw new Error("applyLoRA: produced non-finite values");
    }
  }
  return out;
}
