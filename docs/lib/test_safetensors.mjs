// Round-trip test for safetensors read/write.

import { readSafetensors, writeSafetensors, tensorAsF32, f32ToBuffer } from "./safetensors.js";

// Build a tiny fake safetensors file by hand, then read it.
// Layout: 8-byte header length, JSON header, raw bytes.
function buildFakeFile() {
  const tensors = [
    { name: "a", dtype: "F32", shape: [2, 3], values: new Float32Array([1, 2, 3, 4, 5, 6]) },
    { name: "b", dtype: "F32", shape: [3], values: new Float32Array([7, 8, 9]) },
  ];
  let cursor = 0;
  const header = {};
  for (const t of tensors) {
    const bytes = t.values.length * 4;
    header[t.name] = { dtype: t.dtype, shape: t.shape, data_offsets: [cursor, cursor + bytes] };
    cursor += bytes;
  }
  const headerJSON = JSON.stringify(header);
  const padded = headerJSON + " ".repeat((8 - (headerJSON.length % 8)) % 8);
  const headerBytes = new TextEncoder().encode(padded);
  const total = 8 + headerBytes.length + cursor;
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, headerBytes.length, true);
  dv.setUint32(4, 0, true);
  out.set(headerBytes, 8);
  let off = 8 + headerBytes.length;
  for (const t of tensors) {
    const bytes = new Uint8Array(t.values.buffer, t.values.byteOffset, t.values.byteLength);
    out.set(bytes, off);
    off += bytes.length;
  }
  return { file: out, expected: tensors };
}

function test1() {
  const { file, expected } = buildFakeFile();
  const parsed = readSafetensors(file);
  for (const e of expected) {
    const t = parsed.tensors[e.name];
    if (!t) throw new Error(`test1: missing tensor ${e.name}`);
    const f32 = tensorAsF32(t);
    if (f32.length !== e.values.length) throw new Error(`test1: ${e.name} length mismatch`);
    for (let i = 0; i < e.values.length; i++) {
      if (Math.abs(f32[i] - e.values[i]) > 1e-6) {
        throw new Error(`test1: ${e.name}[${i}] = ${f32[i]} expected ${e.values[i]}`);
      }
    }
  }
  console.log("test1 (read fp32) OK");
}

function test2() {
  // Round-trip: read, then write back, byte-equal.
  const { file } = buildFakeFile();
  const parsed = readSafetensors(file);
  const tensors = Object.values(parsed.tensors).map((t) => ({
    name: t.name, dtype: t.dtype, shape: t.shape, bytes: t.view,
  }));
  const written = writeSafetensors(tensors, parsed.metadata);
  // Header layout may differ in spacing if our write padding logic
  // differs from the original; the tensor bytes should match exactly.
  const reparsed = readSafetensors(written);
  for (const name of Object.keys(parsed.tensors)) {
    const a = parsed.tensors[name];
    const b = reparsed.tensors[name];
    if (a.byteLength !== b.byteLength) throw new Error(`test2: ${name} length mismatch`);
    if (a.dtype !== b.dtype) throw new Error(`test2: ${name} dtype mismatch`);
    if (a.shape.join(",") !== b.shape.join(",")) throw new Error(`test2: ${name} shape mismatch`);
    for (let i = 0; i < a.byteLength; i++) {
      if (a.view[i] !== b.view[i]) throw new Error(`test2: ${name} byte[${i}] mismatch`);
    }
  }
  console.log("test2 (round-trip fp32) OK");
}

function test3() {
  // f32 -> f16 -> f32 should be lossy but stable for representable values.
  const original = new Float32Array([0, 1, -1, 0.5, -0.5, 65504, -65504]);
  const f16Bytes = f32ToBuffer("F16", original);
  // Re-read using a fake file.
  const fake = {
    name: "x", dtype: "F16", shape: [original.length],
    byteLength: f16Bytes.byteLength, view: f16Bytes,
  };
  const back = tensorAsF32(fake);
  if (back.length !== original.length) throw new Error("test3: length");
  for (let i = 0; i < original.length; i++) {
    if (Math.abs(back[i] - original[i]) > 1e-3) {
      throw new Error(`test3: f16 round-trip [${i}] = ${back[i]} expected ${original[i]}`);
    }
  }
  console.log("test3 (f16 round-trip) OK");
}

function test4() {
  // bf16 round-trip — same idea.
  const original = new Float32Array([0, 1, -1, 100, -100, 1e10, -1e10]);
  const bf16Bytes = f32ToBuffer("BF16", original);
  const fake = {
    name: "x", dtype: "BF16", shape: [original.length],
    byteLength: bf16Bytes.byteLength, view: bf16Bytes,
  };
  const back = tensorAsF32(fake);
  if (back.length !== original.length) throw new Error("test4: length");
  // bf16 has ~1% precision; check relative error.
  for (let i = 0; i < original.length; i++) {
    const rel = original[i] === 0 ? Math.abs(back[i]) : Math.abs((back[i] - original[i]) / original[i]);
    if (rel > 0.01) {
      throw new Error(`test4: bf16 round-trip [${i}] = ${back[i]} expected ${original[i]}`);
    }
  }
  console.log("test4 (bf16 round-trip) OK");
}

test1();
test2();
test3();
test4();
console.log("all safetensors tests OK");
