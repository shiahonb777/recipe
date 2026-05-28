// Pure-JS safetensors v0.4 reader and writer.
//
// Spec (https://github.com/huggingface/safetensors):
//   bytes 0..7      uint64 LE: header length (N)
//   bytes 8..8+N    UTF-8 JSON: { tensor_name: { dtype, shape, data_offsets: [a, b] }, ... }
//   bytes 8+N..end  raw tensor bytes; offsets are relative to (8 + N)
//
// We support fp32, fp16, bf16, fp64, int*, uint*, bool — covering anything
// PEFT or transformers actually emit. For arithmetic we widen everything to
// Float32Array via `tensorAsF32` and write back via `f32ToBuffer(dtype, …)`.

export const DTYPES = {
  F64: { bytes: 8, view: "Float64Array" },
  F32: { bytes: 4, view: "Float32Array" },
  F16: { bytes: 2, view: null },        // no native typed array; widen to f32
  BF16: { bytes: 2, view: null },       // ditto
  I64: { bytes: 8, view: "BigInt64Array" },
  I32: { bytes: 4, view: "Int32Array" },
  I16: { bytes: 2, view: "Int16Array" },
  I8:  { bytes: 1, view: "Int8Array" },
  U64: { bytes: 8, view: "BigUint64Array" },
  U32: { bytes: 4, view: "Uint32Array" },
  U16: { bytes: 2, view: "Uint16Array" },
  U8:  { bytes: 1, view: "Uint8Array" },
  BOOL: { bytes: 1, view: "Uint8Array" },
};

/**
 * Parse a safetensors file from an ArrayBuffer or Uint8Array.
 * Returns { metadata, tensors } where each tensor is { dtype, shape, view, byteOffset, byteLength }.
 * Tensor bytes are NOT copied; `view` is a Uint8Array viewing into the original buffer
 * over the tensor's slice. Use `tensorAsF32(tensor)` to get a typed Float32Array (with
 * a copy if dtype is fp16/bf16, since JS has no native types for those).
 */
export function readSafetensors(buf) {
  const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  if (u8.byteLength < 8) throw new Error("safetensors: file too short");
  const dv = new DataView(u8.buffer, u8.byteOffset, 8);
  // header length is uint64 LE; in practice always < 2^32, take low 32.
  const lo = dv.getUint32(0, true);
  const hi = dv.getUint32(4, true);
  if (hi !== 0) throw new Error("safetensors: header > 4 GB; not supported");
  const headerLen = lo;
  if (8 + headerLen > u8.byteLength) throw new Error("safetensors: header overruns file");
  const headerJSON = new TextDecoder().decode(u8.subarray(8, 8 + headerLen));
  const header = JSON.parse(headerJSON);
  const metadata = header.__metadata__ || {};
  const tensors = {};
  const dataOffset = 8 + headerLen;
  for (const [name, spec] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const [a, b] = spec.data_offsets;
    tensors[name] = {
      name,
      dtype: spec.dtype,
      shape: spec.shape.map(Number),
      byteOffset: dataOffset + a,
      byteLength: b - a,
      view: u8.subarray(dataOffset + a, dataOffset + b),
    };
  }
  return { metadata, tensors, headerLen, dataOffset };
}

/**
 * Convert a tensor's bytes to a Float32Array (always a copy for fp16/bf16,
 * a typed-array view for fp32, a converted copy for everything else).
 */
export function tensorAsF32(t) {
  const u8 = t.view;
  switch (t.dtype) {
    case "F32": {
      // Aligned access required; copy to a fresh aligned buffer.
      const out = new Float32Array(t.byteLength / 4);
      const dv = new DataView(u8.buffer, u8.byteOffset, t.byteLength);
      for (let i = 0; i < out.length; i++) out[i] = dv.getFloat32(i * 4, true);
      return out;
    }
    case "F64": {
      const out = new Float32Array(t.byteLength / 8);
      const dv = new DataView(u8.buffer, u8.byteOffset, t.byteLength);
      for (let i = 0; i < out.length; i++) out[i] = dv.getFloat64(i * 8, true);
      return out;
    }
    case "F16": {
      return f16BytesToF32(u8);
    }
    case "BF16": {
      return bf16BytesToF32(u8);
    }
    case "I32": {
      const out = new Float32Array(t.byteLength / 4);
      const dv = new DataView(u8.buffer, u8.byteOffset, t.byteLength);
      for (let i = 0; i < out.length; i++) out[i] = dv.getInt32(i * 4, true);
      return out;
    }
    case "I64": {
      const out = new Float32Array(t.byteLength / 8);
      const dv = new DataView(u8.buffer, u8.byteOffset, t.byteLength);
      for (let i = 0; i < out.length; i++) {
        const lo = dv.getUint32(i * 8, true);
        const hi = dv.getInt32(i * 8 + 4, true);
        out[i] = hi * 4294967296 + lo;
      }
      return out;
    }
    default:
      throw new Error(`tensorAsF32: unsupported dtype ${t.dtype}`);
  }
}

/**
 * Pack a Float32Array back into the wire format for the given dtype.
 * Returns a Uint8Array.
 */
export function f32ToBuffer(dtype, f32) {
  switch (dtype) {
    case "F32": {
      const out = new Uint8Array(f32.length * 4);
      const dv = new DataView(out.buffer);
      for (let i = 0; i < f32.length; i++) dv.setFloat32(i * 4, f32[i], true);
      return out;
    }
    case "F16": {
      return f32ToF16Bytes(f32);
    }
    case "BF16": {
      return f32ToBF16Bytes(f32);
    }
    default:
      throw new Error(`f32ToBuffer: unsupported dtype ${dtype}`);
  }
}

// IEEE 754 half-precision <-> f32 conversion (no SIMD).
function f16BytesToF32(u8) {
  const n = u8.length / 2;
  const out = new Float32Array(n);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  for (let i = 0; i < n; i++) {
    const h = dv.getUint16(i * 2, true);
    out[i] = halfToFloat(h);
  }
  return out;
}

function halfToFloat(h) {
  const sign = (h >> 15) & 0x1;
  const exp  = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) {
    // Zero or subnormal.
    if (frac === 0) return sign ? -0 : 0;
    // Subnormal: value = (-1)^sign * 2^-14 * (frac/1024)
    const f = (frac / 1024) * Math.pow(2, -14);
    return sign ? -f : f;
  }
  if (exp === 31) {
    if (frac === 0) return sign ? -Infinity : Infinity;
    return NaN;
  }
  // Normal.
  const f = (1 + frac / 1024) * Math.pow(2, exp - 15);
  return sign ? -f : f;
}

function f32ToF16Bytes(f32) {
  const out = new Uint8Array(f32.length * 2);
  const dv = new DataView(out.buffer);
  for (let i = 0; i < f32.length; i++) dv.setUint16(i * 2, floatToHalf(f32[i]), true);
  return out;
}

function floatToHalf(val) {
  // Round-to-nearest-even f32 -> f16 conversion.
  if (Number.isNaN(val)) return 0x7e00;
  if (val === Infinity)  return 0x7c00;
  if (val === -Infinity) return 0xfc00;
  if (val === 0) return Object.is(val, -0) ? 0x8000 : 0;
  const buf = new ArrayBuffer(4);
  new DataView(buf).setFloat32(0, val, true);
  const x = new Uint32Array(buf)[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15;
  let frac = x & 0x7fffff;
  if (exp >= 31) return sign | 0x7c00;     // overflow to inf
  if (exp <= 0) {
    if (exp < -10) return sign;            // underflow to zero
    // Subnormal: shift mantissa.
    frac = (frac | 0x800000) >>> (1 - exp);
    // Round to nearest even.
    const r = (frac & 0x1fff) > 0x1000 || ((frac & 0x1fff) === 0x1000 && (frac & 0x2000));
    return sign | ((frac >>> 13) + (r ? 1 : 0));
  }
  // Normal: round to nearest even.
  const r = (frac & 0x1fff) > 0x1000 || ((frac & 0x1fff) === 0x1000 && (frac & 0x2000));
  let halfFrac = (frac >>> 13) + (r ? 1 : 0);
  if (halfFrac === 0x400) { halfFrac = 0; exp += 1; if (exp >= 31) return sign | 0x7c00; }
  return sign | (exp << 10) | halfFrac;
}

// bf16 = top 16 bits of f32 (truncate, no rounding for conversion FROM bf16;
// rounding-to-nearest-even WHEN GOING TO bf16 because that's what PyTorch does).
function bf16BytesToF32(u8) {
  const n = u8.length / 2;
  const out = new Float32Array(n);
  const tmp = new Uint32Array(1);
  const tmpView = new Float32Array(tmp.buffer);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  for (let i = 0; i < n; i++) {
    tmp[0] = dv.getUint16(i * 2, true) << 16;
    out[i] = tmpView[0];
  }
  return out;
}

function f32ToBF16Bytes(f32) {
  // Round-to-nearest-even.
  const out = new Uint8Array(f32.length * 2);
  const dv = new DataView(out.buffer);
  const tmp = new Float32Array(1);
  const tmpU = new Uint32Array(tmp.buffer);
  for (let i = 0; i < f32.length; i++) {
    tmp[0] = f32[i];
    const x = tmpU[0];
    if (Number.isNaN(f32[i])) { dv.setUint16(i * 2, 0x7fc0, true); continue; }
    // Bias: add 0x7fff + (lsb of bf16 retained), then take top 16 bits.
    const bias = 0x7fff + ((x >>> 16) & 1);
    const r = (x + bias) >>> 16;
    dv.setUint16(i * 2, r & 0xffff, true);
  }
  return out;
}

/**
 * Build a safetensors file from an ordered list of tensors.
 * Each entry: { name, dtype, shape, bytes (Uint8Array) }.
 * Returns Uint8Array of the complete file.
 */
export function writeSafetensors(tensors, metadata = {}) {
  // Compute offsets.
  const header = {};
  if (Object.keys(metadata).length) header.__metadata__ = metadata;
  let cursor = 0;
  for (const t of tensors) {
    if (!(t.dtype in DTYPES)) throw new Error(`unknown dtype ${t.dtype}`);
    const expectedBytes = t.shape.reduce((a, b) => a * b, 1) * DTYPES[t.dtype].bytes;
    if (t.bytes.byteLength !== expectedBytes) {
      throw new Error(`tensor ${t.name}: byteLength ${t.bytes.byteLength} != expected ${expectedBytes}`);
    }
    header[t.name] = {
      dtype: t.dtype,
      shape: t.shape,
      data_offsets: [cursor, cursor + expectedBytes],
    };
    cursor += expectedBytes;
  }
  const headerJSON = JSON.stringify(header);
  // Pad header to multiple of 8 with spaces (safetensors recommends this).
  const padded = headerJSON + " ".repeat((8 - (headerJSON.length % 8)) % 8);
  const headerBytes = new TextEncoder().encode(padded);
  const total = 8 + headerBytes.length + cursor;
  const out = new Uint8Array(total);
  // Header length, uint64 LE.
  const dv = new DataView(out.buffer);
  dv.setUint32(0, headerBytes.length, true);
  dv.setUint32(4, 0, true);
  out.set(headerBytes, 8);
  let off = 8 + headerBytes.length;
  for (const t of tensors) {
    out.set(t.bytes, off);
    off += t.bytes.byteLength;
  }
  return out;
}
