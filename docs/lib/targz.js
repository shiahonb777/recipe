// Browser tar.gz reader. Uses native DecompressionStream("gzip"); USTAR header parser.
//
// `extractTarGz(arrayBuffer)` returns an array of { name, bytes } where `bytes`
// is a Uint8Array view of the file content. Only regular files are returned;
// directory entries and other special types are skipped.

export async function extractTarGz(buf) {
  const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  // Decompress gzip natively.
  const stream = new Response(u8).body.pipeThrough(new DecompressionStream("gzip"));
  const reader = stream.getReader();
  const chunks = [];
  let total = 0;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    chunks.push(value);
    total += value.byteLength;
  }
  const tar = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) { tar.set(c, off); off += c.byteLength; }
  return parseTar(tar);
}

function parseTar(tar) {
  // USTAR format: 512-byte blocks, header per file, then padded file content.
  const out = [];
  let pos = 0;
  while (pos + 512 <= tar.length) {
    // Empty header (all zeros) means end-of-archive.
    let allZero = true;
    for (let i = 0; i < 512; i++) {
      if (tar[pos + i] !== 0) { allZero = false; break; }
    }
    if (allZero) break;

    // Parse header.
    const name = readString(tar, pos, 100);
    const sizeOctal = readString(tar, pos + 124, 12);
    const typeflag = String.fromCharCode(tar[pos + 156] || 0x30);
    const prefix = readString(tar, pos + 345, 155);
    const size = parseInt(sizeOctal.replace(/\0/g, "").trim() || "0", 8);

    pos += 512;

    if (typeflag === "0" || typeflag === "" || typeflag === "\0") {
      // Regular file.
      const fullName = prefix ? prefix + "/" + name : name;
      const bytes = tar.subarray(pos, pos + size);
      out.push({ name: fullName, bytes });
    }
    // Else directory ("5"), symlink ("2"), etc. — skip.

    // Advance to next 512-byte boundary.
    const padded = Math.ceil(size / 512) * 512;
    pos += padded;
  }
  return out;
}

function readString(tar, off, max) {
  let end = off;
  while (end < off + max && tar[end] !== 0) end++;
  return new TextDecoder("ascii").decode(tar.subarray(off, end));
}
