// Browser-side materialization: download base + recipe, apply LoRA, write merged safetensors.
//
// Mirrors mlrecipe.materialize.materialize() at the byte level.
//
// Usage:
//   const result = await materializeRecipe({
//     repo: "shiahonb777/gpt2-alpaca-recipe",
//     tag:  "v1",
//     onProgress: (step, info) => { ... },
//   });
//   // result.mergedSafetensors  : Uint8Array, ready for download
//   // result.merged_dtype       : "F32" | "F16" | "BF16"
//   // result.modified           : count of merged tensors
//   // result.base_sha256        : sha256 hex of input base bytes
//   // result.merged_sha256      : sha256 hex of output bytes

import { extractTarGz } from "./targz.js?v=2";
import { readSafetensors, tensorAsF32, f32ToBuffer, writeSafetensors, DTYPES } from "./safetensors.js?v=2";
import { matchLoraTargets, applyLoRA } from "./lora.js?v=2";

// Minimal TOML subset needed for recipe.toml — the main app's parseTOML
// is fine but we want this lib to be self-contained, so we duplicate
// a stripped version here.
function parseTOML(text) {
  const lines = text.split(/\r?\n/);
  const folded = [];
  let buf = null, depth = 0;
  for (const raw of lines) {
    if (buf !== null) {
      buf += " " + raw;
      for (const c of raw) { if (c === "[") depth++; else if (c === "]") depth--; }
      if (depth <= 0) { folded.push(buf); buf = null; depth = 0; }
      continue;
    }
    const eq = raw.indexOf("=");
    const lbr = raw.indexOf("[");
    if (eq >= 0 && lbr > eq) {
      let d = 0, opened = false;
      for (let i = lbr; i < raw.length; i++) {
        if (raw[i] === "[") { d++; opened = true; }
        else if (raw[i] === "]") d--;
      }
      if (opened && d > 0) { buf = raw; depth = d; continue; }
    }
    folded.push(raw);
  }
  if (buf !== null) folded.push(buf);
  const root = {};
  let cur = root;
  function parseValue(s) {
    s = s.trim();
    if (s.startsWith('"') && s.endsWith('"')) return JSON.parse(s);
    if (s === "true") return true;
    if (s === "false") return false;
    if (s.startsWith("[") && s.endsWith("]")) {
      const inner = s.slice(1, -1).trim();
      if (!inner) return [];
      const items = []; let d = 0, q = false, start = 0;
      for (let i = 0; i <= inner.length; i++) {
        const c = inner[i];
        if (c === '"' && inner[i-1] !== "\\") q = !q;
        if (!q && c === "[") d++;
        if (!q && c === "]") d--;
        if (!q && d === 0 && (c === "," || i === inner.length)) {
          const tok = inner.slice(start, i).trim();
          if (tok) items.push(parseValue(tok));
          start = i + 1;
        }
      }
      return items;
    }
    if (/^-?\d+$/.test(s)) return parseInt(s, 10);
    if (/^-?\d+\.\d+(e-?\d+)?$/i.test(s)) return parseFloat(s);
    return s;
  }
  function tableAt(obj, path) {
    for (const p of path) { if (!(p in obj)) obj[p] = {}; obj = obj[p]; }
    return obj;
  }
  for (const raw of folded) {
    const line = raw.split("#")[0].trim();
    if (!line) continue;
    if (line.startsWith("[[") && line.endsWith("]]")) {
      const path = line.slice(2, -2).split(".").map((p) => p.trim());
      const parent = tableAt(root, path.slice(0, -1));
      const last = path[path.length - 1];
      if (!Array.isArray(parent[last])) parent[last] = [];
      const entry = {};
      parent[last].push(entry);
      cur = entry;
      continue;
    }
    if (line.startsWith("[") && line.endsWith("]") && !line.includes("=")) {
      const path = line.slice(1, -1).split(".").map((p) => p.trim());
      cur = tableAt(root, path);
      continue;
    }
    const eq = line.indexOf("=");
    if (eq < 0) continue;
    cur[line.slice(0, eq).trim()] = parseValue(line.slice(eq + 1));
  }
  return root;
}

// Hugging Face model file URL. We hard-code the expected single-shard layout
// for the small bases we support; multi-shard would need an index parser.
const HF_RESOLVE = (ref, revision, file) =>
  `https://huggingface.co/${ref}/resolve/${revision || "main"}/${file}`;

// Models we know fit in browser memory and are sharded as a single file.
const KNOWN_BASES = {
  "gpt2": { file: "model.safetensors", approxBytes: 500e6 },
  "gpt2-medium": { file: "model.safetensors", approxBytes: 1.5e9 },
  "distilgpt2": { file: "model.safetensors", approxBytes: 330e6 },
  "Qwen/Qwen2.5-0.5B": { file: "model.safetensors", approxBytes: 1e9 },
  "Qwen/Qwen2.5-0.5B-Instruct": { file: "model.safetensors", approxBytes: 1e9 },
};

async function fetchWithProgress(url, label, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${label}: HTTP ${res.status}`);
  const total = parseInt(res.headers.get("content-length") || "0", 10);
  if (!total || !res.body) {
    // Server didn't give us content-length; fall back to single-shot.
    onProgress?.({ label, loaded: 0, total: 0 });
    const buf = await res.arrayBuffer();
    onProgress?.({ label, loaded: buf.byteLength, total: buf.byteLength });
    return new Uint8Array(buf);
  }
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    onProgress?.({ label, loaded: received, total });
  }
  const out = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.byteLength; }
  return out;
}

// CORS bypass for github.com/.../releases/download/... — that origin
// does not send Access-Control-Allow-Origin and there is no GitHub API
// way to opt in. We route the request through a free CORS proxy. If
// the proxy is ever down, the page still works for any recipe whose
// artifacts are committed to the repo tree (which we fetch directly
// from raw.githubusercontent.com, which DOES send open CORS headers).
const CORS_PROXY = "https://corsproxy.io/?";
function needsCorsProxy(url) {
  const u = new URL(url);
  return u.hostname === "github.com" && u.pathname.includes("/releases/download/");
}
function viaCorsProxy(url) {
  return CORS_PROXY + encodeURIComponent(url);
}
async function fetchSmart(url, label, onProgress) {
  // If the URL is a known CORS-blocked origin, go straight to the proxy.
  // Otherwise try direct first; on a TypeError (the only way browser CORS
  // surfaces in user code), retry through the proxy.
  if (needsCorsProxy(url)) {
    onProgress?.({ stage: "log", msg: `routing release download through corsproxy.io (GitHub release CORS workaround)`, kind: "info" });
    return fetchWithProgress(viaCorsProxy(url), label, onProgress);
  }
  try {
    return await fetchWithProgress(url, label, onProgress);
  } catch (e) {
    if (e instanceof TypeError) {
      onProgress?.({ stage: "log", msg: `direct fetch blocked, retrying through corsproxy.io: ${url}`, kind: "warn" });
      return await fetchWithProgress(viaCorsProxy(url), label, onProgress);
    }
    throw e;
  }
}

async function sha256Hex(bytes) {
  const buf = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * The whole thing.
 *
 * Strategy:
 *   1. Try to fetch the recipe directly from the repo tree at
 *      raw.githubusercontent.com/<repo>/<branch>/.recipe/. This is
 *      the path that ALWAYS sends open CORS headers, and is what
 *      'mlrecipe push' commits since we added the
 *      'commit recipe to repo tree' patch.
 *   2. If the repo tree doesn't have a .recipe/ directory (older
 *      releases, before that patch), fall back to the release tar.gz.
 *      Github.com release downloads do NOT send CORS headers, so this
 *      path needs the user to enable a CORS proxy or run from a CLI.
 */
export async function materializeRecipe({ repo, tag, onProgress = () => {} }) {
  const log = (msg, kind = "info") => onProgress({ stage: "log", msg, kind });

  // 1. Try the repo tree first. We don't know which branch the recipe
  // lives on a priori; try the common ones in order, plus the requested
  // tag (which could be a branch or a tag — Git supports both as refs).
  const branchCandidates = [tag === "latest" ? null : tag, "main", "master"]
    .filter(Boolean);
  let recipeToml = null;
  let adapterBytes = null;
  let usedBranch = null;
  let recipe = null;

  for (const branch of branchCandidates) {
    log(`try repo tree at ${branch}`);
    const tomlURL = `https://raw.githubusercontent.com/${repo}/${branch}/.recipe/recipe.toml`;
    try {
      log(`  GET ${tomlURL}`);
      const tomlRes = await fetch(tomlURL);
      log(`  -> HTTP ${tomlRes.status}`);
      if (!tomlRes.ok) continue;
      const tomlText = await tomlRes.text();
      const candidate = parseTOML(tomlText);
      if (!candidate.recipe || !candidate.base) {
        log(`  toml present but no [recipe]/[base] section, skipping`, "warn");
        continue;
      }

      // Resolve adapter from .recipe/artifacts/<aa>/<full-hash>.
      const adapter0 = (candidate.adapters || [])[0];
      if (!adapter0?.artifact) continue;
      const hash = adapter0.artifact.replace(/^sha256:/, "");
      if (!/^[0-9a-f]{64}$/.test(hash)) continue;
      const artifactURL = `https://raw.githubusercontent.com/${repo}/${branch}/.recipe/artifacts/${hash.slice(0, 2)}/${hash}`;
      log(`download adapter artifact (${hash.slice(0, 16)}…)`);
      const artBytes = await fetchSmart(
        artifactURL,
        "recipe artifact",
        (p) => onProgress({ stage: "fetch", ...p, label: "recipe bundle" }),
      );
      // Verify the hash to make sure we got the right file.
      const gotHash = await sha256Hex(artBytes);
      if (gotHash !== hash) {
        throw new Error(`adapter hash mismatch: expected ${hash}, got ${gotHash}`);
      }
      recipeToml = tomlText;
      adapterBytes = artBytes;
      recipe = candidate;
      usedBranch = branch;
      log(`recipe loaded from ${repo}@${branch} (.recipe tree)`);
      break;
    } catch (e) {
      log(`  branch ${branch} failed: ${e.message}`, "warn");
    }
  }

  // 2. Fall back to the release tar.gz if the repo tree didn't have it.
  if (!recipe) {
    log(`falling back to GitHub release ${repo}@${tag}`);
    const releaseURL = tag === "latest"
      ? `https://api.github.com/repos/${repo}/releases/latest`
      : `https://api.github.com/repos/${repo}/releases/tags/${tag}`;
    const relRes = await fetch(releaseURL, { headers: { Accept: "application/vnd.github+json" } });
    if (!relRes.ok) throw new Error(`fetch release: HTTP ${relRes.status}`);
    const release = await relRes.json();
    const bundleAsset = release.assets.find((a) => a.name.endsWith(".tar.gz"));
    if (!bundleAsset) throw new Error("no .tar.gz asset on release; was it pushed by mlrecipe?");

    log(`download recipe bundle (${(bundleAsset.size / 1e6).toFixed(2)} MB)`);
    const bundleBytes = await fetchSmart(
      bundleAsset.browser_download_url,
      "recipe bundle",
      (p) => onProgress({ stage: "fetch", ...p }),
    );
    log(`extract bundle`);
    const entries = await extractTarGz(bundleBytes);
    const artifactsByHash = new Map();
    for (const e of entries) {
      if (e.name.endsWith("recipe.toml")) {
        recipeToml = new TextDecoder().decode(e.bytes);
      } else if (e.name.includes("/artifacts/") && e.bytes.byteLength > 0) {
        const fname = e.name.split("/").pop();
        if (/^[0-9a-f]{64}$/.test(fname)) artifactsByHash.set(fname, e.bytes);
      }
    }
    if (!recipeToml) throw new Error("recipe.toml missing from bundle");
    recipe = parseTOML(recipeToml);
    const adapter0 = (recipe.adapters || [])[0];
    if (!adapter0?.artifact) throw new Error("adapter has no artifact hash");
    const hash = adapter0.artifact.replace(/^sha256:/, "");
    adapterBytes = artifactsByHash.get(hash);
    if (!adapterBytes) {
      throw new Error(`bundle is missing artifact ${hash}; corrupted bundle`);
    }
  }

  // Validate.
  const baseRef = recipe.base?.ref;
  if (!baseRef) throw new Error("recipe.base.ref missing");
  const adapters = recipe.adapters || [];
  if (adapters.length === 0) throw new Error("recipe has no adapters");
  if (adapters.length > 1) {
    log(`recipe stacks ${adapters.length} adapters; this browser path only supports one`, "warn");
  }
  const adapter = adapters[0];
  log(`recipe verified: ${recipe.recipe?.name || "?"} (base: ${baseRef})`);

  // 3. Fetch the base model from Hugging Face.
  const baseInfo = KNOWN_BASES[baseRef];
  if (!baseInfo) {
    throw new Error(
      `base model ${baseRef} is not in the supported list for browser-side materialize. ` +
      `Use the CLI for arbitrary bases: mlrecipe clone ${repo}@${tag} && mlrecipe materialize ./merged`
    );
  }
  const baseURL = HF_RESOLVE(baseRef, recipe.base?.revision, baseInfo.file);
  log(`download base model ${baseRef} (~${(baseInfo.approxBytes / 1e6).toFixed(0)} MB)`);
  const baseBytes = await fetchSmart(
    baseURL, "base model",
    (p) => onProgress({ stage: "fetch", ...p }),
  );

  // 4. Parse safetensors.
  log(`parse base safetensors`);
  const baseFile = readSafetensors(baseBytes);
  log(`parse adapter safetensors`);
  const adapterFile = readSafetensors(adapterBytes);

  // 5. Match LoRA -> base keys.
  const targetModules = adapter.target_modules || [];
  const fanInFanOut = !!(adapter.fan_in_fan_out || adapter.extra?.fan_in_fan_out);
  const rank = adapter.rank;
  const alpha = adapter.alpha != null ? adapter.alpha : rank;
  const baseKeys = Object.keys(baseFile.tensors);
  const loraKeys = Object.keys(adapterFile.tensors);
  const pairs = matchLoraTargets(baseKeys, loraKeys, targetModules);
  const numPairs = Object.keys(pairs).length;
  log(`matched ${numPairs} LoRA targets`);
  if (numPairs === 0) {
    throw new Error(`no LoRA targets matched. target_modules=${JSON.stringify(targetModules)}`);
  }

  // 6. Apply LoRA per matched key. Hold merged tensor bytes in a map until write.
  // To keep memory bounded we modify the BASE buffer in place: for each merged
  // weight, we compute the new bytes and overwrite the corresponding slice.
  // The output is a complete safetensors file with the base header preserved
  // and only the merged tensor regions changed.
  const mergedBytesByKey = new Map(); // baseKey -> Uint8Array (replacement payload)
  let modified = 0;
  for (const baseKey of Object.keys(pairs)) {
    const { aKey, bKey } = pairs[baseKey];
    const baseT = baseFile.tensors[baseKey];
    const aT = adapterFile.tensors[aKey];
    const bT = adapterFile.tensors[bKey];
    onProgress({ stage: "merge", baseKey, done: modified, total: numPairs });
    const baseF32 = tensorAsF32(baseT);
    const aF32 = tensorAsF32(aT);
    const bF32 = tensorAsF32(bT);
    const merged = await applyLoRA({
      baseF32, baseShape: baseT.shape,
      aF32,    aShape: aT.shape,
      bF32,    bShape: bT.shape,
      alpha, rank, fanInFanOut,
    });
    mergedBytesByKey.set(baseKey, f32ToBuffer(baseT.dtype, merged));
    modified += 1;
  }

  // 7. Stitch a new safetensors file: same header (offsets unchanged), but
  // tensor regions for merged keys come from mergedBytesByKey.
  log(`assemble merged safetensors`);
  // Compose tensor list in the original order (so the merged file is
  // byte-equivalent to PEFT's output for unmodified tensors).
  const tensorList = [];
  for (const name of baseKeys) {
    const t = baseFile.tensors[name];
    const replacement = mergedBytesByKey.get(name);
    tensorList.push({
      name,
      dtype: t.dtype,
      shape: t.shape,
      bytes: replacement || t.view,
    });
  }
  const mergedSafetensors = writeSafetensors(tensorList, baseFile.metadata);

  // 8. Hashes for verification.
  log(`compute sha256 (input + output)`);
  const baseHash = await sha256Hex(baseBytes);
  const mergedHash = await sha256Hex(mergedSafetensors);

  return {
    mergedSafetensors,
    base_sha256: baseHash,
    merged_sha256: mergedHash,
    modified,
    recipeName: recipe.recipe?.name || "merged",
    baseRef,
  };
}
