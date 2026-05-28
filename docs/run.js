// Driver for run.html. Wires the materialize pipeline to the staged UI.
import { materializeRecipe } from "./lib/materialize.js?v=2";

const $ = (id) => document.getElementById(id);

const log = (msg, kind = "info") => {
  const pre = $("log");
  pre.hidden = false;
  const span = document.createElement("span");
  if (kind !== "info") span.className = kind;
  span.textContent = msg + "\n";
  pre.appendChild(span);
  pre.scrollTop = pre.scrollHeight;
};

function fmtBytes(n) {
  if (n < 1024) return n + " B";
  if (n < 1024 ** 2) return (n / 1024).toFixed(1) + " KB";
  if (n < 1024 ** 3) return (n / 1024 ** 2).toFixed(1) + " MB";
  return (n / 1024 ** 3).toFixed(2) + " GB";
}

function fmtSecs(ms) {
  if (ms < 1000) return ms.toFixed(0) + " ms";
  if (ms < 60_000) return (ms / 1000).toFixed(1) + " s";
  const m = Math.floor(ms / 60_000);
  const s = Math.floor((ms % 60_000) / 1000);
  return `${m}m ${s}s`;
}

let mergedBlob = null;
let recipeName = null;

const stages = ["recipe", "bundle", "base", "merge", "write"];
const stageStartTimes = {};
let activeStage = null;

function setStage(name, state, detailText) {
  if (activeStage && activeStage !== name) {
    // Mark previous as done if not already.
    const prev = $(`st-${activeStage}`);
    if (prev && !prev.classList.contains("failed") && !prev.classList.contains("done")) {
      prev.classList.add("done");
      const elapsed = Date.now() - (stageStartTimes[activeStage] || Date.now());
      const tEl = $(`t-${activeStage}`);
      if (tEl) tEl.textContent = fmtSecs(elapsed);
    }
  }
  if (state === "active") {
    activeStage = name;
    stageStartTimes[name] = Date.now();
    $(`st-${name}`)?.classList.add("active");
  } else if (state === "done") {
    $(`st-${name}`)?.classList.remove("active");
    $(`st-${name}`)?.classList.add("done");
    const elapsed = Date.now() - (stageStartTimes[name] || Date.now());
    const tEl = $(`t-${name}`);
    if (tEl) tEl.textContent = fmtSecs(elapsed);
  } else if (state === "failed") {
    $(`st-${name}`)?.classList.remove("active");
    $(`st-${name}`)?.classList.add("failed");
  }
  if (detailText !== undefined) {
    const dEl = $(`d-${name}`);
    if (dEl) dEl.textContent = detailText;
  }
}

function setProgress(name, frac) {
  const el = $(`p-${name}`);
  if (el) el.style.width = (Math.max(0, Math.min(1, frac)) * 100).toFixed(1) + "%";
}

async function run(refStr) {
  $("error-box").hidden = true;
  $("verify").hidden = true;
  $("stages").hidden = false;
  $("log").hidden = false;
  $("log").innerHTML = "";
  for (const s of stages) {
    const el = $(`st-${s}`);
    el.classList.remove("done", "active", "failed");
    $(`d-${s}`).textContent = "—";
    $(`t-${s}`).textContent = "";
    setProgress(s, 0);
  }
  $("run-btn").disabled = true;
  mergedBlob = null;

  let repo, tag;
  if (refStr.includes("@")) [repo, tag] = refStr.split("@", 2);
  else { repo = refStr; tag = "latest"; }
  if (!repo.includes("/")) {
    showError("Expected user/repo or user/repo@tag");
    $("run-btn").disabled = false;
    return;
  }

  setStage("recipe", "active", `Looking up ${repo}@${tag}…`);

  let bundleStarted = false;
  let baseStarted = false;

  try {
    const result = await materializeRecipe({
      repo, tag,
      onProgress: (e) => {
        switch (e.stage) {
          case "log":
            log(e.msg, e.kind);
            // Map log markers to stages. The materialize pipeline emits
            // these messages in roughly this order; we use them to advance
            // the staged UI.
            if (e.msg.startsWith("try repo tree") || e.msg.startsWith("resolve release") || e.msg.startsWith("falling back")) {
              setStage("recipe", "active");
            } else if (e.msg.startsWith("download recipe bundle") || e.msg.startsWith("download adapter artifact")) {
              setStage("recipe", "done");
              setStage("bundle", "active");
              bundleStarted = true;
            } else if (e.msg.startsWith("extract bundle")) {
              setStage("bundle", "done");
            } else if (e.msg.startsWith("recipe loaded") || e.msg.startsWith("recipe verified")) {
              setStage("recipe", "done");
              setStage("bundle", "done");
              const stripped = e.msg.replace(/^recipe (loaded|verified):? ?/, "");
              if (stripped) $("d-bundle").textContent = stripped;
            } else if (e.msg.startsWith("download base model")) {
              setStage("base", "active");
              baseStarted = true;
            } else if (e.msg.startsWith("parse base safetensors")) {
              setStage("base", "done");
            } else if (e.msg.startsWith("matched ")) {
              setStage("merge", "active");
              $("d-merge").textContent = e.msg;
            } else if (e.msg.startsWith("assemble merged safetensors")) {
              setStage("merge", "done");
              setStage("write", "active");
            } else if (e.msg.startsWith("compute sha256")) {
              $("d-write").textContent = "computing SHA-256…";
            }
            break;
          case "fetch": {
            const total = e.total || 0;
            const loaded = e.loaded || 0;
            const sizeStr = total > 0
              ? `${fmtBytes(loaded)} / ${fmtBytes(total)}`
              : `${fmtBytes(loaded)}`;
            if (e.label === "recipe bundle") {
              if (!bundleStarted) { setStage("bundle", "active"); bundleStarted = true; }
              $("d-bundle").textContent = sizeStr;
              if (total > 0) setProgress("bundle", loaded / total);
            } else if (e.label === "base model") {
              if (!baseStarted) { setStage("base", "active"); baseStarted = true; }
              $("d-base").textContent = sizeStr;
              if (total > 0) setProgress("base", loaded / total);
            }
            break;
          }
          case "merge": {
            const total = e.total || 1;
            const done = e.done || 0;
            $("d-merge").textContent = `${done + 1} / ${total} — ${e.baseKey}`;
            setProgress("merge", (done + 1) / total);
            break;
          }
          default: break;
        }
      },
    });

    setStage("write", "done", `${result.modified} tensors merged · ${fmtBytes(result.mergedSafetensors.byteLength)}`);
    setProgress("merge", 1);
    setProgress("base", 1);
    setProgress("bundle", 1);

    // Render verify card.
    recipeName = result.recipeName;
    $("v-name").textContent = result.recipeName;
    $("v-base").textContent = result.baseRef;
    $("v-modified").textContent = `${result.modified} tensors`;
    $("v-size").textContent = fmtBytes(result.mergedSafetensors.byteLength);
    $("v-hash").textContent = result.merged_sha256;
    $("verify").hidden = false;

    mergedBlob = new Blob([result.mergedSafetensors], { type: "application/octet-stream" });
    log(`done. ${result.modified} tensors merged. SHA-256: ${result.merged_sha256.slice(0, 16)}…`);
  } catch (e) {
    if (activeStage) setStage(activeStage, "failed");
    showError(e.message || String(e));
    log("ERROR: " + (e.message || String(e)), "err");
  } finally {
    $("run-btn").disabled = false;
  }
}

function showError(msg) {
  const box = $("error-box");
  box.hidden = false;
  box.className = "error";
  box.textContent = msg;
}

// Wire up.
$("form").addEventListener("submit", (e) => {
  e.preventDefault();
  const ref = $("ref").value.trim();
  if (ref) run(ref);
});
document.querySelectorAll(".examples code").forEach((c) => {
  c.addEventListener("click", () => {
    const ref = c.dataset.ref;
    $("ref").value = ref;
    run(ref);
  });
});
$("dl-btn").addEventListener("click", () => {
  if (!mergedBlob) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(mergedBlob);
  a.download = (recipeName || "merged") + "-model.safetensors";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

// Auto-run from URL param.
const urlRef = new URLSearchParams(location.search).get("ref");
if (urlRef) {
  $("ref").value = urlRef;
  run(urlRef);
}
