// Quick standalone test for the parseTOML function.
// We extract it from index.html and feed real recipe.toml content.

const fs = require('fs');
const html = fs.readFileSync(__dirname + '/index.html', 'utf-8');
// Pull out the parseTOML function definition and its closing brace.
const start = html.indexOf('function parseTOML(text) {');
if (start < 0) throw new Error('parseTOML not found');
let depth = 0, end = -1;
for (let i = start; i < html.length; i++) {
  if (html[i] === '{') depth++;
  else if (html[i] === '}') { depth--; if (depth === 0) { end = i + 1; break; } }
}
const fn = html.slice(start, end);
eval(fn);

const sample1 = `
[recipe]
version = "0.1"
name = "gpt2-alpaca"

[base]
ref = "gpt2"

[training]
method = "lora"

[[adapters]]
type = "lora"
artifact = "sha256:abc"
target_modules = [
    "c_attn",
]
rank = 8
alpha = 16.0
fan_in_fan_out = true
bias = "none"
`;
const r1 = parseTOML(sample1);
console.log('=== sample1 ===');
console.log(JSON.stringify(r1, null, 2));

// Assertions.
const assert = require('assert');
assert.strictEqual(r1.recipe.name, 'gpt2-alpaca');
assert.strictEqual(r1.base.ref, 'gpt2');
assert.strictEqual(r1.adapters.length, 1);
const a = r1.adapters[0];
assert.strictEqual(a.type, 'lora');
assert.strictEqual(a.rank, 8);
assert.strictEqual(a.alpha, 16.0);
assert.deepStrictEqual(a.target_modules, ['c_attn']);
assert.strictEqual(a.fan_in_fan_out, true);
console.log('sample1 OK');

// Try the real Qwen recipe (multiple target_modules):
const sample2 = `
[recipe]
version = "0.1"
name = "qwen2.5-1.5b-oasst-guanaco"

[base]
ref = "Qwen/Qwen2.5-1.5B"

[training]
method = "lora"

[[adapters]]
type = "lora"
artifact = "sha256:c37f"
target_modules = [
    "v_proj",
    "o_proj",
    "down_proj",
    "up_proj",
    "gate_proj",
    "k_proj",
    "q_proj",
]
rank = 16
alpha = 16.0
bias = "none"
lora_dropout = 0.05
`;
const r2 = parseTOML(sample2);
console.log('=== sample2 ===');
console.log(JSON.stringify(r2, null, 2));
assert.deepStrictEqual(
  r2.adapters[0].target_modules,
  ['v_proj', 'o_proj', 'down_proj', 'up_proj', 'gate_proj', 'k_proj', 'q_proj']
);
assert.strictEqual(r2.adapters[0].lora_dropout, 0.05);
console.log('sample2 OK');
