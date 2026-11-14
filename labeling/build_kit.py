"""
Builds labeling/label_resumes.html: a single-file page where a person labels each hand-authored resume as a strong,
moderate or weak match for its target role. No network access; answers stay in the browser until downloaded.

Usage: python labeling/build_kit.py <repo root>
"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = [json.loads(line) for line in (root / "data" / "handwritten_eval.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
items = [{"id": r["id"], "role": r["role"], "resume": r["resume"]} for r in rows]
assert len(items) >= 20, "expected the hand-authored set"
blob = json.dumps(items, ensure_ascii=False)
assert '"label"' not in blob, "labels must not be embedded in the page"

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Resume screening: your judgment</title>
<style>
body{font-family:system-ui,-apple-system,"Segoe UI",Arial,sans-serif;background:#0f1020;color:#e8e9f3;margin:0;padding:24px;display:flex;justify-content:center}
.wrap{max-width:760px;width:100%}
h1{font-size:22px;margin:0 0 8px}
p,li{line-height:1.55;color:#c5c7dc}
.card{background:#17182d;border:1px solid #2a2c4a;border-radius:14px;padding:18px;margin:14px 0}
pre{white-space:pre-wrap;word-wrap:break-word;font-family:inherit;font-size:15px;line-height:1.55;margin:8px 0 0;color:#e8e9f3}
input[type=text]{width:100%;padding:10px;border-radius:8px;border:1px solid #3a3d63;background:#0f1020;color:inherit;font-size:16px;box-sizing:border-box}
button{font-size:16px;padding:12px 16px;border-radius:10px;border:1px solid #3a3d63;background:#23254a;color:#e8e9f3;cursor:pointer}
button:hover{background:#2d3060}
button:disabled{opacity:.4;cursor:default}
.row{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
.row button{flex:1;min-width:150px}
.strong{border-color:#10b981}
.moderate{border-color:#f59e0b}
.weak{border-color:#ef4444}
.bar{height:8px;background:#23254a;border-radius:6px;overflow:hidden;margin-top:6px}
.bar div{height:100%;background:#6366f1;width:0}
.muted{color:#8f92b8;font-size:14px}
.err{color:#fca5a5;margin-top:8px}
.hidden{display:none}
</style>
</head>
<body>
<div class="wrap">
 <h1>Resume screening: your judgment</h1>

 <div id="start">
  <div class="card">
   <p>You will see __N__ short, fictional resumes, each written for a target role. For each one, choose how well the candidate fits that role:</p>
   <ul>
    <li><b>Strong</b>: you would shortlist this person straight away.</li>
    <li><b>Moderate</b>: worth a closer look or a screening call.</li>
    <li><b>Weak</b>: you would not move forward for this role.</li>
   </ul>
   <p>There are no trick questions, so go with your judgment. It takes about 15 minutes. Please do not discuss your answers with anyone until everyone has finished. Your answers stay in this browser until you download them at the end.</p>
   <p class="muted">Your ID (initials or a nickname, not your full name)</p>
   <input type="text" id="annotator" maxlength="30" autocomplete="off">
   <div id="err" class="err"></div>
   <div class="row"><button onclick="start()">Start</button></div>
  </div>
 </div>

 <div id="task" class="hidden">
  <div class="muted" id="progress"></div>
  <div class="bar"><div id="bar"></div></div>
  <div class="card">
   <div class="muted">Target role</div>
   <div style="font-size:18px;font-weight:600" id="role"></div>
   <pre id="resume"></pre>
  </div>
  <div class="row">
   <button class="strong" onclick="answer('strong')">Strong fit (1)</button>
   <button class="moderate" onclick="answer('moderate')">Moderate fit (2)</button>
   <button class="weak" onclick="answer('weak')">Weak fit (3)</button>
  </div>
  <div class="row"><button id="back" onclick="back()">Back (left arrow)</button></div>
 </div>

 <div id="done" class="hidden">
  <div class="card">
   <h1>Thank you</h1>
   <p>You labelled <b id="count"></b> resumes. Click the button to save your answers as a small file, then send that file to the person who asked you.</p>
   <div class="row"><button onclick="download()">Download my answers</button></div>
  </div>
 </div>
</div>
<script>
var DATA = __DATA__;
var LABELS = {strong: "strong_match", moderate: "moderate_match", weak: "weak_match"};
var state = {id: "", order: [], answers: {}, pos: 0};
function $(i) { return document.getElementById(i); }
function show(name) {
  ["start", "task", "done"].forEach(function (n) { $(n).className = (n === name) ? "" : "hidden"; });
}
function hash(s) {
  var h = 2166136261;
  for (var i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    var t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function storageKey() { return "resume_labels_" + state.id; }
function save() {
  try { localStorage.setItem(storageKey(), JSON.stringify({answers: state.answers, pos: state.pos})); } catch (e) {}
}
function load() {
  try {
    var raw = localStorage.getItem(storageKey());
    if (raw) { var d = JSON.parse(raw); state.answers = d.answers || {}; state.pos = d.pos || 0; }
  } catch (e) {}
}
function start() {
  var id = $("annotator").value.trim();
  if (!id) { $("err").textContent = "Please enter an ID first."; return; }
  state.id = id.replace(/[^A-Za-z0-9_-]/g, "_").slice(0, 30);
  var rnd = mulberry32(hash(state.id));
  var order = DATA.map(function (_, i) { return i; });
  for (var i = order.length - 1; i > 0; i--) {
    var j = Math.floor(rnd() * (i + 1));
    var t = order[i]; order[i] = order[j]; order[j] = t;
  }
  state.order = order; state.answers = {}; state.pos = 0;
  load();
  render();
}
function render() {
  if (state.pos >= state.order.length) { finish(); return; }
  show("task");
  var item = DATA[state.order[state.pos]];
  $("progress").textContent = (state.pos + 1) + " of " + state.order.length;
  $("bar").style.width = (100 * state.pos / state.order.length) + "%";
  $("role").textContent = item.role;
  $("resume").textContent = item.resume;
  $("back").disabled = state.pos === 0;
}
function answer(kind) {
  if (state.pos >= state.order.length) return;
  var item = DATA[state.order[state.pos]];
  state.answers[item.id] = LABELS[kind];
  state.pos += 1;
  save();
  render();
}
function back() { if (state.pos > 0) { state.pos -= 1; save(); render(); } }
function finish() {
  show("done"); save();
  $("count").textContent = Object.keys(state.answers).length + " of " + DATA.length;
}
function download() {
  var out = {annotator: state.id, date: new Date().toISOString().slice(0, 10), n_items: DATA.length, labels: state.answers};
  var blob = new Blob([JSON.stringify(out, null, 2)], {type: "application/json"});
  var a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "labels_" + state.id + ".json";
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
}
document.addEventListener("keydown", function (e) {
  if ($("task").className !== "") return;
  if (e.key === "1") answer("strong");
  else if (e.key === "2") answer("moderate");
  else if (e.key === "3") answer("weak");
  else if (e.key === "ArrowLeft") back();
});
</script>
</body>
</html>
"""
page = TEMPLATE.replace("__N__", str(len(items))).replace("__DATA__", blob.replace("</", "<\\/"))
out = root / "labeling" / "label_resumes.html"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(page, encoding="utf-8", newline="\n")
print("Wrote %s (%d resumes, %d KB)" % (out, len(items), out.stat().st_size // 1024))