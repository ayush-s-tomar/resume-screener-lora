"""
Builds pilot/survey.html (a single offline page) and pilot/INFORMATION_SHEET.md from pilot/questions.json, so the questions live in one place.

Usage:
    python pilot/build_survey.py [repo root]
"""
import html
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
pilot = root / "pilot"
data = json.loads((pilot / "questions.json").read_text(encoding="utf-8-sig"))

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
body{font-family:system-ui,-apple-system,'Segoe UI',Arial,sans-serif;background:#0f1020;color:#e8e9f3;margin:0;padding:24px;display:flex;justify-content:center}
.wrap{max-width:760px;width:100%}
h1{font-size:22px;margin:0 0 12px}
p.intro{line-height:1.55;color:#c5c7dc}
fieldset.q{border:1px solid #2a2c4a;border-radius:12px;margin:14px 0;padding:12px 16px;background:#17182d}
legend{padding:0 6px;font-weight:600}
.help{color:#8f92b8;font-size:13px;margin:4px 0}
label.opt{display:block;margin:6px 0;cursor:pointer}
.scale{display:flex;gap:18px;flex-wrap:wrap;margin:6px 0}
.scale label.opt{display:inline-block}
label.check{display:block;background:#17182d;border:1px solid #3a3d63;border-radius:12px;padding:12px 16px;margin:16px 0;line-height:1.5}
input[type=number],textarea{width:100%;box-sizing:border-box;background:#0f1020;color:inherit;border:1px solid #3a3d63;border-radius:8px;padding:10px;font-size:15px;font-family:inherit}
input[type=number]{max-width:160px}
button{font-size:16px;padding:12px 18px;border-radius:10px;border:1px solid #3a3d63;background:#23254a;color:#e8e9f3;cursor:pointer;margin:6px 8px 6px 0}
button:hover{background:#2d3060}
.msg{color:#fca5a5;margin:10px 0}
.out{margin:16px 0}
pre{white-space:pre-wrap;word-wrap:break-word;background:#17182d;border:1px solid #2a2c4a;border-radius:10px;padding:12px;font-size:13px}
</style>
</head>
<body>
<div class="wrap" id="app"></div>
<script>
var DATA = __DATA__;
var answers = {};
function h(tag, cls, text) {
  var e = document.createElement(tag);
  if (cls) { e.className = cls; }
  if (text !== undefined) { e.textContent = text; }
  return e;
}
function radio(q, value, label) {
  var l = h('label', 'opt');
  var i = h('input');
  i.type = 'radio';
  i.name = q.id;
  i.value = String(value);
  i.onchange = function () { answers[q.id] = value; };
  l.appendChild(i);
  l.appendChild(document.createTextNode(' ' + label));
  return l;
}
function renderQuestion(q) {
  var box = h('fieldset', 'q');
  box.id = 'q_' + q.id;
  box.appendChild(h('legend', '', q.label + (q.required ? ' *' : '')));
  if (q.help) { box.appendChild(h('div', 'help', q.help)); }
  if (q.type === 'choice') {
    q.options.forEach(function (o) { box.appendChild(radio(q, o[0], o[1])); });
  } else if (q.type === 'scale') {
    var row = h('div', 'scale');
    for (var v = q.min; v <= q.max; v++) { row.appendChild(radio(q, v, String(v))); }
    box.appendChild(row);
    box.appendChild(h('div', 'help', q.min + ' = ' + q.low + ', ' + q.max + ' = ' + q.high));
  } else if (q.type === 'number') {
    var n = h('input');
    n.type = 'number';
    n.min = q.min;
    n.max = q.max;
    n.oninput = function () { if (n.value === '') { delete answers[q.id]; } else { answers[q.id] = Number(n.value); } };
    box.appendChild(n);
  } else {
    var t = h('textarea');
    t.maxLength = q.max || 300;
    t.rows = 3;
    t.oninput = function () { var s = t.value.trim(); if (s) { answers[q.id] = s; } else { delete answers[q.id]; } };
    box.appendChild(t);
  }
  return box;
}
function render() {
  var app = document.getElementById('app');
  app.appendChild(h('h1', '', DATA.title));
  DATA.intro.forEach(function (p) { app.appendChild(h('p', 'intro', p)); });
  var consent = h('label', 'check');
  var cb = h('input');
  cb.type = 'checkbox';
  cb.id = 'consent';
  consent.appendChild(cb);
  consent.appendChild(document.createTextNode(' ' + DATA.consent));
  app.appendChild(consent);
  DATA.questions.forEach(function (q) { app.appendChild(renderQuestion(q)); });
  var btn = h('button', '', 'Finish and get my answers file');
  btn.type = 'button';
  btn.onclick = finish;
  app.appendChild(btn);
  var msg = h('div', 'msg');
  msg.id = 'msg';
  app.appendChild(msg);
  var out = h('div', 'out');
  out.id = 'out';
  app.appendChild(out);
}
function finish() {
  var missing = [];
  if (!document.getElementById('consent').checked) { missing.push('the consent box at the top'); }
  DATA.questions.forEach(function (q) {
    var v = answers[q.id];
    if (q.required && v === undefined) { missing.push(q.label); }
    if (q.type === 'number' && v !== undefined && (isNaN(v) || v < q.min || v > q.max)) { missing.push(q.label + ' (a number from ' + q.min + ' to ' + q.max + ')'); }
  });
  var msg = document.getElementById('msg');
  if (missing.length) { msg.textContent = 'Please complete: ' + missing.join(' | '); return; }
  msg.textContent = '';
  var id = '';
  try {
    var bytes = new Uint8Array(4);
    crypto.getRandomValues(bytes);
    for (var k = 0; k < bytes.length; k++) { id += ('0' + bytes[k].toString(16)).slice(-2); }
  } catch (e) {
    id = String(Math.random()).slice(2, 10);
  }
  var data = { version: DATA.version, id: id, created: new Date().toISOString().slice(0, 10), consent: true, answers: answers };
  var json = JSON.stringify(data, null, 2);
  var out = document.getElementById('out');
  out.textContent = '';
  out.appendChild(h('p', 'intro', 'Thank you. Click the button to save your answers as a small file, then send that file to me. If the download does not work, copy the text below into a file named pilot_' + id + '.json instead.'));
  var dl = h('button', '', 'Download my answers');
  dl.type = 'button';
  dl.onclick = function () {
    var blob = new Blob([json], { type: 'application/json' });
    var link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'pilot_' + id + '.json';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  out.appendChild(dl);
  out.appendChild(h('pre', '', json));
}
render();
</script>
</body>
</html>
"""
blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
page = TEMPLATE.replace("__TITLE__", html.escape(data["title"])).replace("__DATA__", blob)
(pilot / "survey.html").write_text(page, encoding="utf-8", newline="\n")

sheet = ["# " + data["title"], ""] + [p + "\n" for p in data["intro"]] + ["**Consent.** " + data["consent"], "", "_Generated from `pilot/questions.json` by `pilot/build_survey.py`._", ""]
(pilot / "INFORMATION_SHEET.md").write_text("\n".join(sheet), encoding="utf-8", newline="\n")
print("Wrote pilot/survey.html (%d KB, %d questions) and pilot/INFORMATION_SHEET.md" % (len(page) // 1024, len(data["questions"])))
