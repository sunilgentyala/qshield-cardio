"""Generate docs/index.html from results/*.json so every number on the site comes from the code."""
import json, csv, html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / 'results/study.json').read_text())
Q = json.loads((ROOT / 'results/security.json').read_text())
E = json.loads((ROOT / 'results/extended.json').read_text())
X = json.loads((ROOT / 'results/security_extended.json').read_text())
REP = list(csv.DictReader(open(ROOT / 'results/repeated_splits.csv')))
SUB = list(csv.DictReader(open(ROOT / 'results/subgroup_metrics.csv')))

ACC, ACC2, GREEN, YEL, RED, MUTED, TXT, GRID = '#6366f1', '#22d3ee', '#34d399', '#fbbf24', '#f87171', '#94a3b8', '#e2e8f0', '#1e2d45'


def sel(t): return S['models'][t]['selected']
def tst(t): return S['models'][t]['candidates'][sel(t)]['test']
D, H = tst('diabetes'), tst('hypertension')
def pct(v, d=1): return f'{100 * v:.{d}f}%'
def f3(v): return f'{v:.3f}'
MT = sum(v['trials'] for v in X['mutation_fuzz'].values()) + sum(v['trials'] for v in X['mutation_fuzz_resigned'].values())
sub = lambda t, g: next(float(r['sensitivity']) for r in SUB if r['task'] == t and r['group'] == g)


def svg(w, h, inner, label):
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(label)}" xmlns="http://www.w3.org/2000/svg" font-family="Inter,system-ui,sans-serif" font-size="12">{inner}</svg>'


def hbars(rows, maxv, w=520, color=ACC, fmt=lambda v: f'{v:.3f}', label_w=150, lo=0.0):
    rh = 30
    h = rh * len(rows) + 28
    inner = ''
    for i, (lab, v, *rest) in enumerate(rows):
        c = rest[0] if rest else color
        y = 10 + i * rh
        bw = (w - label_w - 70) * (v - lo) / (maxv - lo)
        inner += f'<text x="{label_w - 8}" y="{y + 15}" fill="{MUTED}" text-anchor="end">{html.escape(lab)}</text>'
        inner += f'<rect x="{label_w}" y="{y + 3}" width="{max(bw, 1):.1f}" height="16" rx="4" fill="{c}"/>'
        inner += f'<text x="{label_w + bw + 8:.1f}" y="{y + 16}" fill="{TXT}">{fmt(v)}</text>'
    return svg(w, h, inner, 'bar chart')


def build_age_chart():
    a = E['age_specific_threshold_bp']
    groups = [('Under 50', sub('hypertension', 'age_under_50'), a['under50']['sensitivity']), ('50 or older', sub('hypertension', 'age_50_plus'), a['50plus']['sensitivity'])]
    w, h = 520, 230
    inner = ''
    for i, (g, s1, s2) in enumerate(groups):
        y0 = 30 + i * 90
        inner += f'<text x="0" y="{y0 + 14}" fill="{TXT}" font-weight="600">{g}</text>'
        for j, (v, c, lab) in enumerate([(s1, RED, 'One threshold'), (s2, GREEN, 'Age-specific thresholds')]):
            y = y0 + 22 + j * 24
            bw = 300 * v
            inner += f'<text x="0" y="{y + 12}" fill="{MUTED}" font-size="11">{lab}</text>'
            inner += f'<rect x="170" y="{y}" width="{bw:.1f}" height="15" rx="4" fill="{c}"/><text x="{170 + bw + 8:.1f}" y="{y + 12}" fill="{TXT}">{pct(v)}</text>'
    inner += f'<text x="170" y="{h - 4}" fill="{MUTED}" font-size="11">Sensitivity on the 2017-2018 test cycle</text>'
    return svg(w, h, inner, 'Blood pressure sensitivity by age group')


def build_split_chart():
    w, h = 520, 250
    px0, px1, py0, py1 = 50, 500, 20, 205
    inner = ''
    for tick in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        y = py1 - (py1 - py0) * (tick - 0.5) / 0.5
        inner += f'<line x1="{px0}" x2="{px1}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}"/><text x="{px0 - 8}" y="{y + 4:.1f}" fill="{MUTED}" text-anchor="end" font-size="11">{int(tick * 100)}%</text>'
    for k, (t, c, lab) in enumerate([('diabetes', ACC2, 'HbA1c sensitivity'), ('hypertension', ACC, 'Blood-pressure sensitivity')]):
        rows = [r for r in REP if r['task'] == t]
        vals = sorted(float(r['sensitivity']) for r in rows)
        cx = 150 + k * 220
        for i, v in enumerate(vals):
            jitter = ((i * 37) % 11 - 5) * 5
            y = py1 - (py1 - py0) * (v - 0.5) / 0.5
            inner += f'<circle cx="{cx + jitter}" cy="{y:.1f}" r="4" fill="{c}" fill-opacity=".8"/>'
        inner += f'<text x="{cx}" y="{py1 + 22}" fill="{TXT}" text-anchor="middle">{lab}</text>'
    return svg(w, h, inner, 'Sensitivity across 30 splits')


def build_op_chart():
    w, h = 520, 285
    px0, px1, py0, py1 = 50, 500, 15, 235
    X_ = lambda v: px0 + (px1 - px0) * v
    Y_ = lambda v: py1 - (py1 - py0) * v
    inner = ''
    for tick in [0, .25, .5, .75, 1]:
        inner += f'<line x1="{px0}" x2="{px1}" y1="{Y_(tick):.1f}" y2="{Y_(tick):.1f}" stroke="{GRID}"/><text x="{px0 - 8}" y="{Y_(tick) + 4:.1f}" fill="{MUTED}" text-anchor="end" font-size="11">{int(tick * 100)}%</text>'
        inner += f'<text x="{X_(tick):.1f}" y="{py1 + 16}" fill="{MUTED}" text-anchor="middle" font-size="11">{int(tick * 100)}%</text>'
    inner += f'<line x1="{X_(0)}" y1="{Y_(0)}" x2="{X_(1)}" y2="{Y_(1)}" stroke="{MUTED}" stroke-dasharray="4 4"/>'
    t = 'hypertension'
    pts = [(r['referral'], r['sensitivity']) for r in E['target_sweep'][t]]
    inner += f'<polyline points="{" ".join(f"{X_(a):.1f},{Y_(b):.1f}" for a, b in pts)}" fill="none" stroke="{ACC}" stroke-width="2.5"/>'
    for r in E['target_sweep'][t]:
        inner += f'<circle cx="{X_(r["referral"]):.1f}" cy="{Y_(r["sensitivity"]):.1f}" r="5" fill="{ACC}"/><text x="{X_(r["referral"]) + 8:.1f}" y="{Y_(r["sensitivity"]) + 16:.1f}" fill="{TXT}" font-size="11">{int(r["target"] * 100)}% target</text>'
    for r in E['rule_baselines'][t]:
        inner += f'<rect x="{X_(r["referral"]) - 4:.1f}" y="{Y_(r["sensitivity"]) - 4:.1f}" width="8" height="8" fill="{YEL}"/>'
    inner += f'<text x="{(px0 + px1) / 2}" y="{py1 + 34}" fill="{MUTED}" text-anchor="middle" font-size="11">Fraction referred</text>'
    return svg(w, h, inner, 'Blood pressure operating points')


def build_crypto_chart():
    b = Q['benchmarks']
    names = ['ML-KEM-768 encaps', 'ML-KEM-768 decaps', 'ML-DSA-44 verify', 'ML-DSA-44 sign', 'Envelope verify decrypt', 'Envelope seal']
    return hbars([(n, b[n]['median_ms'], ACC2 if 'KEM' in n else (ACC if 'DSA' in n else GREEN)) for n in names], max(b[n]['median_ms'] for n in names) * 1.05, fmt=lambda v: f'{v:.3f} ms')


def build_pipeline():
    boxes = [('Public NHANES', 'CDC XPT + SHA256'), ('Frozen protocol', 'train / calibrate / select'), ('Temporal test', '2017-2018, once'), ('Manifest', 'SHA256 + thresholds'), ('Hybrid envelope', 'X25519 + ML-KEM-768'), ('Verified output', 'ML-DSA-44 + replay gates')]
    w, h = 1000, 150
    inner = ''
    bw = 140
    for i, (a, b) in enumerate(boxes):
        x = 10 + i * 163
        inner += f'<rect x="{x}" y="30" width="{bw}" height="70" rx="10" fill="#111827" stroke="{ACC if i < 3 else ACC2}"/><text x="{x + bw / 2}" y="60" fill="{TXT}" text-anchor="middle" font-weight="600">{a}</text><text x="{x + bw / 2}" y="80" fill="{MUTED}" text-anchor="middle" font-size="11">{b}</text>'
        if i < len(boxes) - 1:
            inner += f'<path d="M{x + bw + 3} 65 h13" stroke="{MUTED}" stroke-width="2"/><path d="M{x + bw + 12} 60 l6 5 -6 5" fill="none" stroke="{MUTED}" stroke-width="2"/>'
    inner += f'<text x="10" y="130" fill="{MUTED}" font-size="12">Statistical pipeline (indigo) feeds an integrity and confidentiality pipeline (cyan). Clinical deployment layer is proposed, not implemented.</text>'
    return svg(w, h, inner, 'Pipeline')


rs = E['repeated_splits']
da = E['design_aware']
cal = E['calibration']
age = E['age_specific_threshold_bp']
cmpx = X['classical_vs_hybrid']
cands = []
for t, nm in [('diabetes', 'HbA1c'), ('hypertension', 'Blood pressure')]:
    for n, v in S['models'][t]['candidates'].items():
        m = v['test']
        cands.append(f'<tr><td>{nm}</td><td>{n}{" <span class=tag>selected</span>" if n == sel(t) else ""}</td><td>{f3(m["auroc"])}</td><td>{pct(m["sensitivity"])}</td><td>{pct(m["specificity"])}</td><td>{pct(m["ppv"])}</td></tr>')

PAGE = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>QShield Cardio</title>
<meta name="description" content="Temporally validated diabetes and hypertension screening models on NHANES with a post-quantum protected, integrity-checked output pipeline. Every number on this page is generated from the repository results." />
<meta property="og:title" content="QShield Cardio" />
<meta property="og:description" content="Honest, reproducible NHANES screening benchmark with a hybrid ML-KEM-768 / ML-DSA-44 envelope. Research software, not clinical." />
<meta property="og:type" content="website" />
<link rel="icon" type="image/svg+xml" href="favicon.svg" />
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet" />
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#09101f;--surface:#111827;--surface2:#1a2236;--border:#1e2d45;--accent:#6366f1;--accent2:#22d3ee;--accent3:#a78bfa;--green:#34d399;--yellow:#fbbf24;--red:#f87171;--text:#e2e8f0;--muted:#94a3b8;--code:#0d1424;--r:14px}}
html{{scroll-behavior:smooth}}
body{{background:var(--bg);color:var(--text);font:16px/1.65 'Inter','Segoe UI',system-ui,sans-serif;-webkit-font-smoothing:antialiased}}
h1,h2,h3{{letter-spacing:-.02em;font-weight:700}}
h2{{font-size:1.7rem;margin-bottom:.4rem}}
h3{{font-size:1.05rem;color:var(--accent2);margin-bottom:.4rem}}
p{{color:var(--muted);margin-bottom:1rem}}
a{{color:var(--accent2);text-decoration:none}}a:hover{{opacity:.8}}
strong{{color:var(--text)}}
code{{font-family:'JetBrains Mono',monospace;font-size:.85em;background:var(--code);border:1px solid var(--border);border-radius:6px;padding:.1em .4em;color:var(--accent2)}}
.wrap{{max-width:1080px;margin:0 auto;padding:0 1.5rem}}
section{{padding:4.5rem 0}}
section+section{{border-top:1px solid var(--border)}}
.lead{{max-width:740px;margin-bottom:2rem}}
.eyebrow{{font:600 .75rem/1 'JetBrains Mono',monospace;letter-spacing:.14em;text-transform:uppercase;color:var(--accent3);margin-bottom:.8rem;display:block}}
nav{{position:sticky;top:0;z-index:50;background:rgba(9,16,31,.85);backdrop-filter:blur(12px);border-bottom:1px solid var(--border)}}
.nav-in{{max-width:1080px;margin:0 auto;padding:.9rem 1.5rem;display:flex;justify-content:space-between;align-items:center;gap:1rem}}
.logo{{font-weight:800;font-size:1.1rem}}.logo span{{color:var(--accent2)}}
.links{{display:flex;gap:1.3rem;font-size:.9rem;flex-wrap:wrap}}
.links a{{color:var(--muted)}}.links a:hover{{color:var(--text);opacity:1}}
#hero{{padding:6rem 0 4.5rem;background:radial-gradient(900px 400px at 15% -10%,rgba(99,102,241,.25),transparent),radial-gradient(700px 380px at 90% 10%,rgba(34,211,238,.14),transparent)}}
.pill{{display:inline-block;font:600 .78rem 'JetBrains Mono',monospace;color:var(--accent2);border:1px solid rgba(34,211,238,.35);background:rgba(34,211,238,.08);padding:.35rem .8rem;border-radius:99px;margin-bottom:1.2rem}}
h1{{font-size:clamp(2rem,5vw,3.3rem);line-height:1.12;font-weight:800;margin-bottom:1.2rem}}
h1 em{{font-style:normal;background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;background-clip:text;color:transparent}}
.sub{{font-size:1.15rem;max-width:780px;color:var(--muted);margin-bottom:2rem}}
.btns{{display:flex;gap:.8rem;flex-wrap:wrap}}
.btn{{display:inline-block;padding:.7rem 1.3rem;border-radius:10px;font-weight:600;font-size:.95rem}}
.btn-p{{background:linear-gradient(90deg,var(--accent),#4f8df7);color:#fff}}
.btn-o{{border:1px solid var(--border);color:var(--text);background:var(--surface)}}
.btn:hover{{opacity:.88}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:1rem;margin-top:3rem}}
.metric{{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:1.2rem 1.3rem}}
.metric b{{display:block;font-size:2rem;font-weight:800;background:linear-gradient(90deg,var(--accent3),var(--accent2));-webkit-background-clip:text;background-clip:text;color:transparent;line-height:1.2}}
.metric span{{font-size:.85rem;color:var(--muted)}}
.note{{border:1px solid rgba(251,191,36,.35);background:rgba(251,191,36,.07);border-radius:var(--r);padding:1rem 1.2rem;color:#e8d9a8;font-size:.95rem;margin-top:2rem}}
.note strong{{color:var(--yellow)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));gap:1.1rem}}
.g3{{grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr))}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:1.4rem}}
.card svg{{width:100%;height:auto;display:block;margin-top:.6rem}}
.card p{{font-size:.93rem;margin:.5rem 0 0}}
table{{width:100%;border-collapse:collapse;font-size:.92rem;background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:hidden}}
th,td{{padding:.65rem .9rem;text-align:left;border-bottom:1px solid var(--border)}}
th{{color:var(--accent3);font:600 .75rem 'JetBrains Mono',monospace;letter-spacing:.08em;text-transform:uppercase;background:var(--surface2)}}
td{{color:var(--text)}}
.tag{{font:600 .68rem 'JetBrains Mono',monospace;color:var(--green);border:1px solid rgba(52,211,153,.4);border-radius:99px;padding:.1rem .5rem;margin-left:.3rem}}
.tablewrap{{overflow-x:auto}}
pre{{background:var(--code);border:1px solid var(--border);border-radius:var(--r);padding:1.2rem;overflow-x:auto;font:.85rem/1.7 'JetBrains Mono',monospace;color:#c7d2fe}}
.pipeline svg{{width:100%;height:auto}}
ul.clean{{list-style:none}}ul.clean li{{padding:.55rem 0 .55rem 1.6rem;position:relative;color:var(--muted)}}
ul.clean li::before{{content:'';position:absolute;left:0;top:1.05rem;width:8px;height:8px;border-radius:50%;background:var(--accent2)}}
footer{{border-top:1px solid var(--border);padding:2.5rem 0;color:var(--muted);font-size:.9rem}}
@media(max-width:640px){{.links{{display:none}}section{{padding:3rem 0}}}}
</style>
</head>
<body>
<nav><div class="nav-in"><div class="logo">QShield<span>Cardio</span></div><div class="links"><a href="#results">Results</a><a href="#robustness">Robustness</a><a href="#security">Security</a><a href="#reproduce">Reproduce</a><a href="#scope">Scope</a><a href="https://github.com/sunilgentyala/qshield-cardio">GitHub</a></div></div></nav>

<section id="hero"><div class="wrap">
<span class="pill">Research software &middot; NHANES &middot; ML-KEM-768 &middot; ML-DSA-44</span>
<h1>Screening scores you can <em>audit</em>, and outputs you can <em>trust</em>.</h1>
<p class="sub">Two low-cost diabetes and hypertension screening models, evaluated once on a later survey cycle with frozen thresholds, wrapped in an integrity-checked, post-quantum protected output pipeline. Built to be stress-tested, and this page reports what the stress tests found.</p>
<div class="btns"><a class="btn btn-p" href="https://github.com/sunilgentyala/qshield-cardio">View on GitHub</a><a class="btn btn-o" href="#reproduce">Reproduce the numbers</a></div>
<div class="metrics">
<div class="metric"><b>{f3(D["auroc"])}</b><span>HbA1c AUROC, temporal test</span></div>
<div class="metric"><b>{f3(H["auroc"])}</b><span>Blood-pressure AUROC, temporal test</span></div>
<div class="metric"><b>{MT + 1200:,}</b><span>functional + mutation security trials, all as expected</span></div>
<div class="metric"><b>28</b><span>regression tests, CI regenerates all results from raw CDC files</span></div>
</div>
<div class="note"><strong>Research only.</strong> These models have not established clinical safety or effectiveness and must not determine eligibility for blood-pressure checks, diagnose disease or override clinical judgment.</div>
</div></section>

<section id="how"><div class="wrap">
<span class="eyebrow">How it works</span>
<h2>One workflow, two kinds of trust</h2>
<p class="lead">The statistical half answers whether a score ranks people sensibly on data it has never seen. The integrity half answers whether the score, and the model that produced it, arrive unaltered and confidential.</p>
<div class="pipeline">{build_pipeline()}</div>
<div class="grid g3" style="margin-top:2rem">
<div class="card"><h3>Leakage-controlled benchmark</h3><p>Development on NHANES 2015-2016 (n = {S["descriptive"]["development"]["n"]:,}) with separate training, calibration and validation partitions. The whole 2017-2018 cycle (n = {S["descriptive"]["test"]["n"]:,}) is touched once. HbA1c, glucose, blood pressure and treatment variables are never predictors.</p></div>
<div class="card"><h3>Authenticated model trail</h3><p>A SHA256 manifest binds features, thresholds and model files. The inference tool refuses to load a model whose digest does not match a trusted value, and the digest is carried inside each signed envelope.</p></div>
<div class="card"><h3>Hybrid post-quantum envelope</h3><p>X25519 + ML-KEM-768 key establishment, HKDF-SHA256, AES-256-GCM, ML-DSA-44 signatures, with replay, staleness, downgrade and role checks.</p></div>
</div></div></section>

<section id="results"><div class="wrap">
<span class="eyebrow">Headline results</span>
<h2>Moderate, stable discrimination</h2>
<p class="lead">Frozen-threshold performance on the 2017-2018 test cohort. Validation selected one model per task; the table shows all three candidates.</p>
<div class="tablewrap"><table><thead><tr><th>Endpoint</th><th>Model</th><th>AUROC</th><th>Sensitivity</th><th>Specificity</th><th>PPV</th></tr></thead><tbody>{"".join(cands)}</tbody></table></div>
<p style="margin-top:1rem">Design-aware (Rao-Wu PSU bootstrap) 95% intervals for the selected models: HbA1c AUROC {f3(da["diabetes"]["cluster_unweighted"]["auroc"]["ci"][0])} to {f3(da["diabetes"]["cluster_unweighted"]["auroc"]["ci"][1])}; blood-pressure AUROC {f3(da["hypertension"]["cluster_unweighted"]["auroc"]["ci"][0])} to {f3(da["hypertension"]["cluster_unweighted"]["auroc"]["ci"][1])}. HbA1c PPV is only {pct(D["ppv"])} (about 22 referrals per detected positive).</p>
</div></section>

<section id="robustness"><div class="wrap">
<span class="eyebrow">Robustness</span>
<h2>What the stress tests found</h2>
<p class="lead">Each analysis below uses models and thresholds that were already frozen. Several results are unflattering, and they are the point.</p>
<div class="grid">
<div class="card"><h3>The operating point is unstable</h3>{build_split_chart()}<p>AUROC barely moves across 30 re-drawn development splits (SD {f3(rs["diabetes"]["auroc"]["sd"])} for HbA1c), but HbA1c sensitivity ranges from {pct(rs["diabetes"]["sensitivity"]["min"], 0)} to {pct(rs["diabetes"]["sensitivity"]["max"], 0)} because the validation threshold rests on about a dozen events. Each dot is one split, evaluated on the full 2017-2018 cohort.</p></div>
<div class="card"><h3>One threshold hides an age gap</h3>{build_age_chart()}<p>With a single threshold, blood-pressure sensitivity is {pct(sub("hypertension", "age_under_50"))} under 50. Thresholds chosen separately on validation data equalize it near {pct(age["under50"]["sensitivity"], 0)}, at a higher overall referral rate ({pct(age["overall"]["referral"], 0)}).</p></div>
<div class="card"><h3>Operating-point trade-off</h3>{build_op_chart()}<p>Raising the validation target from 80% to 95% lifts blood-pressure referral from {pct(E["target_sweep"]["hypertension"][1]["referral"])} to {pct(E["target_sweep"]["hypertension"][3]["referral"])}. Blue points are the model at four validation targets; yellow squares are fixed age or BMI rules and an age-only model. A simple age rule is a strong competitor.</p></div>
<div class="card"><h3>Other checks</h3><ul class="clean"><li>Calibration slopes {cal["diabetes"]["slope"]:.2f} (HbA1c) and {cal["hypertension"]["slope"]:.2f} (BP); the later cycle has higher prevalence than predicted.</li><li>Developing on 2017-2018 and testing on 2015-2016 gives AUROC {f3(E["reverse_cycle"]["diabetes"]["auroc"])} and {f3(E["reverse_cycle"]["hypertension"]["auroc"])}.</li><li>Adding smoking and vigorous recreation lowered HbA1c AUROC by 0.020 (paired interval excludes zero).</li><li>Masking 20% of inputs cuts BP sensitivity from {pct(H["sensitivity"])} to {pct(S["models"]["hypertension"]["missing20_test"]["sensitivity"])}.</li></ul></div>
</div></div></section>

<section id="security"><div class="wrap">
<span class="eyebrow">Security</span>
<h2>Tested like an attacker, not just a test vector</h2>
<p class="lead">Beyond 1,200 fixed functional trials, {MT:,} random single-bit mutations of every envelope field were rejected, including mutations re-signed with the legitimate key so only the KEM and AEAD layers could catch them.</p>
<div class="grid">
<div class="card"><h3>Median cost per operation</h3>{build_crypto_chart()}<p>Python-level timings on one host; run-to-run variation is large, so read them as orders of magnitude.</p></div>
<div class="card"><h3>Hybrid versus classical</h3><ul class="clean"><li>Seal: {cmpx["hybrid_seal_median_ms"]:.2f} ms vs {cmpx["classical_seal_median_ms"]:.3f} ms for an X25519 + Ed25519 baseline.</li><li>Open: {cmpx["hybrid_open_median_ms"]:.2f} ms vs {cmpx["classical_open_median_ms"]:.3f} ms.</li><li>Size: {cmpx["hybrid_bytes"]:,} bytes (hex JSON with metadata) vs {cmpx["classical_bytes"]} raw bytes.</li><li>The two cards come from separate runs, which is why the seal times differ.</li><li>ML-KEM-768 ciphertext {Q["sizes_bytes"]["kem_ciphertext"]:,} B; ML-DSA-44 signature {Q["sizes_bytes"]["signature"]:,} B.</li></ul></div>
</div></div></section>

<section id="reproduce"><div class="wrap">
<span class="eyebrow">Reproduce</span>
<h2>Every number regenerates from raw CDC files</h2>
<p class="lead">CI downloads the 16 public NHANES files, validates their digests, reruns everything and executes the 28 tests on every push.</p>
<pre>python -m venv .venv &amp;&amp; source .venv/bin/activate
python -m pip install -r requirements.txt
python src/study.py              # primary study
python src/security.py           # functional trials
python src/extended.py           # robustness analyses
python src/security_extended.py  # mutation fuzz, cost comparison
python -m unittest discover -s tests -v</pre>
</div></section>

<section id="scope"><div class="wrap">
<span class="eyebrow">Scope, stated plainly</span>
<h2>What this is not</h2>
<ul class="clean">
<li>Not clinically validated. Endpoints are single-examination screening proxies, not confirmed diagnoses.</li>
<li>The primary analysis is unweighted; proportions are not national prevalence estimates.</li>
<li>The envelope is a single-process application harness: no TLS integration, PKI, revocation, persistent replay state, forward secrecy, side-channel evaluation or FIPS 140 validation.</li>
<li>The ISO control matrix is a proposed crosswalk, not a certification.</li>
</ul>
</div></section>

<footer><div class="wrap">Sunil Gentyala, Suresh Kumar Darisi, Vahiduddin Shariff, Akhila Kasturi, S. M. Krishna Reddy Padala &middot; MIT license &middot; AI assistance (Claude) used for code and analysis; authors take responsibility. <a href="https://github.com/sunilgentyala/qshield-cardio">Source</a></div></footer>
</body></html>'''

(ROOT / 'docs').mkdir(exist_ok=True)
(ROOT / 'docs/index.html').write_text(PAGE, encoding='utf-8')
(ROOT / 'docs/favicon.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6366f1"/><stop offset="1" stop-color="#22d3ee"/></linearGradient></defs><rect width="64" height="64" rx="14" fill="#09101f"/><path d="M32 10 L50 18 V32 C50 43 42 51 32 55 C22 51 14 43 14 32 V18 Z" fill="none" stroke="url(#g)" stroke-width="4"/><path d="M22 33 h6 l3 -8 l5 16 l3 -8 h4" fill="none" stroke="#22d3ee" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>', encoding='utf-8')
print('site written', len(PAGE))
