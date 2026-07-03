#!/usr/bin/env python3
"""
analyze_label_transfer.py
=========================
Compares scan_tag (original scanner labels) with deploy_tag (d_v3 deployment
labels) across sentences_labeled.csv. Produces:

  1. Label proportion table (console)
  2. Transfer matrix — how each scan_tag distributed across deploy_tags (console)
  3. Self-contained HTML report:
       - Sankey diagram of tag transfers
       - Confidence histogram by tag

Usage (from repo root):
    python3 eval/analyze_label_transfer.py
    python3 eval/analyze_label_transfer.py --out eval/results/label_transfer.html
    python3 eval/analyze_label_transfer.py --done-only   # skip rows not yet labeled
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import PATHS

TAGS = ["divine_teleology", "non_divine_teleology", "internal_essence", "junk"]
TAG_SHORT = {
    "divine_teleology":     "DT",
    "non_divine_teleology": "NDT",
    "internal_essence":     "IE",
    "junk":                 "junk",
}
COLORS = {
    "divine_teleology":     "#6366f1",
    "non_divine_teleology": "#f59e0b",
    "internal_essence":     "#10b981",
    "junk":                 "#94a3b8",
    "error":                "#ef4444",
}


def load_data(done_only: bool):
    sentences_path = PATHS["sentences_csv"]
    labeled_path   = Path(sentences_path).parent / "sentences_labeled.csv"

    with open(sentences_path, newline="", encoding="utf-8") as f:
        source = list(csv.DictReader(f))
    with open(labeled_path, newline="", encoding="utf-8") as f:
        labeled = list(csv.DictReader(f))

    rows = []
    for s, l in zip(source, labeled):
        if done_only and l.get("deploy_done") != "True":
            continue
        if l.get("deploy_done") != "True":
            continue  # always skip unlabeled
        rows.append({
            "scan_tag":   s.get("scan_tag", "").strip() or "unknown",
            "deploy_tag": l.get("deploy_tag", "").strip() or "error",
            "confidence": float(l.get("deploy_confidence", 0) or 0),
        })
    return rows


def proportion_table(rows):
    total = len(rows)
    scan_counts   = Counter(r["scan_tag"]   for r in rows)
    deploy_counts = Counter(r["deploy_tag"] for r in rows)

    all_tags = sorted(set(list(scan_counts) + list(deploy_counts)),
                      key=lambda t: -scan_counts.get(t, 0))

    print(f"\n{'Tag':<25}  {'scan_tag':>10}  {'deploy_tag':>11}  {'Δ':>7}")
    print("-" * 58)
    for tag in all_tags:
        s = scan_counts.get(tag, 0) / total
        d = deploy_counts.get(tag, 0) / total
        delta = d - s
        print(f"  {tag:<23}  {s:>9.1%}  {d:>10.1%}  {delta:>+7.1%}")
    print(f"\n  Total rows analyzed: {total}")


def transfer_matrix(rows):
    # Only rows where scan_tag is a known class
    known = [r for r in rows if r["scan_tag"] in TAGS]
    total_known = len(known)

    matrix = defaultdict(Counter)
    for r in known:
        matrix[r["scan_tag"]][r["deploy_tag"]] += 1

    deploy_cols = TAGS + ["error"]

    print(f"\nTransfer matrix (row = scan_tag, col = deploy_tag) — {total_known} rows with known scan_tag")
    print(f"\n  {'':25}", end="")
    for col in deploy_cols:
        print(f"  {TAG_SHORT.get(col, col):>6}", end="")
    print(f"  {'total':>6}")
    print("  " + "-" * (25 + 8 * len(deploy_cols) + 8))

    for src in TAGS:
        row_total = sum(matrix[src].values())
        if row_total == 0:
            continue
        print(f"  {src:<25}", end="")
        for col in deploy_cols:
            n = matrix[src].get(col, 0)
            pct = n / row_total * 100
            print(f"  {pct:>5.1f}%", end="")
        print(f"  {row_total:>6}")

    return matrix


def build_html(rows, matrix, out_path: Path):
    # --- Sankey data ---
    # Nodes: scan_tag sources (prefixed S_) + deploy_tag sinks (prefixed D_)
    src_tags = [t for t in TAGS if any(matrix[t].values())]
    dst_tags = TAGS + (["error"] if any(matrix[s].get("error", 0) for s in src_tags) else [])

    nodes = []
    node_idx = {}
    for t in src_tags:
        node_idx[f"S_{t}"] = len(nodes)
        nodes.append({"name": f"{TAG_SHORT.get(t,t)} (scan)", "color": COLORS.get(t, "#888")})
    for t in dst_tags:
        node_idx[f"D_{t}"] = len(nodes)
        nodes.append({"name": f"{TAG_SHORT.get(t,t)} (deploy)", "color": COLORS.get(t, "#888")})

    links = []
    for src in src_tags:
        for dst in dst_tags:
            n = matrix[src].get(dst, 0)
            if n > 0:
                links.append({
                    "source": node_idx[f"S_{src}"],
                    "target": node_idx[f"D_{dst}"],
                    "value":  n,
                })

    sankey_data = {"nodes": nodes, "links": links}

    # --- Confidence histogram data ---
    bins = [i / 20 for i in range(21)]  # 0.00, 0.05, ..., 1.00
    bin_labels = [f"{b:.2f}" for b in bins[:-1]]

    hist_by_tag = {}
    for tag in TAGS + ["error"]:
        confs = [r["confidence"] for r in rows if r["deploy_tag"] == tag]
        counts = [0] * (len(bins) - 1)
        for c in confs:
            idx = min(int(c / 0.05), len(counts) - 1)
            counts[idx] += 1
        hist_by_tag[tag] = counts

    hist_data = {
        "labels":    bin_labels,
        "tags":      list(hist_by_tag.keys()),
        "counts":    hist_by_tag,
        "colors":    {t: COLORS.get(t, "#888") for t in hist_by_tag},
    }

    # --- Proportion data ---
    total = len(rows)
    scan_counts   = Counter(r["scan_tag"]   for r in rows)
    deploy_counts = Counter(r["deploy_tag"] for r in rows)
    prop_data = {
        "tags":   TAGS,
        "scan":   [round(scan_counts.get(t, 0) / total * 100, 1) for t in TAGS],
        "deploy": [round(deploy_counts.get(t, 0) / total * 100, 1) for t in TAGS],
        "colors": [COLORS.get(t, "#888") for t in TAGS],
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Label Transfer Analysis</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0;
         margin: 0; padding: 24px; }}
  h1   {{ font-size: 1.4rem; color: #f1f5f9; margin-bottom: 4px; }}
  h2   {{ font-size: 1.0rem; color: #94a3b8; margin: 32px 0 12px; text-transform: uppercase;
          letter-spacing: .08em; }}
  .subtitle {{ color: #64748b; font-size: .85rem; margin-bottom: 32px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 20px; }}
  .full {{ grid-column: 1 / -1; }}
  canvas {{ max-height: 320px; }}
  #sankey svg {{ width: 100%; height: 420px; }}
  .node rect {{ stroke: none; rx: 3; }}
  .link {{ fill: none; opacity: .45; }}
  .link:hover {{ opacity: .75; }}
  .node text {{ font-size: 12px; fill: #cbd5e1; }}
</style>
</head>
<body>
<h1>Label Transfer Analysis — scan_tag vs deploy_tag (d_v3)</h1>
<p class="subtitle">Labeled rows: {total} / 12913 &nbsp;|&nbsp;
   Generated from sentences_labeled.csv</p>

<div class="grid">
  <div class="card">
    <h2>Label proportions</h2>
    <canvas id="propChart"></canvas>
  </div>
  <div class="card">
    <h2>Confidence distribution</h2>
    <canvas id="confChart"></canvas>
  </div>
  <div class="card full">
    <h2>Tag transfer (Sankey)</h2>
    <div id="sankey"></div>
  </div>
</div>

<script>
const PROP  = {json.dumps(prop_data)};
const HIST  = {json.dumps(hist_data)};
const SDATA = {json.dumps(sankey_data)};

// --- Proportion chart ---
new Chart(document.getElementById('propChart'), {{
  type: 'bar',
  data: {{
    labels: PROP.tags.map(t => t.replace('_', ' ')),
    datasets: [
      {{ label: 'scan_tag',   data: PROP.scan,   backgroundColor: PROP.colors.map(c => c + 'aa') }},
      {{ label: 'deploy_tag', data: PROP.deploy, backgroundColor: PROP.colors }},
    ],
  }},
  options: {{
    responsive: true, plugins: {{ legend: {{ labels: {{ color: '#cbd5e1' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }},
      y: {{ ticks: {{ color: '#94a3b8', callback: v => v + '%' }},
            grid: {{ color: '#334155' }} }},
    }},
  }},
}});

// --- Confidence histogram ---
const confDatasets = HIST.tags.map(tag => ({{
  label: tag.replace('_',' '),
  data:  HIST.counts[tag],
  backgroundColor: HIST.colors[tag] + 'cc',
  stack: 'conf',
}}));
new Chart(document.getElementById('confChart'), {{
  type: 'bar',
  data: {{ labels: HIST.labels, datasets: confDatasets }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#cbd5e1', boxWidth: 12 }} }} }},
    scales: {{
      x: {{ stacked: true, ticks: {{ color: '#94a3b8', maxRotation: 90 }},
            grid: {{ color: '#334155' }},
            title: {{ display: true, text: 'confidence', color: '#94a3b8' }} }},
      y: {{ stacked: true, ticks: {{ color: '#94a3b8' }},
            grid: {{ color: '#334155' }},
            title: {{ display: true, text: 'count', color: '#94a3b8' }} }},
    }},
  }},
}});

// --- Sankey (D3) ---
(function() {{
  const W = document.getElementById('sankey').clientWidth || 900;
  const H = 420, PAD = {{ top: 20, right: 160, bottom: 20, left: 160 }};

  const svg = d3.select('#sankey').append('svg')
    .attr('viewBox', `0 0 ${{W}} ${{H}}`);

  const nodes = SDATA.nodes.map((d, i) => ({{ ...d, id: i }}));
  const links = SDATA.links.map(d => ({{ ...d }}));

  // Simple manual Sankey layout: left column = source, right = target
  const srcNodes = nodes.filter((_, i) => i < {len(src_tags)});
  const dstNodes = nodes.filter((_, i) => i >= {len(src_tags)});

  const innerH = H - PAD.top - PAD.bottom;
  const innerW = W - PAD.left - PAD.right;

  function layoutColumn(col, x, totalVal) {{
    const gap = 8;
    const totalGap = gap * (col.length - 1);
    let y = PAD.top;
    col.forEach(n => {{
      n.x0 = x; n.x1 = x + 20;
      n.value = links.filter(l =>
        (l.source === n.id || l.target === n.id)
      ).reduce((s, l) => s + l.value, 0) || 1;
      n.h = Math.max(4, (n.value / totalVal) * (innerH - totalGap));
      n.y0 = y; n.y1 = y + n.h;
      y += n.h + gap;
    }});
  }}

  const totalVal = d3.sum(links, l => l.value);
  layoutColumn(srcNodes, PAD.left, totalVal);
  layoutColumn(dstNodes, PAD.left + innerW, totalVal);

  // Build per-source offset trackers for link stacking
  const srcOffset = {{}};
  srcNodes.forEach(n => {{ srcOffset[n.id] = n.y0; }});
  const dstOffset = {{}};
  dstNodes.forEach(n => {{ dstOffset[n.id] = n.y0; }});

  const g = svg.append('g');

  // Links
  links.forEach(l => {{
    const sn = nodes[l.source], dn = nodes[l.target];
    const lh = Math.max(1, (l.value / totalVal) * (innerH - 8 * (srcNodes.length - 1)));
    const sy0 = srcOffset[sn.id]; srcOffset[sn.id] += lh;
    const dy0 = dstOffset[dn.id]; dstOffset[dn.id] += lh;

    const path = d3.path();
    const mx = (sn.x1 + dn.x0) / 2;
    path.moveTo(sn.x1, sy0);
    path.bezierCurveTo(mx, sy0, mx, dy0, dn.x0, dy0);
    path.lineTo(dn.x0, dy0 + lh);
    path.bezierCurveTo(mx, dy0 + lh, mx, sy0 + lh, sn.x1, sy0 + lh);
    path.closePath();

    g.append('path')
      .attr('class', 'link')
      .attr('d', path)
      .attr('fill', sn.color)
      .append('title').text(`${{sn.name}} → ${{dn.name}}: ${{l.value}}`);
  }});

  // Nodes
  [...srcNodes, ...dstNodes].forEach(n => {{
    g.append('rect')
      .attr('x', n.x0).attr('y', n.y0)
      .attr('width', n.x1 - n.x0).attr('height', n.y1 - n.y0)
      .attr('fill', n.color).attr('rx', 3);

    const isLeft = n.x0 < W / 2;
    g.append('text')
      .attr('x', isLeft ? n.x0 - 6 : n.x1 + 6)
      .attr('y', (n.y0 + n.y1) / 2)
      .attr('dy', '0.35em')
      .attr('text-anchor', isLeft ? 'end' : 'start')
      .attr('font-size', 12)
      .attr('fill', '#cbd5e1')
      .text(n.name);
  }});
}})();
</script>
</body>
</html>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nHTML report saved: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str,
                        default="eval/results/label_transfer.html")
    args = parser.parse_args()

    print("Loading data...", flush=True)
    rows = load_data(done_only=True)

    if not rows:
        print("No labeled rows found yet. Run again once sentences_labeled.csv has data.")
        sys.exit(0)

    print(f"Analyzing {len(rows)} labeled rows...")

    proportion_table(rows)
    print()
    matrix = transfer_matrix(rows)

    out_path = Path(__file__).resolve().parent.parent / args.out
    build_html(rows, matrix, out_path)


if __name__ == "__main__":
    main()
