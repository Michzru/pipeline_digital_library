import os
import json
import joblib
import html

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURÁCIA
# ─────────────────────────────────────────────────────────────────────────────
json_path = "data/document_graph_with_edges_test.json"
cache_path = "data/cached_images.joblib"
output_dir = "data/interactive_pages3"

os.makedirs(output_dir, exist_ok=True)

print("Načítavam JSON a obrázky z cache...")
with open(json_path, "r", encoding="utf-8") as f:
    document_data = json.load(f)

images = joblib.load(cache_path)

# ─────────────────────────────────────────────────────────────────────────────
# FARBY
# ─────────────────────────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "Caption": "#a78bfa",
    "Picture": "#fb923c",
    "Table": "#38bdf8",
    "Formula": "#34d399",
    "Section-header": "#f472b6",
    "Page-footer": "#94a3b8",
    "Page-header": "#fbbf24",
    "Other": "#6b7280",
    "Unknown": "#374151",
}


def conf_pct(val):
    if val is None: return "—"
    return f"{val * 100:.0f}%"


def get_node_label(node):
    gat = node.get("predicted_label") or node.get("gat_reclassified_label") or node.get("label", "Unknown")
    yolo = node.get("label", "Unknown")
    return gat, yolo


def get_gat_conf(node):
    c = node.get("predicted_confidence") or node.get("gat_confidence")
    return float(c) if c is not None else None


def build_node_id_map(nodes_list):
    return {n["node_id"]: i for i, n in enumerate(nodes_list)}


# ─────────────────────────────────────────────────────────────────────────────
# HTML ŠABLÓNA
# ─────────────────────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="sk">
<head>
<meta charset="UTF-8">
<title>Page {page_number} — Graph Viewer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
  --bg:       #0d1117;
  --panel:    #161b22;
  --border:   #21262d;
  --border2:  #30363d;
  --text:     #e6edf3;
  --muted:    #7d8590;
  --accent:   #58a6ff;
  --mono:     'IBM Plex Mono', monospace;
  --sans:     'IBM Plex Sans', sans-serif;
}}

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
  background: var(--bg);
  color: var(--text);
  font-family: var(--sans);
  font-size: 13px;
  line-height: 1.5;
  min-height: 100vh;
}}

/* ── TOP BAR ── */
.topbar {{
  position: sticky; top: 0; z-index: 200;
  background: rgba(13,17,23,0.95);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--border);
  height: 44px;
  display: flex; align-items: center; gap: 16px;
  padding: 0 20px;
}}
.topbar-brand {{ font-family: var(--mono); font-size: 11px; font-weight: 600; color: var(--accent); letter-spacing: .08em; text-transform: uppercase; }}
.nav-btns {{ display: flex; gap: 6px; }}
.nav-btns a {{ font-family: var(--mono); font-size: 11px; padding: 4px 12px; background: var(--panel); border: 1px solid var(--border2); color: var(--text); text-decoration: none; border-radius: 4px; transition: border-color .1s, color .1s; }}
.nav-btns a:hover {{ border-color: var(--accent); color: var(--accent); }}
.topbar-page {{ margin-left: auto; font-family: var(--mono); font-size: 11px; color: var(--muted); }}

/* ── LEGEND BAR ── */
.legend {{ background: var(--panel); border-bottom: 1px solid var(--border); padding: 8px 20px; display: flex; flex-wrap: wrap; align-items: center; gap: 6px 14px; }}
.legend-label {{ font-family: var(--mono); font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; margin-right: 4px; }}
.lchip {{ display: inline-flex; align-items: center; gap: 5px; font-size: 11px; padding: 2px 8px; border-radius: 3px; border: 1px solid transparent; cursor: default; }}
.lchip-dot {{ width: 7px; height: 7px; border-radius: 50%; flex-shrink:0; }}
.legend-sep {{ width:1px; height:18px; background:var(--border2); }}
.ledge {{ display:inline-flex; align-items:center; gap:5px; font-size:11px; }}

/* ── STATS BAR ── */
.statsbar {{ display: flex; gap: 0; border-bottom: 1px solid var(--border); background: var(--bg); }}
.stat {{ padding: 6px 18px; border-right: 1px solid var(--border); font-family: var(--mono); font-size: 11px; color: var(--muted); }}
.stat b {{ color: var(--text); font-weight: 600; }}
.stat-hi {{ color: #22c55e; }}
.stat-mid {{ color: #f59e0b; }}
.stat-lo {{ color: #f87171; }}

/* ── CANVAS ── */
.canvas-wrap {{ padding: 24px; display: flex; justify-content: center; align-items: flex-start; }}
.page-container {{ position: relative; display: inline-block; border: 1px solid var(--border2); border-radius: 4px; overflow: visible; box-shadow: 0 4px 32px rgba(0,0,0,0.5); }}
.page-container img {{ display: block; max-width: 960px; width: 100%; height: auto; border-radius: 3px; }}
.edges-svg {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 5; overflow: visible; }}

/* ── BOUNDING BOX ── */
.bbox {{ position: absolute; border: 1.5px solid; border-radius: 2px; cursor: pointer; z-index: 10; transition: background .12s; }}
.bbox:hover {{ z-index: 50; background: rgba(255,255,255,0.05) !important; box-shadow: 0 0 0 2px rgba(255,255,255,0.3); }}
.bbox.reclassified {{ border-style: dashed; border-width: 2px; }}

/* ── CHIP ON BOX ── */
.bbox-chip {{ position: absolute; top: -24px; left: -1px; display: flex; align-items: center; gap: 3px; pointer-events: none; z-index: 51; white-space: nowrap; }}
.chip {{ font-family: var(--mono); font-size: 9.5px; font-weight: 600; padding: 2px 6px; border-radius: 3px; color: #fff; line-height: 1.4; }}
.chip-gat {{ opacity: 1; }}
.chip-yolo {{ opacity: 0.6; font-size: 9px; }}

/* ── TOOLTIP ── */
.tooltip {{ visibility: hidden; opacity: 0; position: absolute; top: calc(100% + 8px); left: 0; z-index: 200; width: 260px; background: var(--panel); border: 1px solid var(--border2); border-radius: 6px; padding: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.6); pointer-events: none; transition: opacity .12s; }}
.bbox:hover .tooltip {{ visibility: visible; opacity: 1; }}
.tt-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }}
.tt-badge {{ font-family: var(--mono); font-size: 10px; font-weight: 600; padding: 2px 7px; border-radius: 3px; color: #fff; }}
.tt-badge-yolo {{ background: #1f2937; border: 1px solid var(--border2); color: var(--muted); }}
.tt-changed {{ font-family: var(--mono); font-size: 9px; padding: 1px 5px; border-radius: 2px; background: #7c3aed; color: #fff; margin-left: auto; }}
.tt-agreed {{ font-family: var(--mono); font-size: 9px; padding: 1px 5px; border-radius: 2px; background: #065f46; color: #6ee7b7; margin-left: auto; }}

/* ── CONFIDENCE ROWS ── */
.conf-block {{ display: flex; flex-direction: column; gap: 5px; margin-bottom: 10px; }}
.conf-row {{ display: flex; align-items: center; gap: 7px; }}
.conf-label {{ font-family: var(--mono); font-size: 9px; color: var(--muted); width: 32px; flex-shrink: 0; text-transform: uppercase; }}
.conf-bar-wrap {{ flex: 1; height: 5px; background: var(--border); border-radius: 3px; overflow: hidden; }}
.conf-bar-fill {{ height: 100%; border-radius: 3px; transition: width .2s; }}
.conf-pct {{ font-family: var(--mono); font-size: 9px; color: var(--text); width: 30px; text-align: right; flex-shrink: 0; }}
.tt-divider {{ border: none; border-top: 1px solid var(--border); margin: 8px 0; }}
.tt-text {{ font-size: 10.5px; color: var(--muted); font-style: italic; max-height: 60px; overflow: hidden; line-height: 1.5; }}
</style>
</head>
<body>

<div class="topbar">
  <span class="topbar-brand">▸ Doc Graph Viewer</span>
  <div class="nav-btns">{nav_links}</div>
  <span class="topbar-page">Strana {page_number} / {total_pages}</span>
</div>

<div class="legend">
  <span class="legend-label">Triedy:</span>
  {legend_classes}
</div>

<div class="statsbar">
  {stats_html}
</div>

<div class="canvas-wrap">
  <div class="page-container">
    <img src="image_{page_number}.png" alt="Strana {page_number}">
    <svg class="edges-svg">
      <defs>
        <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
          <polygon points="0 0, 8 4, 0 8" fill="#ef4444" />
        </marker>
      </defs>
      {svg_edges_html}
    </svg>
    {boxes_html}
  </div>
</div>

</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# LEGENDA TRIED
# ─────────────────────────────────────────────────────────────────────────────
def build_legend_classes():
    parts = []
    for label, color in CATEGORY_COLORS.items():
        if label == "Unknown": continue
        parts.append(
            f'<span class="lchip">'
            f'<span class="lchip-dot" style="background:{color}"></span>{label}'
            f'</span>'
        )
    return " ".join(parts)


LEGEND_CLASSES_HTML = build_legend_classes()

# ─────────────────────────────────────────────────────────────────────────────
# HLAVNÁ SLUČKA
# ─────────────────────────────────────────────────────────────────────────────
total_pages = document_data["metadata"]["total_pages"]

for page_data in document_data["pages"]:
    page_number = page_data["page_number"]
    page_index = page_number - 1

    if page_index >= len(images): continue

    print(f"Generujem strana {page_number}...")

    img_path = os.path.join(output_dir, f"image_{page_number}.png")
    images[page_index].save(img_path)

    nodes_list = page_data.get("nodes", [])
    edges_list = page_data.get("edges", [])
    node_id_map = build_node_id_map(nodes_list)

    # ── ŠTATISTIKY ──
    reclassified = sum(1 for n in nodes_list if get_node_label(n)[0] != get_node_label(n)[1])
    agreed = len(nodes_list) - reclassified
    high_e = sum(1 for e in edges_list if e.get("probability", 0) >= 0.7)
    mid_e = sum(1 for e in edges_list if 0.5 <= e.get("probability", 0) < 0.7)
    low_e = sum(1 for e in edges_list if e.get("probability", 0) < 0.5)

    gat_confs = [c for n in nodes_list if (c := get_gat_conf(n)) is not None]
    yolo_confs = [c for n in nodes_list if isinstance(c := n.get("yolo_confidence"), float)]
    avg_gat = f"{sum(gat_confs) / len(gat_confs) * 100:.0f}%" if gat_confs else "—"
    avg_yolo = f"{sum(yolo_confs) / len(yolo_confs) * 100:.0f}%" if yolo_confs else "—"

    stats_html = (
        f'<div class="stat">Uzly: <b>{len(nodes_list)}</b></div>'
        f'<div class="stat" style="color:#22c55e">✓ Zhodli sa: <b>{agreed}</b></div>'
        f'<div class="stat" style="color:#a78bfa">✦ GAT zmenil: <b>{reclassified}</b></div>'
        f'<div class="stat">Hrany: <b>{len(edges_list)}</b></div>'
        f'<div class="stat stat-hi">▲ ≥70%: <b>{high_e}</b></div>'
        f'<div class="stat stat-mid">● 50–70%: <b>{mid_e}</b></div>'
        f'<div class="stat stat-lo">▼ &lt;50%: <b>{low_e}</b></div>'
    )

    # ── BOUNDING BOXY ──
    boxes_html = ""
    for node in nodes_list:
        norm_x1, norm_y1, norm_x2, norm_y2 = node["geometry"]["normalized_coords"]
        lp = norm_x1 * 100
        tp = norm_y1 * 100
        wp = (norm_x2 - norm_x1) * 100
        hp = (norm_y2 - norm_y1) * 100

        gat_label, yolo_label = get_node_label(node)
        changed = (gat_label != yolo_label)
        box_color = CATEGORY_COLORS.get(gat_label, CATEGORY_COLORS["Unknown"])

        yolo_conf_val = node.get("yolo_confidence")
        gat_conf_val = get_gat_conf(node)

        chip_html = f'<div class="bbox-chip">'
        chip_html += f'<span class="chip chip-gat" style="background:{box_color};">{"✦ " if changed else ""}{gat_label} {conf_pct(gat_conf_val)}</span>'
        if changed:
            yolo_color = CATEGORY_COLORS.get(yolo_label, CATEGORY_COLORS["Unknown"])
            chip_html += f'<span class="chip chip-yolo" style="background:{yolo_color};">↤ {yolo_label} {conf_pct(yolo_conf_val)}</span>'
        chip_html += "</div>"


        def bar(val, color):
            pct = int((val or 0) * 100)
            return f'<div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{pct}%;background:{color};"></div></div>'


        conf_block = '<div class="conf-block">'
        conf_block += f'<div class="conf-row"><span class="conf-label">GAT</span>{bar(gat_conf_val, box_color)}<span class="conf-pct">{conf_pct(gat_conf_val)}</span></div>'

        yolo_color_bar = CATEGORY_COLORS.get(yolo_label, CATEGORY_COLORS["Unknown"])
        conf_block += f'<div class="conf-row"><span class="conf-label">YOLO</span>{bar(yolo_conf_val, yolo_color_bar)}<span class="conf-pct">{conf_pct(yolo_conf_val)}</span></div>'
        conf_block += "</div>"

        status_badge = f'<span class="tt-changed">ZMENENÉ</span>' if changed else f'<span class="tt-agreed">ZHODLI SA</span>'
        extracted_text = html.escape(node.get("text", "")) or "<i>(bez textu)</i>"

        tooltip = (
            f'<div class="tooltip">'
            f'<div class="tt-header"><span class="tt-badge" style="background:{box_color};">{gat_label}</span><span class="tt-badge tt-badge-yolo">{yolo_label}</span>{status_badge}</div>'
            f'{conf_block}<hr class="tt-divider"><div class="tt-text">{extracted_text}</div>'
            f'</div>'
        )

        border_style = "dashed" if changed else "solid"
        boxes_html += f'<div class="bbox {"reclassified" if changed else ""}" style="left:{lp:.3f}%;top:{tp:.3f}%;width:{wp:.3f}%;height:{hp:.3f}%;border-color:{box_color};border-style:{border_style};">{chip_html}{tooltip}</div>\n'

    # ── SVG HRANY (OPRAVENÁ LOGIKA: Najlepšie hrany + Treshold + Správne súradnice) ──
    svg_edges_html = ""
    CAPTION_PARTNERS = {"Picture", "Table"}

    # 1. Odfiltrovanie a výber najlepšej hrany pre každý objekt
    best_edges_for_objects = {}

    for edge in edges_list:
        prob = edge.get("probability", 1.0)

        # Ignorujeme "smeti" – vykreslíme len hrany s istotou aspoň 40% (môžeš upraviť)
        if prob < 0.4: continue

        src_raw = edge.get("source", edge.get("from"))
        dst_raw = edge.get("target", edge.get("to"))

        if src_raw is None or dst_raw is None: continue
        src_idx = node_id_map.get(src_raw, src_raw) if isinstance(src_raw, int) else None
        dst_idx = node_id_map.get(dst_raw, dst_raw) if isinstance(dst_raw, int) else None

        if src_idx is None or dst_idx is None: continue
        if src_idx >= len(nodes_list) or dst_idx >= len(nodes_list): continue

        src_label = get_node_label(nodes_list[src_idx])[0]
        dst_label = get_node_label(nodes_list[dst_idx])[0]

        obj_idx = None
        cap_idx = None

        if src_label in CAPTION_PARTNERS and dst_label == "Caption":
            obj_idx, cap_idx = src_idx, dst_idx
        elif dst_label in CAPTION_PARTNERS and src_label == "Caption":
            obj_idx, cap_idx = dst_idx, src_idx

        # Uložíme len tú najsilnejšiu väzbu pre daný obrázok/tabuľku
        if obj_idx is not None:
            if obj_idx not in best_edges_for_objects or prob > best_edges_for_objects[obj_idx]["prob"]:
                best_edges_for_objects[obj_idx] = {
                    "src": obj_idx,  # Smerujeme šípku OD objektu
                    "dst": cap_idx,  # K textu (caption)
                    "prob": prob
                }

    # 2. Vykreslenie vyfiltrovaných a očistených hrán
    for obj_idx, edge_data in best_edges_for_objects.items():
        src_node = nodes_list[edge_data["src"]]
        dst_node = nodes_list[edge_data["dst"]]
        prob = edge_data["prob"]

        sc = src_node["geometry"]["normalized_center"]
        dc = dst_node["geometry"]["normalized_center"]

        x1, y1 = sc[0] * 100, sc[1] * 100
        x2, y2 = dc[0] * 100, dc[1] * 100
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2

        opacity = round(max(0.4, prob), 2)

        # Čiara zakončená šípkou
        svg_edges_html += (
            f'<line x1="{x1}%" y1="{y1}%" x2="{x2}%" y2="{y2}%" '
            f'stroke="#ef4444" stroke-width="2.5" stroke-dasharray="6,4" '
            f'opacity="{opacity}" stroke-linecap="round" marker-end="url(#arrowhead)"/>\n'
        )

        # Tmavé pozadie a biely text s percentami v strede šípky
        prob_text = f"{prob * 100:.0f}%"
        svg_edges_html += (
            f'<rect x="calc({mx}% - 16px)" y="calc({my}% - 10px)" width="32" height="20" rx="4" fill="#0d1117" opacity="0.85" />\n'
            f'<text x="{mx}%" y="{my}%" fill="#fca5a5" font-size="11px" font-family="IBM Plex Mono, monospace" font-weight="600" '
            f'text-anchor="middle" dominant-baseline="central">{prob_text}</text>\n'
        )

    # ── NAVIGÁCIA ──
    prev_link = f'<a href="page_{page_number - 1}.html">&larr; Predch.</a>' if page_number > 1 else ""
    next_link = f'<a href="page_{page_number + 1}.html">Ďalšia &rarr;</a>' if page_number < total_pages else ""

    final_html = HTML_TEMPLATE.format(
        page_number=page_number,
        total_pages=total_pages,
        nav_links=f"{prev_link} {next_link}",
        legend_classes=LEGEND_CLASSES_HTML,
        stats_html=stats_html,
        svg_edges_html=svg_edges_html,
        boxes_html=boxes_html,
    )

    html_path = os.path.join(output_dir, f"page_{page_number}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(final_html)

print(f"\nHotovo! Uložené v: '{output_dir}'")