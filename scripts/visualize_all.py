import os
import json
import joblib
import html

# Konfigurácia
json_path = "data/document_graph_with_edges.json"
cache_path = "data/cached_images.joblib"
output_dir = "data/interactive_pages2"

os.makedirs(output_dir, exist_ok=True)

print("Načítavam JSON a obrázky z cache...")
with open(json_path, 'r', encoding='utf-8') as f:
    document_data = json.load(f)

images = joblib.load(cache_path)

# Slovník farieb pre jednotlivé triedy
CATEGORY_COLORS = {
    "Caption": "#9c27b0",  # Fialová
    "Picture": "#ff9800",  # Oranžová
    "Table": "#2196f3",  # Modrá
    "Formula": "#00bcd4",  # Tyrkysová
    "Section-header": "#e91e63",  # Ružová
    "Page-footer": "#607d8b",  # Modro-šedá
    "Page-header": "#795548",  # Hnedá
    "Other": "#9e9e9e",  # Šedá
    "Unknown": "#000000"  # Čierna (fallback)
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="sk">
<head>
    <meta charset="UTF-8">
    <title>Strana {page_number}</title>
    <style>
        body {{ background-color: #f0f0f0; padding: 20px; font-family: sans-serif; }}
        .page-container {{ position: relative; display: inline-block; box-shadow: 0 4px 8px rgba(0,0,0,0.2); background: white; margin-top: 10px; }}
        .page-container img {{ display: block; max-width: 1000px; height: auto; }}
        .edges-overlay {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 5; }}

        /* Bounding Boxy */
        .bbox {{ position: absolute; border: 3px solid; cursor: pointer; box-sizing: border-box; transition: background-color 0.2s; z-index: 6; }}
        .bbox:hover {{ background-color: rgba(255, 255, 255, 0.3); z-index: 10; }}

        /* Permanentný štítok viditeľný na boxe */
        .bbox-label {{ position: absolute; top: -20px; left: -3px; color: white; font-size: 11px; font-weight: bold; padding: 2px 6px; border-radius: 4px 4px 0 0; white-space: nowrap; z-index: 11; box-shadow: 0 2px 4px rgba(0,0,0,0.3); }}

        /* Špeciálny štýl ak GAT opravil YOLO */
        .label-fixed {{ border: 2px dashed #000; }}

        /* Tooltip (zobrazí sa pri hover) */
        .tooltip {{ visibility: hidden; position: absolute; top: 100%; left: 0; background-color: #222; color: #fff; padding: 12px; border-radius: 6px; width: max-content; max-width: 450px; white-space: pre-wrap; box-shadow: 0 4px 10px rgba(0,0,0,0.5); font-size: 13px; opacity: 0; transition: opacity 0.2s; pointer-events: none; line-height: 1.5; z-index: 12; }}
        .bbox:hover .tooltip {{ visibility: visible; opacity: 1; }}

        .nav {{ margin-bottom: 20px; }}
        .nav a {{ padding: 10px 15px; background: #333; color: white; text-decoration: none; border-radius: 5px; margin-right: 10px; }}
        .nav a:hover {{ background: #555; }}

        /* Text pre hrany */
        .edge-text {{ fill: #000; font-size: 14px; font-weight: bold; text-shadow: 2px 2px 0 #fff, -2px -2px 0 #fff, 2px -2px 0 #fff, -2px 2px 0 #fff, 0 2px 0 #fff, 0 -2px 0 #fff, -2px 0 0 #fff, 2px 0 0 #fff; z-index: 20; }}
    </style>
</head>
<body>
    <div class="nav">{nav_links}</div>
    <h2>Strana {page_number} / {total_pages}</h2>
    <div class="page-container">
        <img src="image_{page_number}.png" alt="Strana {page_number}">
        <svg class="edges-overlay">{svg_edges_html}</svg>
        {boxes_html}
    </div>
</body>
</html>
"""

total_pages = document_data["metadata"]["total_pages"]

for page_data in document_data["pages"]:
    page_number = page_data["page_number"]
    page_index = page_number - 1

    if page_index >= len(images):
        continue

    print(f"Generujem HTML stranu {page_number}...")

    img_filename = f"image_{page_number}.png"
    img_path = os.path.join(output_dir, img_filename)
    images[page_index].save(img_path)

    boxes_html = ""
    nodes_list = page_data.get("nodes", [])

    # 1. Generovanie bounding boxov
    for node in nodes_list:
        norm_x1, norm_y1, norm_x2, norm_y2 = node["geometry"]["normalized_coords"]
        left_pct = norm_x1 * 100
        top_pct = norm_y1 * 100
        width_pct = (norm_x2 - norm_x1) * 100
        height_pct = (norm_y2 - norm_y1) * 100

        # Bezpečné vytiahnutie hodnôt pre YOLO a GAT
        yolo_label = node.get("label", "Unknown")
        yolo_conf_val = node.get("yolo_confidence")
        yolo_conf_str = f"{yolo_conf_val:.4f}" if isinstance(yolo_conf_val, (float, int)) else "N/A"

        gat_label = node.get("gat_reclassified_label", yolo_label)  # fallback na YOLO ak chýba
        gat_conf_val = node.get("gat_confidence")
        gat_conf_str = f"{gat_conf_val:.4f}" if isinstance(gat_conf_val, (float, int)) else "N/A"

        # Určenie farby pre GAT triedu
        box_color = CATEGORY_COLORS.get(gat_label, CATEGORY_COLORS["Unknown"])
        extracted_text = html.escape(node.get("text", "")) if node.get("text") else "<i>(Bez textu)</i>"

        # Logika: Zhodli sa alebo GAT opravil?
        if gat_label != yolo_label:
            visible_title = f"✨ {gat_label} ({gat_conf_str}) [bolo: {yolo_label}]"
            extra_class = "label-fixed"
        else:
            visible_title = f"✓ {gat_label} ({gat_conf_str})"
            extra_class = ""

        # Skladanie HTML pre box
        box_div = f"""
        <div class="bbox" style="left: {left_pct}%; top: {top_pct}%; width: {width_pct}%; height: {height_pct}%; border-color: {box_color};">
            <div class="bbox-label {extra_class}" style="background-color: {box_color};">
                {visible_title}
            </div>
            <div class="tooltip">
                <strong style="color: #4CAF50;">GAT (Finálny):</strong> {gat_label} (Conf: {gat_conf_str})<br>
                <strong style="color: #FF9800;">YOLO (Pôvodný):</strong> {yolo_label} (Conf: {yolo_conf_str})<br>
                <hr style="border: 0; border-top: 1px solid #555; margin: 8px 0;">
                {extracted_text}
            </div>
        </div>
        """
        boxes_html += box_div

    # 2. Generovanie hrán (Edges)
    svg_edges_html = ""
    edges_list = page_data.get("edges", [])
    best_edges_for_objects = {}

    for edge in edges_list:
        src_idx = edge.get("source", edge.get("from"))
        dst_idx = edge.get("target", edge.get("to"))
        prob = edge.get("probability", 1.0)

        if src_idx is None or dst_idx is None or src_idx >= len(nodes_list) or dst_idx >= len(nodes_list):
            continue

        label_src = nodes_list[src_idx].get("gat_reclassified_label", nodes_list[src_idx].get("label"))
        label_dst = nodes_list[dst_idx].get("gat_reclassified_label", nodes_list[dst_idx].get("label"))

        obj_idx = None
        cap_idx = None

        # Zisťujeme vzťahy Picture->Caption alebo Table->Caption
        if label_src in ["Picture", "Table"] and label_dst == "Caption":
            obj_idx, cap_idx = src_idx, dst_idx
        elif label_dst in ["Picture", "Table"] and label_src == "Caption":
            obj_idx, cap_idx = dst_idx, src_idx

        if obj_idx is not None and prob > 0.3:
            if obj_idx not in best_edges_for_objects or prob > best_edges_for_objects[obj_idx]["probability"]:
                best_edges_for_objects[obj_idx] = {
                    "source": obj_idx,
                    "target": cap_idx,
                    "probability": prob
                }

    # Vykreslenie vyfiltrovaných hrán
    for obj_idx, best_edge in best_edges_for_objects.items():
        src_node = nodes_list[best_edge["source"]]
        dst_node = nodes_list[best_edge["target"]]

        src_cx, src_cy = src_node["geometry"]["normalized_center"]
        dst_cx, dst_cy = dst_node["geometry"]["normalized_center"]

        x1_pct = src_cx * 100
        y1_pct = src_cy * 100
        x2_pct = dst_cx * 100
        y2_pct = dst_cy * 100

        line_html = f'<line x1="{x1_pct}%" y1="{y1_pct}%" x2="{x2_pct}%" y2="{y2_pct}%" stroke="#e02424" stroke-width="4" stroke-dasharray="5,5" opacity="0.9" />\n'

        prob_text = f"Edge: {(best_edge['probability'] * 100):.1f}%"
        mid_x = (x1_pct + x2_pct) / 2
        mid_y = (y1_pct + y2_pct) / 2

        line_html += f'<text class="edge-text" x="{mid_x}%" y="{mid_y}%" text-anchor="middle" dominant-baseline="middle">{prob_text}</text>\n'
        svg_edges_html += line_html

    # Navigácia strán
    prev_link = f'<a href="page_{page_number - 1}.html">&laquo; Predchádzajúca</a>' if page_number > 1 else ''
    next_link = f'<a href="page_{page_number + 1}.html">Ďalšia &raquo;</a>' if page_number < total_pages else ''
    nav_links = f"{prev_link} {next_link}"

    final_html = HTML_TEMPLATE.format(
        page_number=page_number,
        total_pages=total_pages,
        nav_links=nav_links,
        boxes_html=boxes_html,
        svg_edges_html=svg_edges_html
    )

    html_filepath = os.path.join(output_dir, f"page_{page_number}.html")
    with open(html_filepath, 'w', encoding='utf-8') as f:
        f.write(final_html)

print(f"\nHotovo! Uložené v priečinku: '{output_dir}'")