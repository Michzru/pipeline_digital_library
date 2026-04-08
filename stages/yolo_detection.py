from models.yolo import get_yolo_model
from tqdm import tqdm

def run_yolo_detection(pngs, filename, verbose, gpu):
    model = get_yolo_model(verbose=verbose, gpu=gpu)

    # Data Structure
    document_data = {
        "metadata": {
            "filename": filename,
            "total_pages": len(pngs)
        },
        "pages": []
    }

    iterator = tqdm(
        enumerate(pngs),
        total=len(pngs),
        desc="YOLO pages",
        leave=False,
        disable=not verbose
    )

    # Run YOLO on each page
    for page_idx, pil_image in iterator:
        if verbose:
            iterator.set_postfix(page=page_idx + 1)

        page_width, page_height = pil_image.size

        page_data = {
            "page_number": page_idx + 1,
            "width": page_width,
            "height": page_height,
            "nodes": [],
            "edges": []
        }

        # Perform prediction
        yolo_results = model.predict(
            source=pil_image,
            imgsz=1120,
            conf=0.2,
            iou=0.5,
            agnostic_nms=True,
            verbose=False
        )

        # Grab the results for this specific image
        yolo_result = yolo_results[0]

        valid_boxes = []

        # Prepare boxes
        for box in yolo_result.boxes:
            # Get coordinates in xyxy format
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            category_id = int(box.cls[0].item())
            class_name = yolo_result.names[category_id]
            confidence = box.conf[0].item()

            valid_boxes.append({
                "coords": [x1, y1, x2, y2],
                "class_id": category_id,
                "class_name": class_name,
                "confidence": confidence
            })

        # Filtering, keeping larger element
        boxes_to_remove = set()
        threshold = 0.80

        for i in range(len(valid_boxes)):
            if i in boxes_to_remove: continue

            for j in range(i + 1, len(valid_boxes)):
                if j in boxes_to_remove: continue

                box_i = valid_boxes[i]["coords"]
                box_j = valid_boxes[j]["coords"]

                # Bboxes of intersection
                x_left = max(box_i[0], box_j[0])
                y_top = max(box_i[1], box_j[1])
                x_right = min(box_i[2], box_j[2])
                y_bottom = min(box_i[3], box_j[3])

                # If not intersected continue
                if x_right < x_left or y_bottom < y_top:
                    continue

                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
                area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])

                # IoA (Intersection over Area) pre oba boxy
                ioa_i = intersection_area / area_i if area_i > 0 else 0
                ioa_j = intersection_area / area_j if area_j > 0 else 0

                if ioa_i > threshold and ioa_j > threshold:
                    # If nearly identical on the same place delete the one with less confidence
                    if valid_boxes[i]["confidence"] > valid_boxes[j]["confidence"]:
                        boxes_to_remove.add(j)
                    else:
                        boxes_to_remove.add(i)
                        break
                elif ioa_i > threshold:
                    # Box "i" is smaller and consumed by box j. Delede i
                    boxes_to_remove.add(i)
                    break
                elif ioa_j > threshold:
                    # other way around
                    boxes_to_remove.add(j)

        # Final list
        filtered_boxes = [box for i, box in enumerate(valid_boxes) if i not in boxes_to_remove]

        # Make node
        for node_id, box_data in enumerate(filtered_boxes):
            x1, y1, x2, y2 = box_data["coords"]

            # Normalization of coords 0 to 1
            norm_x1 = x1 / page_width
            norm_y1 = y1 / page_height
            norm_x2 = x2 / page_width
            norm_y2 = y2 / page_height

            # Center and params of the rectangle
            center_x = (norm_x1 + norm_x2) / 2
            center_y = (norm_y1 + norm_y2) / 2
            width = norm_x2 - norm_x1
            height = norm_y2 - norm_y1

            extracted_text = ""

            node = {
                "node_id": node_id,
                "label": box_data["class_name"],
                "label_id": box_data["class_id"],
                "yolo_confidence": box_data["confidence"],
                "text": extracted_text.strip(),
                "geometry": {
                    "absolute_pixel_coords": [int(x1), int(y1), int(x2), int(y2)],
                    "normalized_coords": [norm_x1, norm_y1, norm_x2, norm_y2],
                    "normalized_center": [center_x, center_y],
                    "normalized_size": [width, height]
                }
            }
            page_data["nodes"].append(node)

        document_data["pages"].append(page_data)

    return document_data