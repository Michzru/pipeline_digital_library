from models.yolo import get_yolo_model
from utils.iou import calculate_iou

def run_yolo_detection(pngs, filename):
    model = get_yolo_model()

    # Data Structure
    document_data = {
        "metadata": {
            "filename": filename,
            "total_pages": len(pngs)
        },
        "pages": []
    }

    # Run YOLO on each page
    for page_idx, pil_image in enumerate(pngs):
        print(f"\nProcessing page {page_idx + 1}/{len(pngs)}...")
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
            agnostic_nms=True
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

            # ******
            # TREBA VYMAZAT TOTEN QR CODE
            # ******

            # Remove QR code in the left corner
            is_in_left_zone = x2 < (page_width * 0.1)
            is_in_bottom_zone = y1 > (page_height * 0.90)

            if class_name == "Picture" and is_in_left_zone and is_in_bottom_zone:
                print(f" -> Ignoring image/QR code in bottom left corner: {x1:.0f}, {y1:.0f})")
                continue

            valid_boxes.append({
                "coords": [x1, y1, x2, y2],
                "class_id": category_id,
                "class_name": class_name,
                "confidence": confidence
            })

        # Make node and map text to correct node
        for node_id, box_data in enumerate(valid_boxes):
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