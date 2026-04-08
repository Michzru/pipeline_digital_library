# Helper function to calculate iou over two boxes
def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection_area / float(box1_area + box2_area - intersection_area)

def safe_crop(image, bbox):
    x1, y1, x2, y2 = bbox

    if x2 <= x1 or y2 <= y1:
        return None

    return image.crop((x1, y1, x2, y2))

# Crop table from image with margin
def crop_with_margin(image, bbox, margin_ratio=0.05):
    x1, y1, x2, y2 = bbox
    w, h = image.size

    bw = x2 - x1
    bh = y2 - y1

    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)

    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)

    return image.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)