from tqdm import tqdm
from models.ocr import get_ocr_model
import numpy as np

def run_text_detection(pngs, data, verbose, gpu):
    model = get_ocr_model(verbose=verbose, gpu=gpu)

    iterator = tqdm(
        enumerate(pngs),
        total=len(pngs),
        desc="OCR pages",
        leave=False,
        disable=not verbose
    )

    for page_idx, pil_image in iterator:
        if verbose:
            iterator.set_postfix(page=page_idx + 1)


        # Acquire data for particular page from yolo extractions
        page_data = data["pages"][page_idx]

        # Full page ocr, faster and more accurate
        page_image_np = np.array(pil_image)
        full_page_ocr_results = model.readtext(page_image_np)

        if not full_page_ocr_results or full_page_ocr_results[0] is None:
            continue

        # Go through existing yolo nodes and add the text extracted using OCR, can be modified to extract directly from PDF
        for node in page_data["nodes"]:
            # Get coordinates from yolo box in pixels
            x1, y1, x2, y2 = node["geometry"]["absolute_pixel_coords"]

            node_texts = []
            margin = 10

            for line in full_page_ocr_results:
                ocr_box, text, confidence = line

                ocr_center_x = sum(pt[0] for pt in ocr_box) / 4.0
                ocr_center_y = sum(pt[1] for pt in ocr_box) / 4.0

                # Safety if the center of OCR is in the center of YOLO box
                if (x1 - margin <= ocr_center_x <= x2 + margin) and \
                        (y1 - margin <= ocr_center_y <= y2 + margin):
                    node_texts.append(text)

            node["text"] = " ".join(node_texts).strip()

    return data