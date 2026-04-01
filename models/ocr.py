from config import YOLO_MODEL_PATH

_model = None

def get_ocr_model(language='sk', gpu=True):
    global _model

    if _model is None:
        from paddleocr import PaddleOCR

        if gpu:
            # GPU odel
            _ocr_model = PaddleOCR(use_angle_cls=True, show_log=False, lang=language, use_gpu=True)

        else:
            # CPU model
            _model = PaddleOCR(use_angle_cls=True, show_log=False, lang=language)

    return _model