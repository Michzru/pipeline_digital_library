_model = None

def get_ocr_model(language='sk', gpu=True):
    global _model

    if _model is None:
        from paddleocr import PaddleOCR

        if gpu:
            # GPU model
            _model = PaddleOCR(use_angle_cls=True, show_log=False, lang=language, use_gpu=gpu)

        else:
            # CPU model
            _model = PaddleOCR(use_angle_cls=True, show_log=False, lang=language, use_gpu=gpu)

    return _model