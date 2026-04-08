_model = None

def get_ocr_model(verbose, gpu=True):
    global _model

    if _model is None:
        import easyocr
        _model = easyocr.Reader(['sk', 'cs', 'en'], gpu=gpu, verbose=verbose)
        if verbose:
            print(f"OCR model loaded on {'GPU' if gpu else 'CPU'}.")

    return _model