_model = None


def get_ocr_model(verbose, gpu=True):
    global _model

    if _model is None:
        import easyocr
        import torch

        is_mps = torch.backends.mps.is_available()

        actual_gpu = gpu and (torch.cuda.is_available() or is_mps)

        _model = easyocr.Reader(['sk', 'cs', 'en'], gpu=actual_gpu, verbose=verbose)

        if verbose:
            device_name = "CPU"
            if actual_gpu:
                device_name = "MPS (Apple Silicon)" if is_mps else "CUDA"
            print(f"OCR model loaded on {device_name}.")

    return _model