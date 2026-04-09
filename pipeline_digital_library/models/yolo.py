from config import YOLO_MODEL_PATH
import torch

_model = None

def get_yolo_model(verbose, gpu=True):
    global _model

    if _model is None:
        from doclayout_yolo import YOLOv10

        model = YOLOv10(YOLO_MODEL_PATH)

        if gpu:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        else:
            device = 'cpu'

        _model = model.to(device)

        if verbose:
            print(f"YOLO model loaded on {device.upper()}.")

    return _model