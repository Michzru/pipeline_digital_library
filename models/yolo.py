from config import YOLO_MODEL_PATH

_model = None

def get_yolo_model(verbose, gpu=True):
    global _model

    if _model is None:
        from doclayout_yolo import YOLOv10

        model = YOLOv10(YOLO_MODEL_PATH)
        if gpu:
            _model = model.to('cuda')
            if verbose:
                print("YOLO model loaded on GPU.")
        else:
            _model = model.to('cpu')
            if not verbose:
                print("YOLO model loaded on CPU.")

    return _model