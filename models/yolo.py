from config import YOLO_MODEL_PATH

_model = None

def get_yolo_model(gpu=True):
    global _model

    if _model is None:
        from doclayout_yolo import YOLOv10

        model = YOLOv10(YOLO_MODEL_PATH)
        if gpu:
            _model = model.to('cuda')
            print("YOLO model load on GPU.")
        else:
            _model = model.to('cpu')
            print("YOLO model load on CPU.")

    return _model