from config import TRANSFORMER_MODEL_NAME

_model = None

def get_transformer_model(gpu=True):
    global _model

    if _model is None:
        from sentence_transformers import SentenceTransformer

        if gpu:
            _model = SentenceTransformer(TRANSFORMER_MODEL_NAME, device='cuda')
        else:
            _model = SentenceTransformer(TRANSFORMER_MODEL_NAME, device='cpu')

    return _model