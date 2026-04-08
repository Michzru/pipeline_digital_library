from config import TRANSFORMER_MODEL_NAME

_model = None

def get_transformer_model(verbose, gpu=True):
    global _model

    if _model is None:
        from sentence_transformers import SentenceTransformer

        if gpu:
            _model = SentenceTransformer(TRANSFORMER_MODEL_NAME, device='cuda')
            if verbose:
                print("TRANSFORMER model loaded on GPU.")
        else:
            _model = SentenceTransformer(TRANSFORMER_MODEL_NAME, device='cpu')
            if not verbose:
                print("TRANSFORMER model loaded on CPU.")

    return _model