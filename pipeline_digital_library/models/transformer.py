from config import TRANSFORMER_MODEL_NAME
import torch

_model = None


def get_transformer_model(verbose, gpu=True):
    global _model

    if _model is None:
        from sentence_transformers import SentenceTransformer

        if gpu:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        else:
            device = 'cpu'

        _model = SentenceTransformer(TRANSFORMER_MODEL_NAME, device=device)

        if verbose:
            print(f"TRANSFORMER model loaded on {device.upper()}.")

    return _model