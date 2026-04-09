import torch
import sys
import os
from functools import partial
from torch import nn
import tokenizers as tk

from config import (
    UNITABLE_DIR,
    UNITABLE_STRUCTURE_WEIGHTS,
    UNITABLE_BBOX_WEIGHTS
)

sys.path.insert(0, UNITABLE_DIR)

from models.unitable.src.model import EncoderDecoder, ImgLinearBackbone, Encoder, Decoder

D_MODEL = 768
PATCH_SIZE = 16
NHEAD = 12
DROPOUT = 0.2

_models = {}

WEIGHTS = {
    "structure": UNITABLE_STRUCTURE_WEIGHTS,
    "bbox": UNITABLE_BBOX_WEIGHTS
}

VOCAB_FILES = {
    "structure": os.path.join(UNITABLE_DIR, "vocab", "vocab_html.json"),
    "bbox": os.path.join(UNITABLE_DIR, "vocab", "vocab_bbox.json"),
}

MAX_SEQ_LEN = {
    "structure": 784,
    "bbox": 1024,
}


def _build_model(vocab_size: int, max_seq_len: int) -> EncoderDecoder:
    backbone = ImgLinearBackbone(d_model=D_MODEL, patch_size=PATCH_SIZE)

    encoder = Encoder(
        d_model=D_MODEL,
        nhead=NHEAD,
        dropout=DROPOUT,
        activation="gelu",
        norm_first=True,
        nlayer=12,
        ff_ratio=4,
    )

    decoder = Decoder(
        d_model=D_MODEL,
        nhead=NHEAD,
        dropout=DROPOUT,
        activation="gelu",
        norm_first=True,
        nlayer=4,
        ff_ratio=4,
    )

    model = EncoderDecoder(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        vocab_size=vocab_size,
        d_model=D_MODEL,
        padding_idx=0,
        max_seq_len=max_seq_len,   # ✅ FIX
        dropout=DROPOUT,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    return model


def get_unitable_model(task="structure", verbose=True, gpu=True):
    global _models

    if task not in _models:
        if gpu:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")

        vocab = tk.Tokenizer.from_file(VOCAB_FILES[task])
        vocab_size = vocab.get_vocab_size()

        # Build model
        model = _build_model(vocab_size, MAX_SEQ_LEN[task])
        model.padding_idx = vocab.token_to_id("<pad>")

        state = torch.load(WEIGHTS[task], map_location="cpu", weights_only=True)
        model.load_state_dict(state)

        model.eval()
        model.to(device)

        _models[task] = (vocab, model, device)

        if verbose:
            dev_type = "MPS (Apple Silicon)" if device.type == "mps" else device.type.upper()
            print(f"UniTable [{task}] loaded on {dev_type}.")

    return _models[task]