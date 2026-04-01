from models.transformer import get_transformer_model
from models.yolo import get_yolo_model
from models.ocr import get_ocr_model
from models.gat import get_gat_model

# model = get_yolo_model()
#
# print(model)

# model = get_ocr_model('en', gpu=False)
# print(model)

# model = get_transformer_model(gpu=False)
# print(model)

# model = get_gat_model(gpu=False)
# print(model)

from pipelines.document_pipeline import run_document_pipeline

if __name__ == "__main__":
    run_document_pipeline('../data/Document.pdf')
