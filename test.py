#from models.transformer import get_transformer_model
#from models.yolo import get_yolo_model
#from models.ocr import get_ocr_model
#from models.gat import get_gat_model

#model = get_yolo_model(gpu=True)

#print(model)

# model = get_ocr_model('en', gpu=True)
# print(model)

# model = get_transformer_model()
# print(model)

# model = get_gat_model(gpu=True)
# print(model)

# from pipelines.document_pipeline import run_document_pipeline
#
# if __name__ == "__main__":
#     run_document_pipeline('../data/Document.pdf')

# from pipelines.test_pipeline import run_test_pipeline
#
# print(run_test_pipeline('dbbf8dde-1d40-481b-ba5f-4d84c2de3e54.pdf', True))

from core import run_pipeline

if __name__ == "__main__":
    # Cesta k tvojmu testovaciemu PDF
    cesta_k_pdf = "dbbf8dde-1d40-481b-ba5f-4d84c2de3e54.pdf"

    print("Spúšťam pipeline...")

    # Zavolanie entry pointu
    vysledok = run_pipeline(
        file_path=cesta_k_pdf,
        pipeline="document",
        verbose=True,  # Zapne vypisovanie do konzoly a tqdm progress bary
        gpu=True  # Zapne využitie GPU pre YOLO a GAT
    )

    print("Hotovo! Pipeline zbehla úspešne.")