from .pipelines.document_pipeline import run_document_pipeline

def run_pipeline(file_path: str, pipeline: str = "document") -> dict:
    if pipeline == "document":
        return run_document_pipeline(file_path)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")