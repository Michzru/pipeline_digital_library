from .pipelines.document_pipeline import run_document_pipeline

def run_pipeline(file_path: str, pipeline: str = "document", verbose: bool = True, gpu: bool = True) -> dict:
    """
    Main entry point for running the pipeline.

    Args:
        file_path (str): Path to the input file.
        pipeline (str): Name of the pipeline to run.
        verbose (bool): If False, output will be printed to stdout.
        gpu (bool): If True, uses GPU acceleration where applicable.
    """

    if pipeline == "document":
        return run_document_pipeline(file_path, verbose=verbose, gpu=gpu)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")