import os
from pdf2image import convert_from_path

# Helper Function to convert pdf into images
def get_png_images(pdf_path, thread_count=4, verbose=True):
    if verbose:
        print(f"[PDF] Converting: {pdf_path}, number of threads: {thread_count}")

    images = convert_from_path(
        pdf_path,
        dpi=300,
        thread_count=thread_count
    )

    filename = os.path.basename(pdf_path)

    if verbose:
        print(f"[PDF] Total pages: {len(images)}")
        print(f"[PDF] Done")

    return images, filename