
# Helper Function to convert pdf into images
def get_png_images(pdf_path):
    import os
    from pdf2image import convert_from_path

    print("Converting PDF to images...")
    images = convert_from_path(pdf_path, dpi=300)
    filename = os.path.basename(pdf_path)

    return images, filename

