def save_png_images(images, filename, cache_dir="../data/cache", verbose=False):
    import os

    os.makedirs(cache_dir, exist_ok=True)

    base_name = os.path.splitext(filename)[0]
    folder = os.path.join(cache_dir, base_name)

    os.makedirs(folder, exist_ok=True)

    for i, img in enumerate(images):
        path = os.path.join(folder, f"page_{i+1}.png")
        img.save(path)

    if verbose:
        print(f"[CACHE] Saved images to {folder}")

    return folder


def load_png_images(folder, verbose=False):
    from PIL import Image
    import os
    import re

    # Získame zoznam a zoradíme ho podľa čísla v názve
    files = [f for f in os.listdir(folder) if f.endswith(".png")]

    # Funkcia vytiahne číslo zo stringu "page_12.png" -> 12
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    images = []
    for f in files:
        path = os.path.join(folder, f)
        images.append(Image.open(path))

    if verbose:
        print(f"[CACHE] Loaded {len(images)} images from {folder}")

    return images