"""Standardize raw images for an affective imageset."""

# %% Imports
from PIL import Image
from tqdm import tqdm

from config import PROJECT_ROOT, STANDARD_IMAGE_SIZE

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
STANDARDIZE_PIPELINE = {
    "OASIS": [],
    "EmoMadrid": ["crop_black_borders"],
}


# %% Image processing
def _crop_black_borders(image: Image.Image) -> Image.Image:
    """Crop black borders using the nonblack bounding box."""
    bbox = image.convert("L").getbbox()
    return image.crop(bbox) if bbox else image


def standardize_images(imageset: str):
    """Process raw images into standardized 224x224 JPG files."""
    raw_dir = PROJECT_ROOT / "stimuli" / imageset / "raw" / "images"
    output_dir = PROJECT_ROOT / "stimuli" / imageset / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in raw_dir.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for path in tqdm(image_paths, desc=f"Standardizing {imageset}", unit="image"):
        with Image.open(path) as image:
            if "crop_black_borders" in STANDARDIZE_PIPELINE.get(imageset, []):
                image = _crop_black_borders(image)
            image = image.resize(STANDARD_IMAGE_SIZE, Image.Resampling.LANCZOS)
            image = image.convert("RGB")
            image.save(output_dir / f"{path.stem}.jpg", "JPEG", quality=95)


# %% Run
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("imageset")
    args = parser.parse_args()
    standardize_images(args.imageset)
