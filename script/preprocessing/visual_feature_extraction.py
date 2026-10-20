"""Extract visual statistics from standardized images."""

# %% Imports
import io

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import PROJECT_ROOT


# %% Color conversion
def _rgb_to_lab(rgb_arr: np.ndarray) -> np.ndarray:
    """Convert sRGB pixel values (0-255) to CIELAB using D65 white point."""
    rgb = rgb_arr.astype(np.float64) / 255.0
    rgb_linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )

    rgb_to_xyz = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = rgb_linear @ rgb_to_xyz.T
    xyz_scaled = xyz / np.array([0.95047, 1.00000, 1.08883])

    delta = 6 / 29
    f = np.where(
        xyz_scaled > delta ** 3,
        np.cbrt(xyz_scaled),
        xyz_scaled / (3 * delta ** 2) + 4 / 29,
    )
    L = 116 * f[:, :, 1] - 16
    a = 500 * (f[:, :, 0] - f[:, :, 1])
    b = 200 * (f[:, :, 1] - f[:, :, 2])
    return np.stack([L, a, b], axis=2)


# %% Image statistics
def extract_image_physical_properties(image_path) -> dict:
    """Compute grayscale, JPEG, CIELAB, and RGB image properties."""
    with Image.open(image_path) as image:
        img = image.convert("RGB")

    gray_arr = np.asarray(img.convert("L"), dtype=np.float64)
    luminance = float(gray_arr.mean())
    contrast = float(gray_arr.std())

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=80)
    jpeg_size = buffer.tell()

    hist, _ = np.histogram(gray_arr.ravel(), bins=256, range=(0, 256))
    prob = hist / hist.sum()
    nonzero = prob[prob > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))

    rgb_arr = np.asarray(img, dtype=np.float64)
    lab_arr = _rgb_to_lab(rgb_arr)

    return {
        "Luminance": luminance,
        "Contrast": contrast,
        "JpegSize": jpeg_size,
        "Entropy": entropy,
        "Ldim_mean": float(lab_arr[:, :, 0].mean()),
        "Adim_mean": float(lab_arr[:, :, 1].mean()),
        "Bdim_mean": float(lab_arr[:, :, 2].mean()),
        "Rch_mean": float(rgb_arr[:, :, 0].mean()),
        "Gch_mean": float(rgb_arr[:, :, 1].mean()),
        "Bch_mean": float(rgb_arr[:, :, 2].mean()),
    }


# %% Feature extraction
def extract_visual_features(imageset: str):
    """Write image-level visual statistics to visual_features.csv."""
    image_dir = PROJECT_ROOT / "stimuli" / imageset / "images"
    image_paths = sorted(image_dir.glob("*.jpg"))

    rows = []
    for path in tqdm(image_paths, desc=f"Visual features: {imageset}", unit="image"):
        rows.append({"Image_id": path.stem, **extract_image_physical_properties(path)})

    output_path = PROJECT_ROOT / "stimuli" / imageset / "visual_features.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)


# %% Run
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("imageset")
    args = parser.parse_args()
    extract_visual_features(args.imageset)
