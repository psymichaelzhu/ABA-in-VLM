"""Run image, ratings/subset, and visual feature preprocessing."""

# %% Imports
from script.preprocessing.image_standardization import standardize_images
from script.preprocessing.rating_cleaning import clean_ratings
from script.preprocessing.visual_feature_extraction import extract_visual_features


# %% Pipeline
def run_preprocessing(imageset: str):
    """Prepare standardized images, ratings, subsets, and visual features."""
    standardize_images(imageset)
    clean_ratings(imageset)
    extract_visual_features(imageset)


# %% Run
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("imageset", choices=["OASIS", "EmoMadrid","NAPS"])
    args = parser.parse_args()
    run_preprocessing(args.imageset)
