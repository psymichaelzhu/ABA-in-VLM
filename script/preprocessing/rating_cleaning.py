"""Clean affective ratings and update category-derived subsets."""

# %% Imports
import pandas as pd

from config import PROJECT_ROOT


# %% Dataset-specific ratings cleaning
def clean_oasis_ratings() -> pd.DataFrame:
    """Clean the OASIS ratings file."""
    raw_path = PROJECT_ROOT / "stimuli" / "OASIS" / "raw" / "oasis.csv"
    df = pd.read_csv(raw_path, dtype={"Theme": str})
    df = df.rename(columns={
        "Theme": "Image_id",
        "Valence_mean": "Valence",
        "Arousal_mean": "Arousal",
    })
    df = df.drop(columns=["Source"])
    df = df.drop(columns=df.columns[0])
    df["Image_id"] = df["Image_id"].str.strip()
    return df


def clean_emomadrid_ratings() -> pd.DataFrame:
    """Clean the EmoMadrid ratings file."""
    raw_path = (
        PROJECT_ROOT / "stimuli" / "EmoMadrid" / "raw"
        / "1_EMindex_2025_07_07(EMindex).csv"
    )
    df = pd.read_csv(
        raw_path,
        sep=";",
        decimal=",",
        skiprows=1,
        encoding="cp1252",
        dtype={"EM CODE": str},
    )
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.rename(columns={
        "EM CODE": "Image_id",
        "Valence mean": "Valence",
        "Arousal mean": "Arousal",
    })
    df = df.drop(columns=["Thumbnail"])
    df.columns = df.columns.str.replace(" ", "_", regex=False)
    df["Image_id"] = df["Image_id"].str.strip()
    return df


def clean_naps_ratings() -> pd.DataFrame:
    """Load the NAPS ratings file without cleaning."""
    raw_path = PROJECT_ROOT / "stimuli" / "NAPS" / "ratings.csv"
    return pd.read_csv(raw_path, dtype={"Image_id": str})


# %% Category subsets
def _update_category_subsets(imageset: str, ratings: pd.DataFrame):
    """Regenerate category columns while preserving other subset columns."""
    path = PROJECT_ROOT / "stimuli" / imageset / "subsets.csv"
    subsets = ratings[["Image_id"]].copy()

    if "Category" in ratings:
        category_columns = pd.get_dummies(
            ratings["Category"], prefix="category", prefix_sep="__", dtype=int
        )
        subsets = pd.concat([subsets, category_columns], axis=1)

    if path.exists():
        previous = pd.read_csv(path, dtype={"Image_id": str}).set_index("Image_id")
        preserved_columns = [
            column for column in previous
            if not column.startswith("category__")
        ]
        for column in preserved_columns:
            subsets[column] = (
                previous[column]
                .reindex(subsets["Image_id"])
                .fillna(0)
                .astype(int)
                .to_numpy()
            )

    subsets.to_csv(path, index=False)


# %% Preprocessing entry point
def clean_ratings(imageset: str):
    """Write standardized ratings and associated category subsets."""
    cleaners = {
        "OASIS": clean_oasis_ratings,
        "EmoMadrid": clean_emomadrid_ratings,
        "NAPS": clean_naps_ratings
    }
    ratings = cleaners[imageset]()
    ratings.to_csv(PROJECT_ROOT / "stimuli" / imageset / "ratings.csv", index=False)
    _update_category_subsets(imageset, ratings)


# %% Run
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("imageset", choices=["OASIS", "EmoMadrid","NAPS"])
    args = parser.parse_args()
    clean_ratings(args.imageset)
