# ABA_VLM preprocessing

Standardize raw affective image datasets for experiments. Run from the **project root**.

## Files

| Script | Responsibility |
| --- | --- |
| `image_standardization.py` | Collect raw images, optionally crop black borders, resize to 224 × 224, save JPGs. |
| `rating_cleaning.py` | Clean OASIS/EmoMadrid ratings; generate `category__*` columns in `subsets.csv` without overwriting manually maintained columns. |
| `visual_feature_extraction.py` | Extract image-level grayscale, JPEG, CIELAB, and RGB statistics. |
| `run_preprocessing.py` | Run all three steps in sequence. |

## Directory layout

```text
stimuli/
  OASIS/
    raw/
      images/                  # source images (nested folders are supported)
      oasis.csv                # original OASIS ratings
    images/                   # standardized JPGs
    ratings.csv
    subsets.csv
    visual_features.csv
  EmoMadrid/
    raw/
      images/
      1_EMindex_2025_07_07(EMindex).csv
    images/
    ratings.csv
    subsets.csv
    visual_features.csv
```

## Run

```bash
python -m script.preprocessing.run_preprocessing OASIS
python -m script.preprocessing.run_preprocessing EmoMadrid
```

Individual steps can be run with the same module pattern:

```bash
python -m script.preprocessing.image_standardization OASIS
python -m script.preprocessing.rating_cleaning OASIS
python -m script.preprocessing.visual_feature_extraction OASIS
```

The ratings cleaning currently implements the **OASIS** and **EmoMadrid** source formats from the project's earlier preprocessing scripts. Adding NAPS requires its own source-specific ratings cleaning function. Image standardization resizes both datasets to `STANDARD_IMAGE_SIZE` from `config`, currently `(224, 224)`. The EmoMadrid procedure also crops the nonblack bounding box before resizing, matching the earlier preprocessing logic.

## Standardized outputs

- `images/<Image_id>.jpg`: RGB images at the standard size, JPEG quality 95.
- `ratings.csv`: original metadata retained with standardized `Image_id`, `Valence`, `Arousal`, and `Category` when present in the source.
- `subsets.csv`: `Image_id` followed by binary subset membership columns.
- `visual_features.csv`: `Image_id`, `Luminance`, `Contrast`, `JpegSize`, `Entropy`, `Ldim_mean`, `Adim_mean`, `Bdim_mean`, `Rch_mean`, `Gch_mean`, `Bch_mean`.

Visual features retain the definitions from the corrected feature-extraction script: grayscale mean and standard deviation, grayscale Shannon entropy, JPEG byte size at quality 80, mean D65 CIELAB coordinates, and mean RGB channels.

## Subset management

Each `subsets.csv` has one row per image, identified by `Image_id`.

```csv
Image_id,category__Animal,category__People,manual__Textureless
A,1,0,0
B,0,1,1
C,1,0,1
```

`category__*` columns are automatically regenerated from the `Category` column of `ratings.csv`. If that column is absent, no category columns are generated. Other columns (for example, `manual__Textureless` or `manual__NSFW`) are retained when preprocessing is rerun, matched by `Image_id` rather than row order. Newly introduced image IDs receive `0` for existing manual subset columns. The reserved subset name `all` means all image IDs and does not require a separate column.

For **experiment code**, update `get_image_ids()` in `script/experiment/stimulus.py` to read `subsets.csv` rather than `ratings.csv`:

```python
def get_image_ids(imageset_id: str, subset: str) -> list[str]:
    path = PROJECT_ROOT / "stimuli" / imageset_id / "subsets.csv"
    subsets = pd.read_csv(path, dtype={"Image_id": str})
    if subset == "all":
        image_ids = subsets["Image_id"]
    else:
        image_ids = subsets.loc[subsets[subset] == 1, "Image_id"]
    return sorted(image_ids.tolist())
```

When subset membership changes, regenerate any previously cached design matrices that were constructed from that subset name (or use a versioned subset name).
