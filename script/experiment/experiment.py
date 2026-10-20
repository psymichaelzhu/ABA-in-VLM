"""
Run experiments and save results in chunks.
"""

# %% Imports
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from config import CHUNK_SIZE, PROJECT_ROOT, message, save_config_snapshot
from script.experiment.design_matrix import ensure_design_matrix
from script.experiment.stimulus import (
    ensure_constituent_embeddings,
    load_image_cache,
    make_composite,
)
from script.experiment.model import (
    get_batch_size,
    get_device,
    load_model,
    forward_composites,
    adapt_model_input_size,
    setup_attention,
    remove_hooks,
)
from script.experiment.metrics import compute_metrics


# %% Result saving
def _prepare_results(experiment: dict) -> tuple[Path, Path, Path]:
    """Prepare result paths and verify the configuration snapshot."""
    result_dir = (
        PROJECT_ROOT / "results"
        / experiment["protocol_id"] / experiment["experiment_id"]
    )
    chunk_dir = result_dir / "chunks"
    result_path = result_dir / "results.csv"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    config_path = result_dir / "config_snapshot.yml"
    if config_path.exists():
        if yaml.safe_load(config_path.read_text(encoding="utf-8")) != experiment:
            raise ValueError(f"Experiment configuration changed: {config_path}")
    else:
        save_config_snapshot(experiment, config_path)

    return result_dir, chunk_dir, result_path

def _format_chunk_path(chunk_dir: Path, start: int, end: int) -> Path:
    """Build a chunk path for the trial interval."""
    fmt = lambda i: f"{i//1000}k" if i >= 1000 and i % 1000 == 0 else str(i)
    return chunk_dir / f"chunk_{fmt(start)}_{fmt(end)}.csv"

def _save_chunk(chunk_results: list[pd.DataFrame], path: Path):
    """Save a chunk."""
    chunk_df = pd.concat(chunk_results, ignore_index=True)
    temp_path = path.with_suffix(".tmp")
    chunk_df.to_csv(temp_path, index=False)
    temp_path.replace(path)

def _parse_chunk_start(path: Path) -> int:
    """Parse the starting trial index from a chunk filename."""
    start = path.stem.split("_")[1]
    return int(start[:-1]) * 1000 if start.endswith("k") else int(start)

def _merge_chunks(chunk_dir: Path, result_path: Path):
    """Merge chunks to get results.csv."""
    temp_path = result_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        for i, path in enumerate(sorted(chunk_dir.glob("chunk_*_*.csv"), key=_parse_chunk_start)):
            pd.read_csv(path).to_csv(f, index=False, header=(i == 0))
    temp_path.replace(result_path)


# %% Model and embedding preparation
def _prepare_model_and_embeddings(
    experiment: dict,
    device: torch.device,
):
    """Prepare the composite model and constituent embedding cache."""
    model_id = experiment["model"]
    canvas_size = tuple(experiment["layout"]["canvas_size"])

    embedding_configs = experiment["metrics"].get("embedding_similarity", {})
    needs_isolate = any(
        config["present"] == "isolate"
        for config in embedding_configs.values()
    )

    initial_size = (224, 224) if needs_isolate else canvas_size
    model = load_model(model_id, device, initial_size)
    batch_size = get_batch_size(model)

    if canvas_size == (224, 224):
        constituent_embeddings = ensure_constituent_embeddings(
            experiment, model, batch_size=batch_size
        )
    else:
        isolate_embeddings = {}
        if needs_isolate:
            isolate_embeddings = ensure_constituent_embeddings(
                experiment, model,
                batch_size=batch_size,
                present="isolate",
            )
            adapt_model_input_size(model, canvas_size)

        semi_embeddings = ensure_constituent_embeddings(
            experiment, model,
            batch_size=batch_size,
            present="semi",
        )
        constituent_embeddings = {
            **isolate_embeddings,
            **semi_embeddings,
        }

    return model, batch_size, constituent_embeddings


# %% Batch processing
# Process trials in batches to limit memory usage.
def _prepare_batch_inputs(
    batch_design_matrix: pd.DataFrame,
    image_cache: dict,
    constituent_embedding_cache: dict,
    layout: dict,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """Prepare composite images and constituent embeddings for one batch."""
    constituent_columns = [
        col for col in batch_design_matrix
        if col.startswith("Constituent_")
    ]

    constituent_images = []
    embeddings_by_variant = {
        config_key: []
        for config_key in constituent_embedding_cache
    }

    for position_index, column in enumerate(constituent_columns):
        image_ids = batch_design_matrix[column]

        images = torch.stack([
            image_cache[image_id]
            for image_id in image_ids
        ]).to(device=device, dtype=torch.float32) / 255

        constituent_images.append(images)

        for config_key, cached_embeddings in constituent_embedding_cache.items():
            embeddings = torch.stack([
                cached_embeddings[image_id][position_index]
                for image_id in image_ids
            ])
            embeddings_by_variant[config_key].append(embeddings)

    composites = make_composite(constituent_images, layout)
    return composites, embeddings_by_variant


def _process_batch(
    batch_design_matrix: pd.DataFrame,
    experiment: dict,
    model,
    image_cache: dict,
    constituent_embedding_cache: dict,
    attention_cache: list,
    constituent_patch_indices: list[list[int]],
) -> pd.DataFrame:
    """Process one batch and return long-form metric results."""
    device = next(model.parameters()).device

    composites, constituent_embeddings_by_variant = _prepare_batch_inputs(
        batch_design_matrix,
        image_cache,
        constituent_embedding_cache,
        experiment["layout"],
        device,
    )

    composite_outputs = forward_composites(
        model, composites, experiment["metrics"], attention_cache
    )

    return compute_metrics(
        batch_design_matrix["Trial_id"],
        composite_outputs,
        experiment["metrics"],
        constituent_embeddings_by_variant,
        constituent_patch_indices,
    )


# %% Chunk execution
# Group batches into chunks for periodic saving and resumption.
def _run_by_chunks(
    experiment: dict,
    design_matrix: pd.DataFrame,
    model,
    image_cache: dict,
    constituent_embeddings_by_variant: dict,
    attention_cache: list,
    constituent_patch_indices: list[list[int]],
    batch_size: int,
    chunk_dir: Path,
):
    """Process and save experiment results in resumable chunks."""
    with tqdm(
        total=len(design_matrix),
        desc=experiment["experiment_id"],
        unit="trial",
        position=1,
        leave=False,
    ) as progress:
        for chunk_start in range(0, len(design_matrix), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(design_matrix))
            path = _format_chunk_path(chunk_dir, chunk_start, chunk_end)

            if path.exists():
                progress.update(chunk_end - chunk_start)
                continue

            chunk_results = []
            for start in range(chunk_start, chunk_end, batch_size):
                batch_design_matrix = design_matrix.iloc[
                    start:min(start + batch_size, chunk_end)
                ]
                batch_results = _process_batch(
                    batch_design_matrix,
                    experiment,
                    model,
                    image_cache,
                    constituent_embeddings_by_variant,
                    attention_cache,
                    constituent_patch_indices,
                )
                chunk_results.append(batch_results)
                progress.update(len(batch_design_matrix))

            _save_chunk(chunk_results, path)


# %% Experiment execution
def run_experiment(experiment: dict) -> Path:
    """The full pipeline of running an experiment."""
    imageset = experiment["imageset"]
    subset = experiment["subset"]

    result_dir, chunk_dir, result_path = _prepare_results(experiment)
    if result_path.exists():
        message(f"Already completed: {experiment['experiment_id']}")
        return result_dir

    design_matrix = ensure_design_matrix(experiment)
    device = get_device()
    model, batch_size, constituent_embeddings_by_variant = (
        _prepare_model_and_embeddings(experiment, device)
    )

    message(f"Experiment: {experiment['experiment_id']}")
    message(f"Device: {device} | Batch size: {batch_size}")

    image_cache = load_image_cache(imageset, subset)
    attention_cache, handles, constituent_patch_indices = setup_attention(
        model, experiment["metrics"], experiment["layout"]
    )

    try:
        _run_by_chunks(
            experiment,
            design_matrix,
            model,
            image_cache,
            constituent_embeddings_by_variant,
            attention_cache,
            constituent_patch_indices,
            batch_size,
            chunk_dir,
        )
    finally:
        remove_hooks(handles)

    _merge_chunks(chunk_dir, result_path)
    message(f"Results saved: {result_path}")
    return result_dir


# %% Test
if __name__ == "__main__":
    # Test batch preparation and chunk saving with synthetic data.
    layout = {
        "canvas_size": [256, 224],
        "item_size": [96, 96],
        "positions": [[16, 64], [144, 64]],
        "background": [128, 128, 128],
    }
    design_matrix = pd.DataFrame({
        "Trial_id": [0, 1],
        "Constituent_0": ["A", "B"],
        "Constituent_1": ["B", "A"],
    })
    image_cache = {
        "A": torch.zeros(3, 224, 224, dtype=torch.uint8),
        "B": torch.full((3, 224, 224), 255, dtype=torch.uint8),
    }
    embedding_cache = {
        "semi_visual": {
            "A": torch.ones(2, 4),
            "B": -torch.ones(2, 4),
        }
    }

    composites, embeddings = _prepare_batch_inputs(
        design_matrix, image_cache, embedding_cache, layout,
        torch.device("cpu"),
    )
    print("Composites:", composites.shape)
    print("Constituent embeddings:",
          [x.shape for x in embeddings["semi_visual"]])

    output_dir = PROJECT_ROOT / "test" / "experiment"
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_chunk([pd.DataFrame({"Value": [0.2, 0.4]})], _format_chunk_path(output_dir, 0, 2))
    _save_chunk([pd.DataFrame({"Value": [0.6, 0.8]})], _format_chunk_path(output_dir, 2, 4))
    _merge_chunks(output_dir, output_dir / "results.csv")
    print("Merged results:\n", pd.read_csv(output_dir / "results.csv"))
