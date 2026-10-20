"""
Load pretrained or random-initialized SLIP-family models and collect forward outputs.
"""

# %% Imports
import sys

import torch
from timm.models.vision_transformer import resize_pos_embed

from config import BATCH_SIZES, PROJECT_ROOT

from script.experiment.stimulus import normalize_images

sys.path.insert(0, str(PROJECT_ROOT / "external" / "SLIP"))
import models


# %% Model loading
RANDOM_VIT_MODELS = {
    "vits16": "SIMCLR_VITS16",
    "vitb16": "SIMCLR_VITB16",
    "vitl16": "SIMCLR_VITL16",
}


def _load_random_vit(model_id: str):
    """Construct an untrained visual backbone using official SLIP constructors."""
    _, architecture, seed_id = model_id.split("_")
    seed = int(seed_id.removeprefix("seed"))

    # The model constructor performs PyTorch/timm's standard initialization.
    # Save and restore the caller's CPU RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = getattr(models, RANDOM_VIT_MODELS[architecture])(
            rand_embed=False,
            ssl_mlp_dim=4096,
            ssl_emb_dim=256,
        )

    return model


def get_device() -> torch.device:
    """Select the available computation device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def adapt_model_input_size(model, canvas_size: tuple[int, int]):
    """Resize ViT positional embeddings to a new canvas size."""
    width, height = canvas_size
    vision = model.visual
    patch_embed = vision.patch_embed

    if patch_embed.img_size == (height, width):  # timm uses (H, W)
        return

    patch_h, patch_w = patch_embed.patch_size
    grid_size = (height // patch_h, width // patch_w)

    target_pos_embed = vision.pos_embed.new_empty(
        1,  # Broadcast across batches
        grid_size[0] * grid_size[1] + vision.num_tokens,
        vision.pos_embed.shape[-1],  # Embedding dimension
    )

    with torch.no_grad():
        resized = resize_pos_embed(
            vision.pos_embed,
            target_pos_embed,  # the target shape is used
            num_tokens=vision.num_tokens,
            gs_new=grid_size,
        )

    vision.pos_embed = torch.nn.Parameter(resized)
    patch_embed.img_size = (height, width)
    patch_embed.grid_size = grid_size
    patch_embed.num_patches = grid_size[0] * grid_size[1]

def load_model(
    model_id: str,
    device: torch.device,
    canvas_size: tuple[int, int] = (224, 224),
):
    """Load pretrained weights, or construct a seeded random-init ViT."""
    if model_id.startswith("random_"):
        model = _load_random_vit(model_id)
    else:
        checkpoint = torch.load(
            PROJECT_ROOT / "checkpoints" / "fb_slip" / f"{model_id}.pt",
            map_location="cpu",
        )
        args = checkpoint["args"]

        model = getattr(models, args.model)(
            rand_embed=False,
            ssl_mlp_dim=args.ssl_mlp_dim,
            ssl_emb_dim=args.ssl_emb_dim,
        )

        state_dict = {
            k.replace("module.", "", 1): v
            for k, v in checkpoint["state_dict"].items()
        }
        model.load_state_dict(state_dict, strict=True)
        del state_dict, checkpoint

    adapt_model_input_size(model, canvas_size)
    
    return model.to(device).eval()


def get_batch_size(model) -> int:
    """Select batch size based on model size to limit memory usage."""
    model_size = {
        384: "small",
        768: "base",
        1024: "large",
    }[model.vision_width]
    return BATCH_SIZES[model_size]


# %% Attention setup
def _register_attention_hooks(model):
    """Capture attention matrices from ViT layers."""
    cache = []

    def hook(module, inputs, output):
        cache.append(output.detach())

    handles = [
        block.attn.attn_drop.register_forward_hook(hook)
        for block in model.visual.blocks
    ]
    return cache, handles


def _get_patch_indices(layout: dict, patch_size: int):
    """Convert constituent regions to patch indices."""
    grid_width = layout["canvas_size"][0] // patch_size
    item_w, item_h = layout["item_size"]

    return [
        [
            row * grid_width + col
            for row in range(y // patch_size, (y + item_h) // patch_size)
            for col in range(x // patch_size, (x + item_w) // patch_size)
        ]
        for x, y in layout["positions"]
    ]


def setup_attention(model, metrics: dict, layout: dict):
    """Register attention hooks and locate constituent patches."""
    if not any(name.startswith("attention_") for name in metrics):
        return [], [], []

    attention_cache, handles = _register_attention_hooks(model)
    patch_size = model.visual.patch_embed.patch_size[0]
    constituent_patch_indices = _get_patch_indices(layout, patch_size)
    return attention_cache, handles, constituent_patch_indices


def remove_hooks(handles: list):
    """Remove registered attention hooks."""
    for handle in handles:
        handle.remove()


# %% Model forward
def forward_composites(
    model,
    composites: torch.Tensor,
    metrics: dict,
    attention_cache: list,
):
    """Forward composites once and collect raw embeddings and attention."""
    embedding_functions = {
        config["embedding_function"]
        for config in metrics.get("embedding_similarity", {}).values()
    }

    with torch.no_grad():
        visual_embeddings = model.visual(normalize_images(composites))
        embeddings = {}

        if "visual" in embedding_functions:
            embeddings["visual"] = visual_embeddings

        if "encode_image" in embedding_functions:
            embeddings["encode_image"] = (
                visual_embeddings @ model.image_projection
                if hasattr(model, "image_projection")
                else visual_embeddings  # SimCLR has no image_projection
            )

    attentions = (
        torch.stack(attention_cache, dim=1)
        if attention_cache
        else None  # No attention metrics requested; hooks not registered
    )
    attention_cache.clear()

    return embeddings, attentions


# %% Test
if __name__ == "__main__":
     # Test rectangular ViT forward and attention extraction with synthetic inputs.
    from timm.models.vision_transformer import VisionTransformer

    model = torch.nn.Module()
    model.visual = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=32,
        depth=1, num_heads=4, num_classes=0,
    ).eval()

    layout = {
        "canvas_size": [256, 224],
        "item_size": [96, 96],
        "positions": [[16, 64], [144, 64]],
    }
    adapt_model_input_size(model, tuple(layout["canvas_size"]))

    metrics = {
        "embedding_similarity": {
            "semi_visual": {"embedding_function": "visual"},
        },
        "attention_layer": {"last_max_mean": {}},
    }
    attention_cache, handles, patch_indices = setup_attention(model, metrics, layout)
    outputs = forward_composites(
        model, torch.randn(1, 3, 224, 256), metrics, attention_cache
    )
    remove_hooks(handles)

    print("Visual embedding:", outputs[0]["visual"].shape)
    print("Attention:", outputs[1].shape)
    print("Constituent patches:", [len(p) for p in patch_indices])
