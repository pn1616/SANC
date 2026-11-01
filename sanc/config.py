"""Central configuration: paths and hyper-parameters.

All paths default to the Google-Drive layout used in the original Colab
notebook (``/content/drive/MyDrive/SANC``). Override the root with the
``SANC_ROOT`` environment variable, e.g.::

    export SANC_ROOT=/path/to/SANC
"""
import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Root folders
# ----------------------------------------------------------------------------
SANC_ROOT = Path(os.environ.get("SANC_ROOT", "/content/drive/MyDrive/SANC"))
DATASETS_ROOT = SANC_ROOT / "datasets"
MODELS_DIR = SANC_ROOT / "models"
RESULTS_DIR = Path(os.environ.get("SANC_RESULTS", SANC_ROOT / "results"))

# Per-dataset roots (exact Drive layout from the notebook)
KVASIR_ROOT = DATASETS_ROOT / "kvasir"                       # original/ low_light/ masks/
KVASIR_META_CSV = KVASIR_ROOT / "kvasir_metadata.csv"
SPLIT_META_DIR = DATASETS_ROOT / "metadata"                  # kvasir_{train,val,test}.csv

ORIGA_ROOT = DATASETS_ROOT / "ORIGA_processed" / "integration"  # images/ low_light/ masks/
ORIGA_RAW_META_CSV = ORIGA_ROOT / "ORIGA_metadata.csv"
ORIGA_META_CSV = ORIGA_ROOT / "ORIGA_common_metadata.csv"

HAM_ROOT = DATASETS_ROOT / "ham" / "HAM preprocessing"       # images/images, low_light/low_light, masks/masks
HAM_META_CSV = HAM_ROOT / "metadata.csv"

# Sub-folder names per dataset: column -> folder (relative to dataset root)
DATASET_LAYOUT = {
    "Kvasir-SEG": {"root": KVASIR_ROOT, "original": "original", "low_light": "low_light", "mask": "masks"},
    "ORIGA": {"root": ORIGA_ROOT, "original": "images", "low_light": "low_light", "mask": "masks"},
    "HAM10000": {"root": HAM_ROOT, "original": "images/images", "low_light": "low_light/low_light", "mask": "masks/masks"},
}
DATASET_NAMES = ["Kvasir-SEG", "ORIGA", "HAM10000"]
COMMON_COLUMNS = ["filename", "original", "low_light", "mask", "dataset", "split"]

# Kvasir test image whose mask file is empty (0 bytes) -> excluded everywhere
CORRUPT_MASK_FILENAME = "cju89y9h0puti0818i5yw29e6.jpg"

# ----------------------------------------------------------------------------
# Data / training
# ----------------------------------------------------------------------------
SEED = 42
IMAGE_SIZE = int(os.environ.get("SANC_IMAGE_SIZE", 128))   # override only for quick CPU smoke tests
BATCH_SIZE = 8
KVASIR_SPLIT = (0.70, 0.15, 0.15)   # train / val / test, random_state=42

# Low-light simulation (ORIGA settings from the notebook)
LOW_LIGHT_BRIGHTNESS = 0.40
LOW_LIGHT_GAMMA = 1.5
LOW_LIGHT_NOISE_STD = 0.01

# ----------------------------------------------------------------------------
# Diffusion
# ----------------------------------------------------------------------------
T = 100
BETA_START = 1e-4
BETA_END = 0.02

# Reverse network
REVERSE_IN_CHANNELS = 7      # x_t (3) + low-light (3) + saliency (1)
REVERSE_BASE_CHANNELS = 64
REVERSE_TIME_DIM = 128
REVERSE_LR = 1e-4
REVERSE_WEIGHT_DECAY = 1e-4
REVERSE_EPOCHS = 10

# ----------------------------------------------------------------------------
# SANC extensions (A: structure, B: uncertainty, C: timestep)
# ----------------------------------------------------------------------------
LAMBDA_S = 0.30   # A: Sobel structural fusion weight
LAMBDA_U = 0.50   # B: entropy-uncertainty attenuation
LAMBDA_T = 0.05   # C: timestep spectral shift (largest value that passed audit A3)

# Ablation sampler (frozen in STEP 14 of the notebook)
ABLATION_T0 = 99
ABLATION_STEPS = 25

# ----------------------------------------------------------------------------
# Checkpoint file names (same names as on Drive)
# ----------------------------------------------------------------------------
CKPT = {
    "saliency": MODELS_DIR / "best_unet_saliency.pth",
    "saliency_bce_dice": MODELS_DIR / "best_unet_bce_dice.pth",
    "saliency_tversky_aug": MODELS_DIR / "best_unet_tversky_augmented.pth",
    "attention_unet": MODELS_DIR / "best_attention_unet.pth",
    "unetpp": MODELS_DIR / "best_unetpp_saliency.pth",
    "saliency_finetuned": MODELS_DIR / "best_unet_finetuned.pth",
    "reverse": MODELS_DIR / "reverse_diffusion_model.pth",
}
