"""PyTorch datasets.

``SANCMedicalDataset`` is the final version of the class (the notebook
redefined it several times while debugging the mask encoding). Masks are
binarised with ``mask > 0`` which works for Kvasir (0..255) as well as
ORIGA / HAM10000 (0..1) - the earlier ``> 0.5`` after ``/255`` logic was the
reason the first test run reported Dice 0.0 on ORIGA and HAM10000.
"""
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .. import config as C


def _to_tensor_rgb(img):
    arr = np.array(img, dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _to_tensor_mask(mask):
    arr = np.array(mask, dtype=np.float32, copy=True)
    return torch.from_numpy((arr > 0).astype(np.float32)).unsqueeze(0)


class SANCMedicalDataset(Dataset):
    def __init__(self, dataframe, image_size=C.IMAGE_SIZE):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def _load(self, row):
        size = (self.image_size, self.image_size)
        low = Image.open(row["low_light"]).convert("RGB").resize(size, Image.Resampling.BILINEAR)
        orig = Image.open(row["original"]).convert("RGB").resize(size, Image.Resampling.BILINEAR)
        mask = Image.open(row["mask"]).convert("L").resize(size, Image.Resampling.NEAREST)
        return low, orig, mask

    def _pack(self, row, low, orig, mask):
        return {
            "low_light": _to_tensor_rgb(low),
            "original": _to_tensor_rgb(orig),
            "mask": _to_tensor_mask(mask),
            "dataset": row["dataset"],
            "filename": row["filename"],
            "split": row["split"],
        }

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return self._pack(row, *self._load(row))


class AugmentedSANCMedicalDataset(SANCMedicalDataset):
    """Experiment C: spatial + photometric augmentation (photometric on low-light only)."""

    def __init__(self, dataframe, training=False, image_size=C.IMAGE_SIZE):
        super().__init__(dataframe, image_size)
        self.training = training

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        low, orig, mask = self._load(row)
        if self.training:
            if random.random() < 0.5:
                low, orig, mask = TF.hflip(low), TF.hflip(orig), TF.hflip(mask)
            if random.random() < 0.3:
                low, orig, mask = TF.vflip(low), TF.vflip(orig), TF.vflip(mask)
            if random.random() < 0.4:
                angle = random.uniform(-15, 15)
                bil, near = TF.InterpolationMode.BILINEAR, TF.InterpolationMode.NEAREST
                orig = TF.rotate(orig, angle, interpolation=bil)
                low = TF.rotate(low, angle, interpolation=bil)
                mask = TF.rotate(mask, angle, interpolation=near)
            if random.random() < 0.5:
                low = TF.adjust_brightness(low, random.uniform(0.8, 1.2))
                low = TF.adjust_contrast(low, random.uniform(0.8, 1.2))
        return self._pack(row, low, orig, mask)


def make_loader(df, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=0, augmented=False, training=False):
    ds = AugmentedSANCMedicalDataset(df, training=training) if augmented else SANCMedicalDataset(df)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=__import__("torch").cuda.is_available())
