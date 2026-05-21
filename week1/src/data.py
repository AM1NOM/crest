"""Small COCO image-caption dataset utilities for Week 1 training."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokenize(text: str) -> List[str]:
    """Lowercase regex tokenizer shared by the dataset and vocab builder."""
    return TOKEN_PATTERN.findall(text.lower())


def load_vocab(vocab_path: str | Path) -> Dict[str, object]:
    """Load a JSON vocab with token_to_idx, idx_to_token, and specials fields."""
    with Path(vocab_path).open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    if "token_to_idx" not in vocab:
        raise ValueError(f"Missing token_to_idx in vocab file: {vocab_path}")
    return vocab


def _read_manifest(manifest_path: str | Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                image_path, caption = line.split("\t", 1)
            except ValueError as exc:
                raise ValueError(
                    f"Manifest line {line_number} must be tab-separated: {raw_line!r}"
                ) from exc
            rows.append((image_path, caption))

    if not rows:
        raise ValueError(f"No image-caption rows found in manifest: {manifest_path}")
    return rows


def numericalize_caption(
    caption: str,
    token_to_idx: Dict[str, int],
    max_length: int,
    pad_token: str = "<pad>",
    unk_token: str = "<unk>",
    bos_token: str = "<bos>",
    eos_token: str = "<eos>",
) -> torch.Tensor:
    """Convert a caption to a fixed-length LongTensor with BOS/EOS and padding."""
    if max_length < 2:
        raise ValueError("max_length must be at least 2 to include BOS and EOS tokens")

    pad_idx = token_to_idx[pad_token]
    unk_idx = token_to_idx[unk_token]
    ids = [token_to_idx[bos_token]]
    ids.extend(token_to_idx.get(token, unk_idx) for token in tokenize(caption))
    ids.append(token_to_idx[eos_token])

    ids = ids[:max_length]
    if ids[-1] != token_to_idx[eos_token]:
        ids[-1] = token_to_idx[eos_token]
    ids.extend([pad_idx] * (max_length - len(ids)))
    return torch.tensor(ids, dtype=torch.long)


def default_image_transform(image_size: int = 224) -> Callable[[Image.Image], torch.Tensor]:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


class ImageCaptionDataset(Dataset):
    """Dataset that returns normalized image tensors and padded caption tensors."""

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path,
        vocab_path: str | Path,
        image_size: int = 224,
        max_length: int = 32,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root)
        self.rows = _read_manifest(self.manifest_path)
        self.vocab = load_vocab(vocab_path)
        self.token_to_idx: Dict[str, int] = self.vocab["token_to_idx"]
        self.max_length = max_length
        self.transform = transform or default_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        relative_path, caption = self.rows[index]
        image_path = self.image_root / relative_path

        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))

        caption_tensor = numericalize_caption(
            caption=caption,
            token_to_idx=self.token_to_idx,
            max_length=self.max_length,
        )
        return image_tensor, caption_tensor


def get_dataloader(
    manifest_path: str | Path,
    image_root: str | Path,
    vocab_path: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    max_length: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for the Week 1 COCO caption subset."""
    dataset = ImageCaptionDataset(
        manifest_path=manifest_path,
        image_root=image_root,
        vocab_path=vocab_path,
        image_size=image_size,
        max_length=max_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


__all__: Sequence[str] = (
    "ImageCaptionDataset",
    "default_image_transform",
    "get_dataloader",
    "load_vocab",
    "numericalize_caption",
    "tokenize",
)
