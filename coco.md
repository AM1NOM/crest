# Week 1 COCO Caption Data

This folder contains a small, reproducible image-caption subset for training data pipeline experiments. It uses 100 COCO 2017 validation images and one caption per image.

## Files

- `data/manifest.txt` lists the selected image paths and captions.
- `vocab.json` contains the tokenizer vocabulary built from the manifest captions.
- `src/data.py` provides `ImageCaptionDataset` and `get_dataloader()`.

## Recreate the Dataset

Download the official COCO files, but do not commit the raw images or annotation archives:

```powershell
mkdir week1\local_data
curl.exe -L http://images.cocodataset.org/zips/val2017.zip -o week1\local_data\val2017.zip
curl.exe -L http://images.cocodataset.org/annotations/annotations_trainval2017.zip -o week1\local_data\annotations_trainval2017.zip
tar -xf week1\local_data\val2017.zip -C week1\local_data
tar -xf week1\local_data\annotations_trainval2017.zip -C week1\local_data
```

The manifest paths are relative to `week1/local_data`, for example `val2017/000000179765.jpg`. You can keep the full `val2017` folder locally, or copy only the 100 images listed in `data/manifest.txt` into the same relative layout.

## Use the DataLoader

```python
from week1.src.data import get_dataloader

dataloader = get_dataloader(
    manifest_path="week1/data/manifest.txt",
    image_root="week1/local_data",
    vocab_path="week1/vocab.json",
    batch_size=8,
)

images, captions = next(iter(dataloader))
print(images.shape)    # torch.Size([8, 3, 224, 224])
print(captions.shape)  # torch.Size([8, 32])
```

`local_data/` is intentionally ignored by convention and should stay out of git.
