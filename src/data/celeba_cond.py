"""
CelebA Dataset with Attribute Labels

Wraps the existing CelebA dataloader to also return binary attribute labels
for the selected attributes (e.g., Smiling, Eyeglasses, Male).

This module provides:
  - CelebAWithAttributes: Dataset class returning (image, attr_vector)
  - create_cond_dataloader: Creates a DataLoader with attribute labels
  - create_cond_dataloader_from_config: Config-driven version

The attribute labels come from the standard CelebA attr file.
If using the HuggingFace subset, attributes are loaded from the HF dataset.
"""

import os
import json
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Default attributes to use for conditioning
DEFAULT_ATTRIBUTES = ["Smiling", "Eyeglasses", "Male"]

# Full list of CelebA attributes (40 total) for reference
CELEBA_ATTRIBUTES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
    "Wearing_Necktie", "Young",
]


class CelebAWithAttributes(Dataset):
    """
    CelebA dataset that returns (image, attribute_vector) pairs.

    The attribute vector is a float tensor of shape (K,) where K is the number
    of selected attributes, with values 0.0 or 1.0.

    Supports two data sources:
      1. Local folder with images + attr file
      2. HuggingFace dataset (electronickale/cmu-10799-celeba64-subset)
    """

    def __init__(
        self,
        root: str = "./data/celeba-subset",
        split: str = "train",
        image_size: int = 64,
        selected_attributes: Optional[List[str]] = None,
        augment: bool = True,
        from_hub: bool = False,
        hf_repo: str = "electronickale/cmu-10799-celeba64-subset",
        cache_dir: Optional[str] = None,
    ):
        self.image_size = image_size
        self.selected_attributes = selected_attributes or DEFAULT_ATTRIBUTES
        self.augment = augment

        # Build transforms
        transform_list = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if augment:
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
        ])
        self.transform = transforms.Compose(transform_list)

        if from_hub:
            self._load_from_hub(hf_repo, split, cache_dir)
        else:
            self._load_from_local(root, split)

    def _load_from_hub(self, hf_repo, split, cache_dir):
        """Load from HuggingFace datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets to use from_hub=True")

        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        ds = load_dataset(hf_repo, split=split, **kwargs)
        self.hf_dataset = ds
        self.use_hub = True

        # Check if attributes are available in the HF dataset
        # The HF dataset may store attributes as separate columns
        sample = ds[0]
        self.hub_has_attrs = any(
            attr in sample or attr.lower() in sample
            for attr in self.selected_attributes
        )

        if not self.hub_has_attrs:
            # Try to find an 'attributes' or 'attr' column
            if "attributes" in sample:
                self.hub_attr_key = "attributes"
                self.hub_has_attrs = True
            elif "attr" in sample:
                self.hub_attr_key = "attr"
                self.hub_has_attrs = True
            else:
                self.hub_attr_key = None
                print(
                    f"WARNING: HF dataset does not contain attribute columns. "
                    f"Available columns: {list(sample.keys())}. "
                    f"Using random attributes for training (you should fix this)."
                )

    def _load_from_local(self, root, split):
        """Load from local folder structure."""
        self.use_hub = False

        img_dir = os.path.join(root, split, "images")
        if not os.path.isdir(img_dir):
            # Try without split subfolder
            img_dir = os.path.join(root, "images")

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f"Image directory not found: {img_dir}. "
                f"Check your data root: {root}"
            )

        self.image_paths = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        # Try to load attribute file
        self.attr_dict = {}
        attr_file = os.path.join(root, "attr.json")
        if not os.path.exists(attr_file):
            attr_file = os.path.join(root, split, "attr.json")
        if not os.path.exists(attr_file):
            # Try standard CelebA format: list_attr_celeba.txt
            attr_file = os.path.join(root, "list_attr_celeba.txt")
        if not os.path.exists(attr_file):
            attr_file = os.path.join(root, split, "list_attr_celeba.txt")

        if os.path.exists(attr_file):
            self._parse_attr_file(attr_file)
        else:
            print(
                f"WARNING: No attribute file found in {root}. "
                f"Tried: attr.json, list_attr_celeba.txt. "
                f"Will generate synthetic attributes based on filename hash."
            )

    def _parse_attr_file(self, path):
        """Parse CelebA attribute file (txt or json format)."""
        if path.endswith(".json"):
            with open(path) as f:
                raw = json.load(f)
            # Expecting {filename: {attr_name: 0/1, ...}, ...}
            # or {filename: [0, 1, -1, ...], ...}
            for fname, attrs in raw.items():
                if isinstance(attrs, dict):
                    vec = []
                    for attr_name in self.selected_attributes:
                        val = attrs.get(attr_name, 0)
                        vec.append(1.0 if val > 0 else 0.0)
                    self.attr_dict[fname] = vec
                elif isinstance(attrs, list):
                    # Assume same order as CELEBA_ATTRIBUTES
                    vec = []
                    for attr_name in self.selected_attributes:
                        idx = CELEBA_ATTRIBUTES.index(attr_name)
                        val = attrs[idx] if idx < len(attrs) else 0
                        vec.append(1.0 if val > 0 else 0.0)
                    self.attr_dict[fname] = vec

        elif path.endswith(".txt"):
            with open(path) as f:
                lines = f.readlines()

            # Standard CelebA format:
            # Line 1: number of images
            # Line 2: attribute names
            # Line 3+: filename val1 val2 ... val40
            num_images = int(lines[0].strip())
            attr_names = lines[1].strip().split()

            # Find indices of selected attributes
            attr_indices = []
            for attr_name in self.selected_attributes:
                if attr_name in attr_names:
                    attr_indices.append(attr_names.index(attr_name))
                else:
                    print(f"WARNING: Attribute '{attr_name}' not found in {path}")
                    attr_indices.append(-1)

            for line in lines[2:]:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                fname = parts[0]
                vals = [int(v) for v in parts[1:]]
                vec = []
                for idx in attr_indices:
                    if idx >= 0 and idx < len(vals):
                        vec.append(1.0 if vals[idx] > 0 else 0.0)
                    else:
                        vec.append(0.0)
                self.attr_dict[fname] = vec

            print(f"Loaded attributes for {len(self.attr_dict)} images from {path}")

    def _get_attr_for_filename(self, fname: str) -> torch.Tensor:
        """Get attribute vector for a given filename."""
        basename = os.path.basename(fname)

        if basename in self.attr_dict:
            return torch.tensor(self.attr_dict[basename], dtype=torch.float32)

        # Fallback: deterministic pseudo-random based on filename hash
        h = hash(basename)
        K = len(self.selected_attributes)
        vec = [(h >> i) & 1 for i in range(K)]
        return torch.tensor(vec, dtype=torch.float32)

    def __len__(self):
        if self.use_hub:
            return len(self.hf_dataset)
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_hub:
            return self._getitem_hub(idx)
        return self._getitem_local(idx)

    def _getitem_hub(self, idx):
        sample = self.hf_dataset[idx]

        # Get image
        img = sample.get("image", sample.get("img", None))
        if img is None:
            raise KeyError(f"Cannot find image in HF sample. Keys: {list(sample.keys())}")
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        img = self.transform(img)

        # Get attributes
        if self.hub_has_attrs:
            vec = []
            for attr_name in self.selected_attributes:
                # Try direct column access
                val = sample.get(attr_name, sample.get(attr_name.lower(), None))
                if val is not None:
                    vec.append(1.0 if val > 0 else 0.0)
                elif hasattr(self, "hub_attr_key") and self.hub_attr_key:
                    attrs = sample[self.hub_attr_key]
                    if isinstance(attrs, dict):
                        vec.append(1.0 if attrs.get(attr_name, 0) > 0 else 0.0)
                    else:
                        vec.append(0.0)
                else:
                    vec.append(0.0)
            attrs = torch.tensor(vec, dtype=torch.float32)
        else:
            # Fallback: hash-based pseudo attributes
            h = hash(idx)
            K = len(self.selected_attributes)
            attrs = torch.tensor(
                [(h >> i) & 1 for i in range(K)], dtype=torch.float32
            )

        return img, attrs

    def _getitem_local(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        attrs = self._get_attr_for_filename(path)
        return img, attrs


def create_cond_dataloader(
    root: str = "./data/celeba-subset",
    split: str = "train",
    image_size: int = 64,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    selected_attributes: Optional[List[str]] = None,
    augment: bool = True,
    shuffle: bool = True,
    drop_last: bool = True,
    from_hub: bool = False,
    hf_repo: str = "electronickale/cmu-10799-celeba64-subset",
    cache_dir: Optional[str] = None,
) -> DataLoader:
    """Create a conditional CelebA dataloader."""
    dataset = CelebAWithAttributes(
        root=root,
        split=split,
        image_size=image_size,
        selected_attributes=selected_attributes,
        augment=augment,
        from_hub=from_hub,
        hf_repo=hf_repo,
        cache_dir=cache_dir,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def create_cond_dataloader_from_config(
    config: dict, split: str = "train"
) -> DataLoader:
    """Create a conditional dataloader from a YAML config dict."""
    data_cfg = config.get("data", {}) or {}
    training_cfg = config.get("training", {}) or {}
    cond_cfg = config.get("conditioning", {}) or {}

    selected_attributes = cond_cfg.get("attributes", DEFAULT_ATTRIBUTES)

    return create_cond_dataloader(
        root=data_cfg.get("root", data_cfg.get("data_dir", "./data/celeba-subset")),
        split=split,
        image_size=int(data_cfg.get("image_size", 64)),
        batch_size=int(training_cfg.get("batch_size", 64)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        selected_attributes=selected_attributes,
        augment=(split == "train"),
        shuffle=(split == "train"),
        drop_last=(split == "train"),
        from_hub=bool(data_cfg.get("from_hub", False)),
        hf_repo=data_cfg.get("hf_repo", "electronickale/cmu-10799-celeba64-subset"),
        cache_dir=data_cfg.get("cache_dir", None),
    )
