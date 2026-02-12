"""
CelebA Attribute Classifier

Trains a lightweight attribute classifier on CelebA 64×64 to measure
edit success (control accuracy) of the conditional generation / editing.

The classifier is a simple ResNet-18 backbone with a multi-label head.

Usage:
  # Train classifier
  python train_classifier.py --config configs/cfg_flow_matching.yaml --epochs 10

  # Evaluate generated samples
  python train_classifier.py --config configs/cfg_flow_matching.yaml \
      --eval --eval_dir ./cfg_samples --classifier_ckpt classifier.pt
"""

import os
import argparse
import json
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image

import yaml

from src.data.celeba_cond import CelebAWithAttributes, DEFAULT_ATTRIBUTES


class AttributeClassifier(nn.Module):
    """
    Multi-label attribute classifier based on ResNet-18.

    Input: (B, 3, 64, 64) images in [0, 1] (not [-1, 1])
    Output: (B, K) logits for K binary attributes
    """

    def __init__(self, num_attributes: int = 3, pretrained: bool = False):
        super().__init__()
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        # Replace final FC
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_attributes)

    def forward(self, x):
        return self.backbone(x)

    def predict_probs(self, x):
        """Return probabilities for each attribute."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def predict_labels(self, x, threshold=0.5):
        """Return binary predictions."""
        probs = self.predict_probs(x)
        return (probs > threshold).float()


def train_classifier(config: dict, epochs: int = 10, save_path: str = "classifier.pt"):
    """Train the attribute classifier on CelebA."""
    cond_cfg = config.get("conditioning", {}) or {}
    data_cfg = config.get("data", {}) or {}
    attr_names = cond_cfg.get("attributes", DEFAULT_ATTRIBUTES)
    num_attrs = len(attr_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training classifier on {device}")
    print(f"Attributes: {attr_names}")

    # Dataset
    dataset = CelebAWithAttributes(
        root=data_cfg.get("root", "./data/celeba-subset"),
        split="train",
        image_size=int(data_cfg.get("image_size", 64)),
        selected_attributes=attr_names,
        augment=True,
        from_hub=bool(data_cfg.get("from_hub", False)),
        hf_repo=data_cfg.get("hf_repo", "electronickale/cmu-10799-celeba64-subset"),
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    print(f"Dataset size: {len(dataset)}")

    # Model
    classifier = AttributeClassifier(num_attributes=num_attrs, pretrained=True).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # Training
    classifier.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, attrs in pbar:
            # images are in [-1, 1], convert to [0, 1] for classifier
            images = (images * 0.5 + 0.5).to(device)
            attrs = attrs.to(device)

            logits = classifier(images)
            loss = F.binary_cross_entropy_with_logits(logits, attrs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == attrs).sum().item()
            total += attrs.numel()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.3f}"})

        scheduler.step()
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: loss={total_loss/len(dataloader):.4f}, acc={epoch_acc:.4f}")

    # Save
    torch.save({
        "state_dict": classifier.state_dict(),
        "num_attributes": num_attrs,
        "attributes": attr_names,
    }, save_path)
    print(f"Saved classifier to {save_path}")


def evaluate_samples(
    classifier_path: str,
    sample_dir: str,
    target_attrs: Optional[torch.Tensor] = None,
    attr_names: Optional[List[str]] = None,
):
    """Evaluate generated samples using the trained classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load classifier
    ckpt = torch.load(classifier_path, map_location=device)
    num_attrs = ckpt["num_attributes"]
    attr_names = attr_names or ckpt.get("attributes", DEFAULT_ATTRIBUTES[:num_attrs])

    classifier = AttributeClassifier(num_attributes=num_attrs).to(device)
    classifier.load_state_dict(ckpt["state_dict"])
    classifier.eval()

    # Load images
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        # Classifier expects [0, 1]
    ])

    image_files = sorted([
        f for f in os.listdir(sample_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    print(f"Evaluating {len(image_files)} images from {sample_dir}")

    all_probs = []
    batch_size = 64
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        imgs = []
        for fname in batch_files:
            img = Image.open(os.path.join(sample_dir, fname)).convert("RGB")
            imgs.append(transform(img))
        imgs = torch.stack(imgs).to(device)

        with torch.no_grad():
            probs = classifier.predict_probs(imgs)
        all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_preds = (all_probs > 0.5).float()

    # Report per-attribute statistics
    print("\n--- Attribute Predictions ---")
    for i, name in enumerate(attr_names):
        mean_prob = all_probs[:, i].mean().item()
        frac_positive = all_preds[:, i].mean().item()
        print(f"  {name}: mean_prob={mean_prob:.3f}, fraction_positive={frac_positive:.3f}")

    # If target attributes provided, compute accuracy
    if target_attrs is not None:
        target = target_attrs.unsqueeze(0).expand(all_preds.shape[0], -1)
        per_attr_acc = (all_preds == target).float().mean(dim=0)
        overall_acc = (all_preds == target).float().mean().item()
        print("\n--- Control Accuracy ---")
        for i, name in enumerate(attr_names):
            print(f"  {name}: accuracy={per_attr_acc[i].item():.3f}")
        print(f"  Overall: {overall_acc:.3f}")

    return all_probs, all_preds


def main():
    parser = argparse.ArgumentParser(description="Attribute Classifier")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="classifier.pt")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_dir", type=str, default="cfg_samples")
    parser.add_argument("--classifier_ckpt", type=str, default="classifier.pt")
    parser.add_argument("--target_attrs", type=str, default=None,
                        help="Target attributes for accuracy, e.g., '1,0,1'")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.eval:
        target = None
        if args.target_attrs:
            target = torch.tensor([float(v) for v in args.target_attrs.split(",")])
        evaluate_samples(
            args.classifier_ckpt,
            args.eval_dir,
            target_attrs=target,
            attr_names=config.get("conditioning", {}).get("attributes"),
        )
    else:
        train_classifier(config, epochs=args.epochs, save_path=args.save_path)


if __name__ == "__main__":
    main()
