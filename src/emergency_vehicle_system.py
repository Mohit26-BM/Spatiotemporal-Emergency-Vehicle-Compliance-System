"""
================================================================================
  SPATIOTEMPORAL COMPLIANCE ASSESSMENT FOR EMERGENCY VEHICLE PRIORITY
  Full Pipeline — Training + Inference + Compliance + Violation Logging
================================================================================

  What this file does (in order):
  ─────────────────────────────────
  PART 1  — Install dependencies
  PART 2  — Global constants and imports
  PART 3  — Dataset download and preparation
  PART 4  — K-Fold cross-validation splits
  PART 5  — PyTorch dataset class with augmentation
  PART 6  — Train YOLOv8-cls (one fold)
  PART 7  — Train ResNet18 (one fold)
  PART 8  — K-Fold CV master loop
  PART 9  — Test-Time Augmentation (TTA)
  PART 10 — Ensemble Classifier (YOLOv8-cls + ResNet18)
  PART 11 — Evaluate on held-out test fold
  PART 12 — DeepSORT multi-object tracker
  PART 13 — Compliance Engine (New Addition))
  PART 14 — Violation Logger with SQLite (New Addition)
  PART 15 — EmergencyVehicleSystem (inference + compliance)
  PART 16 — Main training pipeline
  PART 17 — Run on video (call after training)

  HOW TO RUN
  ──────────
  Step 1 — Install dependencies (run once):
      pip install ultralytics deep-sort-realtime kagglehub torch torchvision
                  opencv-python matplotlib seaborn scikit-learn streamlit

  Step 2 — Train the models (run once, takes time):
      python emergency_vehicle_system.py

  Step 3 — After training, run on a video by calling run_on_video()
      at the bottom of this file or from a separate script.

  IMPORTANT — Fix before running:
      CLASS_NAMES must be ["non-emergency", "emergency"]
      folder 0 = non-emergency, folder 1 = emergency
      This is fixed correctly in this file.
================================================================================
"""


# ================================================================================
# PART 1 — INSTALL DEPENDENCIES
# ================================================================================
# Uncomment the line below if running in a notebook (Colab / Jupyter)
# If running as a plain .py script, install via terminal instead.

# !pip install ultralytics deep-sort-realtime kagglehub torch torchvision opencv-python matplotlib seaborn scikit-learn


# ================================================================================
# PART 2 — IMPORTS AND GLOBAL CONSTANTS
# ================================================================================

import os
import shutil
import random
import json
import math
import sqlite3
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models

import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

# ── Device: use GPU if available, otherwise CPU ──────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

# ── Class names — IMPORTANT: folder 0 = non-emergency, folder 1 = emergency ──
# This was a bug in the original code (they were swapped). Fixed here.
CLASS_NAMES = ["non-emergency", "emergency"]   # index 0 = non-emergency, 1 = emergency

# ── Supported image file extensions ──────────────────────────────────────────
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ================================================================================
# PART 3 — DATASET DOWNLOAD AND PREPARATION
# ================================================================================

def download_dataset() -> str:
    """
    Downloads the emergency vs non-emergency vehicle classification
    dataset from Kaggle using kagglehub.

    Dataset structure after download:
        dataset/
          train/
            0/   ← non-emergency images
            1/   ← emergency images
          validation/
            0/
            1/
          test/
            0/
            1/

    Returns the local path to the downloaded dataset.
    """
    import kagglehub
    path = kagglehub.dataset_download(
        "parthplc/emergency-vs-nonemergency-vehicle-classification"
    )
    print(f"[INFO] Dataset downloaded to: {path}")
    return path


def find_root(dataset_path: str) -> Path:
    """
    Locates the root folder that contains train/validation/test subfolders.
    Handles cases where Kaggle adds an extra wrapper folder on download.
    """
    p = Path(dataset_path)
    for candidate in [p, *p.iterdir()]:
        if candidate.is_dir() and any(
            (candidate / s).is_dir() for s in ("train", "validation", "test")
        ):
            return candidate
    raise FileNotFoundError(f"Cannot find train/val/test under {dataset_path}")


def collect_all_samples(dataset_path: str) -> tuple:

    """
    Pools all images from every split (train/validation/test) into
    one flat list. We do this so we can manage our own K-Fold splits
    rather than using the pre-defined ones.

    Returns:
        image_paths — list of absolute path strings, one per image
        labels      — parallel list of integers (0=non-emergency, 1=emergency)
    """

    root        = find_root(dataset_path)
    image_paths = []
    labels      = []

    for split_dir in root.iterdir():
        if not split_dir.is_dir():
            continue
        for cls_folder in sorted(split_dir.iterdir()):
            # Only process folders named "0" or "1"
            if not cls_folder.is_dir() or cls_folder.name not in ("0", "1"):
                continue
            label = int(cls_folder.name)   # 0 = non-emergency, 1 = emergency
            for p in cls_folder.iterdir():
                if p.suffix.lower() in EXTS:
                    image_paths.append(str(p))
                    labels.append(label)

    print(f"[INFO] Total images collected : {len(image_paths)}")
    print(f"[INFO]   non-emergency (0)    : {labels.count(0)}")
    print(f"[INFO]   emergency     (1)    : {labels.count(1)}")
    return image_paths, labels


# ================================================================================
# PART 4 — STRATIFIED K-FOLD SPLITS
# ================================================================================

def make_kfold_splits(image_paths: list, labels: list,
                      k: int = 5, seed: int = 42) -> list:
    """
    Divides all images into K stratified folds.

    Strategy:
    - Fold K-1 (the last fold) is permanently held out as the TEST set.
      It is never used in any training or validation step.
    - Folds 0 through K-2 are used for cross-validation.
    - For each CV fold i:
        val   = fold i
        train = all CV folds except fold i
        test  = held-out fold K-1 (same every time)

    Stratified means each fold has the same class ratio as the full dataset,
    preventing imbalanced splits on a small dataset.

    Returns a list of dicts, one per CV fold, each with keys:
        fold, train (indices), val (indices), test (indices)
    """
    skf   = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = [idx.tolist() for _, idx in skf.split(image_paths, labels)]

    # Last fold is permanently held out — never touched during training
    test_idx = folds[-1]
    cv_folds = folds[:-1]   # the remaining K-1 folds for CV

    splits = []
    for i, val_fold in enumerate(cv_folds):
        # Training = all CV folds except the current validation fold
        train_folds = [f for j, f in enumerate(cv_folds) if j != i]
        train_idx   = [idx for fold in train_folds for idx in fold]
        splits.append({
            "fold" : i,
            "train": train_idx,
            "val"  : val_fold,
            "test" : test_idx,
        })

    print(f"\n[INFO] K-Fold split (K={k}):")
    print(f"[INFO]   Held-out test set : {len(test_idx)} images (fold {k-1})")
    for s in splits:
        print(f"[INFO]   CV fold {s['fold']}  →  "
              f"train={len(s['train'])}  val={len(s['val'])}")
    return splits


def write_fold_dataset(image_paths: list, labels: list,
                       split_info: dict, fold_dir: str) -> str:
    """
    Physically copies images into the folder structure that YOLOv8-cls expects:
        fold_dir/
          train/
            non-emergency/
            emergency/
          val/
            non-emergency/
            emergency/
          test/
            non-emergency/
            emergency/

    Returns the fold_dir path string (passed as data= to YOLO training).
    """
    fold_dir = Path(fold_dir)
    # Clean up previous fold data to avoid stale images
    if fold_dir.exists():
        shutil.rmtree(fold_dir)

    for split_name, indices in [("train", split_info["train"]),
                                 ("val",   split_info["val"]),
                                 ("test",  split_info["test"])]:
        # Create class subfolders
        for cls_name in CLASS_NAMES:
            (fold_dir / split_name / cls_name).mkdir(parents=True, exist_ok=True)
        # Copy images into correct class folder
        for idx in indices:
            src      = Path(image_paths[idx])
            cls_name = CLASS_NAMES[labels[idx]]
            # Prefix with index to avoid filename collisions across splits
            dst = fold_dir / split_name / cls_name / f"{idx}_{src.name}"
            shutil.copy2(src, dst)

    return str(fold_dir)


# ================================================================================
# PART 5 — PYTORCH DATASET WITH AUGMENTATION
# ================================================================================

# Training transforms — augmentation makes the model more robust by showing
# it variations of the same image: different crops, flips, colours, rotations.
TRAIN_TF = T.Compose([
    T.Resize((256, 256)),          # slightly larger than target for random crop
    T.RandomCrop(224),             # random 224x224 crop
    T.RandomHorizontalFlip(),      # random left-right flip
    T.ColorJitter(brightness=0.3,  # random brightness/contrast/saturation
                  contrast=0.3,
                  saturation=0.2),
    T.RandomRotation(15),          # random rotation up to 15 degrees
    T.ToTensor(),                  # convert PIL image to PyTorch tensor
    T.Normalize([0.485, 0.456, 0.406],   # ImageNet mean and std
                [0.229, 0.224, 0.225]),  # (standard for pretrained models)
])

# Validation/test transforms — no augmentation, just resize and normalise
VAL_TF = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class FoldDataset(Dataset):
    """
    PyTorch Dataset that reads images from a fold directory.
    Automatically applies training or validation transforms based on split name.
    """

    def __init__(self, fold_dir: str, split: str):
        # Use augmentation for training, plain transforms for val/test
        self.tf      = TRAIN_TF if split == "train" else VAL_TF
        self.samples = []   # list of (image_path, label_index) tuples

        split_dir = Path(fold_dir) / split
        for cls_name in CLASS_NAMES:
            cls_dir = split_dir / cls_name
            if not cls_dir.is_dir():
                continue
            label = CLASS_NAMES.index(cls_name)   # 0 or 1
            for p in cls_dir.iterdir():
                if p.suffix.lower() in EXTS:
                    self.samples.append((str(p), label))

        # Shuffle so batches have mixed classes
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Open as RGB (some images might be grayscale — convert ensures 3 channels)
        img = Image.open(path).convert("RGB")
        return self.tf(img), label


# ================================================================================
# PART 6 — TRAIN YOLOv8-cls FOR ONE FOLD
# ================================================================================

def train_yolo_cls_fold(fold_dir: str, fold_idx: int,
                        epochs: int = 8, imgsz: int = 224,
                        batch: int = 32) -> str:
    """
    Fine-tunes YOLOv8n-cls (nano classification variant) on one CV fold.

    YOLOv8-cls takes a cropped image and outputs class probabilities.
    We use the nano variant for speed — it is fast enough for real-time
    inference on every detected vehicle crop per frame.

    Returns path to the best weights file saved by Ultralytics.
    """
    from ultralytics import YOLO

    # yolov8n-cls.pt = nano classification model pretrained on ImageNet
    model = YOLO("yolov8n-cls.pt")

    results = model.train(
        data    = fold_dir,          # path to fold directory
        epochs  = epochs,            # number of training epochs
        imgsz   = imgsz,             # input image size (224x224)
        batch   = batch,             # batch size
        name    = f"emerg_cls_fold{fold_idx}",
        device  = 0 if torch.cuda.is_available() else "cpu",
        patience= 10,                # early stopping patience
        augment = True,              # enable built-in Ultralytics augmentation
        plots   = True,              # save training plots
    )

    best = str(Path(results.save_dir) / "weights" / "best.pt")
    print(f"[INFO] YOLOv8-cls fold {fold_idx} best weights: {best}")
    return best


# ================================================================================
# PART 7 — TRAIN RESNET18 FOR ONE FOLD
# ================================================================================

def build_resnet(pretrained: bool = True) -> nn.Module:
    """
    Builds a ResNet18 model with a custom classification head.

    ResNet18 was pretrained on ImageNet (1.2M images, 1000 classes).
    We replace the final fully-connected layer with our own 2-class head
    and add dropout for regularisation. This is called transfer learning.

    The pretrained weights give the model a strong starting point —
    it already knows how to detect edges, textures, and shapes.
    We only need to fine-tune it to distinguish emergency vs non-emergency.
    """
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT if pretrained else None
    )
    # Replace the original 1000-class head with our 2-class head
    model.fc = nn.Sequential(
        nn.Dropout(0.4),                              # regularisation
        nn.Linear(model.fc.in_features, len(CLASS_NAMES)),  # 2 output classes
    )
    return model.to(DEVICE)


def train_resnet_fold(fold_dir: str, fold_idx: int,
                      epochs: int = 4, batch: int = 32,
                      lr: float = 1e-4) -> tuple:
    """
    Trains ResNet18 on one CV fold using PyTorch directly.

    Key training choices:
    - AdamW optimiser: adapts learning rate per parameter, weight decay
      prevents overfitting. Lower lr for pretrained layers, higher for head.
    - Cosine annealing: smoothly reduces learning rate over epochs,
      helping the model converge without manual tuning.
    - Label smoothing 0.1: instead of hard 0/1 targets, uses 0.05/0.95.
      This prevents the model from being overconfident on training data.

    Returns (path_to_best_weights, best_validation_accuracy).
    """
    train_ds = FoldDataset(fold_dir, "train")
    val_ds   = FoldDataset(fold_dir, "val")

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False,
                          num_workers=2, pin_memory=True)

    model     = build_resnet()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Lower learning rate for pretrained backbone layers,
    # higher for the new classification head
    optimizer = optim.AdamW([
        {"params": list(model.parameters())[:-2], "lr": lr * 0.1},
        {"params": list(model.parameters())[-2:], "lr": lr},
    ], weight_decay=1e-4)

    # Cosine annealing: lr starts at lr and decreases to ~0 over T_max epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc  = 0.0
    save_path = f"resnet18_fold{fold_idx}.pt"

    for epoch in range(1, epochs + 1):

        # ── Training phase ───────────────────────────────────────────────────
        model.train()
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()      # compute gradients
            optimizer.step()     # update weights
        scheduler.step()         # reduce learning rate

        # ── Validation phase ─────────────────────────────────────────────────
        model.eval()
        correct = total = 0
        with torch.no_grad():    # no gradient computation needed for evaluation
            for imgs, lbls in val_dl:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                preds   = model(imgs).argmax(dim=1)
                correct += (preds == lbls).sum().item()
                total   += len(lbls)
        acc = correct / total

        # Save weights if this epoch is the best so far
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)

        if epoch % 2 == 0 or epoch == epochs:
            print(f"    epoch {epoch:3d}/{epochs}  "
                  f"val_acc={acc:.4f}  best={best_acc:.4f}")

    print(f"[INFO] ResNet18 fold {fold_idx} — best val acc: {best_acc:.4f}")
    return save_path, best_acc


# ================================================================================
# PART 8 — K-FOLD CROSS-VALIDATION MASTER LOOP
# ================================================================================

def run_kfold_cv(image_paths: list, labels: list,
                 k: int = 5, yolo_epochs: int = 8,
                 resnet_epochs: int = 4, batch: int = 32) -> dict:
    """
    Runs the full K-Fold cross-validation loop.

    For each of the K-1 CV folds:
        1. Write the fold dataset to disk
        2. Train YOLOv8-cls
        3. Train ResNet18
        4. Compute ensemble validation accuracy (average of both)

    After all folds, selects the best fold by ensemble accuracy,
    plots a bar chart of fold accuracies, and returns everything
    needed for final test evaluation.

    Returns a dict with keys: cv_results, best, mean_acc, std_acc
    """
    splits     = make_kfold_splits(image_paths, labels, k=k)
    cv_results = []

    for split_info in splits:
        fold_idx = split_info["fold"]
        print(f"\n{'='*60}")
        print(f"  CV FOLD {fold_idx + 1} / {k - 1}")
        print(f"{'='*60}")

        fold_dir = f"kfold_data/fold_{fold_idx}"

        # Write this fold's images to disk in YOLO format
        write_fold_dataset(image_paths, labels, split_info, fold_dir)

        # Train both models on this fold
        yolo_w          = train_yolo_cls_fold(fold_dir, fold_idx,
                                              epochs=yolo_epochs, batch=batch)
        resnet_w, r_acc = train_resnet_fold(fold_dir, fold_idx,
                                            epochs=resnet_epochs, batch=batch)

        # Get YOLO's validation accuracy on this fold
        from ultralytics import YOLO
        yolo_model = YOLO(yolo_w)
        yolo_val   = yolo_model.val(data=fold_dir, split="val", verbose=False)
        y_acc      = float(yolo_val.top1)   # top-1 classification accuracy

        # Ensemble accuracy = simple average of both models
        ensemble_acc = (y_acc + r_acc) / 2
        print(f"  Fold {fold_idx}  YOLO={y_acc:.4f}  "
              f"ResNet={r_acc:.4f}  Ensemble≈{ensemble_acc:.4f}")

        cv_results.append({
            "fold"            : fold_idx,
            "fold_dir"        : fold_dir,
            "yolo_weights"    : yolo_w,
            "resnet_weights"  : resnet_w,
            "yolo_val_acc"    : y_acc,
            "resnet_val_acc"  : r_acc,
            "ensemble_val_acc": ensemble_acc,
            "test_indices"    : split_info["test"],
        })

    # ── Print CV summary ──────────────────────────────────────────────────────
    accs = [r["ensemble_val_acc"] for r in cv_results]
    print(f"\n{'='*60}")
    print(f"  K-FOLD CV SUMMARY  (K={k},  CV folds={k-1})")
    print(f"{'='*60}")
    for r in cv_results:
        print(f"  Fold {r['fold']}  ensemble_val_acc = {r['ensemble_val_acc']:.4f}")
    print(f"  Mean = {np.mean(accs):.4f}  ±  {np.std(accs):.4f}")

    # ── Plot fold accuracies ──────────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.bar([f"Fold {r['fold']}" for r in cv_results], accs,
            color="steelblue", edgecolor="black")
    plt.axhline(np.mean(accs), color="red", linestyle="--",
                label=f"Mean = {np.mean(accs):.4f}")
    plt.ylim(0, 1)
    plt.ylabel("Ensemble Val Accuracy")
    plt.title(f"K-Fold CV Results  (K={k})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("kfold_cv_results.png", dpi=130)
    plt.show()
    print("[INFO] Saved: kfold_cv_results.png")

    # ── Select best fold ──────────────────────────────────────────────────────
    best = max(cv_results, key=lambda r: r["ensemble_val_acc"])
    print(f"\n[INFO] Best fold: {best['fold']} "
          f"(ensemble_val_acc={best['ensemble_val_acc']:.4f})")

    return {
        "cv_results": cv_results,
        "best"      : best,
        "mean_acc"  : float(np.mean(accs)),
        "std_acc"   : float(np.std(accs)),
    }


# ================================================================================
# PART 9 — TEST-TIME AUGMENTATION (TTA)
# ================================================================================

# Five different transforms applied at inference time.
# We run the model on all 5 versions of the same image and average
# the probability outputs. This makes predictions more stable.
_NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

TTA_TRANSFORMS = [
    T.Compose([T.Resize((224, 224)),                             T.ToTensor(), _NORM]),
    T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(1),  T.ToTensor(), _NORM]),
    T.Compose([T.Resize((256, 256)), T.CenterCrop(224),          T.ToTensor(), _NORM]),
    T.Compose([T.Resize((224, 224)), T.ColorJitter(0.2, 0.2),    T.ToTensor(), _NORM]),
    T.Compose([T.Resize((224, 224)), T.RandomRotation(10),       T.ToTensor(), _NORM]),
]


def resnet_predict_tta(model: nn.Module, pil_img: Image.Image) -> np.ndarray:
    """
    Runs ResNet18 on 5 augmented versions of the image and averages
    the softmax probability vectors.
    Returns a numpy array of shape (num_classes,).
    """
    model.eval()
    probs = []
    with torch.no_grad():
        for tf in TTA_TRANSFORMS:
            tensor = tf(pil_img).unsqueeze(0).to(DEVICE)
            prob   = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
            probs.append(prob)
    return np.mean(probs, axis=0)   # average across 5 augmentations


# ================================================================================
# PART 10 — ENSEMBLE CLASSIFIER
# ================================================================================

class EnsembleClassifier:
    """
    Combines YOLOv8-cls and ResNet18 into a single classifier.

    For each vehicle crop:
        1. YOLOv8-cls produces a probability vector
        2. ResNet18 produces a probability vector (with TTA)
        3. Both are averaged 50/50
        4. The class with the highest average probability is the prediction

    This is more reliable than either model alone because the two models
    are trained differently and tend to fail on different images.
    """

    def __init__(self, yolo_cls_weights: str, resnet_weights: str,
                 yolo_weight: float = 0.5, resnet_weight: float = 0.5,
                 conf_thresh: float = 0.35):
        """
        Parameters:
            yolo_cls_weights  — path to best.pt from train_yolo_cls_fold()
            resnet_weights    — path to resnet18_foldN.pt from train_resnet_fold()
            yolo_weight       — how much to trust YOLO's prediction (0-1)
            resnet_weight     — how much to trust ResNet's prediction (0-1)
            conf_thresh       — minimum confidence to call something "emergency"
                                below this threshold defaults to non-emergency
        """
        from ultralytics import YOLO
        self.yolo_cls    = YOLO(yolo_cls_weights)
        self.resnet      = build_resnet(pretrained=False)
        self.resnet.load_state_dict(
            torch.load(resnet_weights, map_location=DEVICE)
        )
        self.resnet.eval()
        self.yw          = yolo_weight
        self.rw          = resnet_weight
        self.conf_thresh = conf_thresh

    def predict(self, pil_img: Image.Image) -> tuple:
        """
        Classifies one cropped vehicle image.

        Returns:
            label  — "emergency" or "non-emergency"
            conf   — confidence score (0.0 to 1.0)
            probs  — full probability array [p_non_emergency, p_emergency]
        """
        # Get probability vector from YOLOv8-cls
        yolo_probs   = (self.yolo_cls(pil_img, verbose=False)[0]
                        .probs.data.cpu().numpy())

        # Get probability vector from ResNet18 with TTA
        resnet_probs = resnet_predict_tta(self.resnet, pil_img)

        # Weighted average of both probability vectors
        combined = self.yw * yolo_probs + self.rw * resnet_probs

        cls_id = int(np.argmax(combined))
        conf   = float(combined[cls_id])
        label  = CLASS_NAMES[cls_id]

        # If confidence is below threshold, default to non-emergency
        # This reduces false emergency alerts
        if conf < self.conf_thresh:
            label = "non-emergency"
            conf  = float(combined[0])

        return label, conf, combined


# ================================================================================
# PART 11 — EVALUATE ON HELD-OUT TEST FOLD
# ================================================================================

def evaluate_on_test_fold(cv_info: dict, image_paths: list, labels: list):
    """
    Final evaluation using the best fold's weights on the held-out test set.

    This fold was never seen during any training or validation.
    Prints classification report (precision, recall, F1 per class)
    and saves a confusion matrix image.
    """
    best         = cv_info["best"]
    test_indices = best["test_indices"]
    clf          = EnsembleClassifier(best["yolo_weights"], best["resnet_weights"])

    preds, trues = [], []
    print(f"\n[INFO] Evaluating on held-out test fold "
          f"({len(test_indices)} images) …")

    for idx in test_indices:
        try:
            pil        = Image.open(image_paths[idx]).convert("RGB")
            pred, _, _ = clf.predict(pil)
            preds.append(CLASS_NAMES.index(pred))
            trues.append(labels[idx])
        except Exception as e:
            print(f"  Skipping index {idx}: {e}")

    if not preds:
        print("[WARNING] No predictions — skipping evaluation.")
        return

    # Print precision, recall, F1 for each class
    print("\n" + classification_report(
        trues, preds,
        target_names=CLASS_NAMES,
        labels=list(range(len(CLASS_NAMES)))
    ))

    # Plot and save confusion matrix
    cm = confusion_matrix(trues, preds, labels=list(range(len(CLASS_NAMES))))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix — Held-Out Test Fold\n"
              f"(Best fold: {best['fold']}, "
              f"val_acc={best['ensemble_val_acc']:.4f})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix_test.png", dpi=130)
    plt.show()
    print("[INFO] Saved: confusion_matrix_test.png")


# ================================================================================
# PART 12 — DEEPSORT MULTI-OBJECT TRACKER
# ================================================================================

class DeepSORTTracker:
    """
    Wrapper around DeepSORT that maintains vehicle identities across frames.

    What DeepSORT does:
        - Kalman filter predicts where each vehicle will be next frame
          based on its current velocity and direction
        - MobileNet generates a visual "fingerprint" (appearance embedding)
          for each detected vehicle crop
        - Hungarian algorithm matches detections to existing tracks
          using both position (Kalman prediction) and appearance (embedding)
        - If a vehicle is temporarily hidden, DeepSORT can re-identify it
          when it reappears using its appearance embedding

    The majority-vote stabilisation over 15 frames prevents the label
    (emergency/non-emergency) from flickering frame-to-frame due to
    classification uncertainty.
    """

    def __init__(self, max_age: int = 30):
        """
        Parameters:
            max_age — how many frames a track survives without a detection
                      before being deleted (default 30 = ~1 second at 30fps)
        """
        from deep_sort_realtime.deepsort_tracker import DeepSort
        self.tracker = DeepSort(
            max_age      = max_age,
            embedder     = "mobilenet",               # appearance embedding network
            half         = torch.cuda.is_available(), # FP16 on GPU for speed
            embedder_gpu = torch.cuda.is_available(),
        )
        # Stores label history per track for majority-vote stabilisation
        # { track_id: ["emergency", "non-emergency", "emergency", ...] }
        self.history: dict = defaultdict(list)

    def update(self, detections: list, frame: np.ndarray) -> list:
        """
        Updates tracker with new detections from the current frame.

        Parameters:
            detections — list of [x1, y1, x2, y2, conf, label_string]
            frame      — current BGR frame (needed for appearance embedding)

        Returns:
            list of (track_id, x1, y1, x2, y2, stable_label)
        """
        # Convert to DeepSORT's expected format: ([x,y,w,h], conf, label)
        ds_input = [
            ([x1, y1, x2 - x1, y2 - y1], conf, lbl)
            for x1, y1, x2, y2, conf, lbl in detections
        ]

        tracks  = self.tracker.update_tracks(ds_input, frame=frame)
        outputs = []

        for tr in tracks:
            # Only process confirmed tracks (seen in 3+ consecutive frames)
            if not tr.is_confirmed():
                continue

            tid   = tr.track_id
            ltrb  = tr.to_ltrb()   # bounding box as (left, top, right, bottom)
            label = tr.get_det_class() or "unknown"

            # Add this frame's label to the history buffer
            self.history[tid].append(label)

            # Majority vote over the last 15 frames
            # e.g. if 12 of 15 say "emergency" → stable label is "emergency"
            recent = self.history[tid][-15:]
            stable = max(set(recent), key=recent.count)

            outputs.append((tid, *ltrb, stable))

        return outputs


# ================================================================================
# PART 13 — COMPLIANCE ENGINE TO EVALUATE YIELDING BEHAVIOUR
# ================================================================================

class ComplianceEngine:
    """
    Evaluates whether non-emergency vehicles are yielding to
    emergency vehicles.

    The compliance rule:
        A non-emergency vehicle is COMPLIANT if, within W frames of
        an emergency vehicle entering a proximity radius of R pixels,
        its centroid distance from the EV increases by at least D pixels.

        If the distance does NOT increase by D pixels → VIOLATION.

    Parameters:
        R = 250 pixels  — proximity zone radius
        W = 20 frames   — reaction window (~0.67 seconds at 30fps)
        D = 30 pixels   — minimum distance increase required to be compliant

    Why these values?
        W=20 at 30fps = 0.67 seconds — conservative lower bound for
        initial driver reaction time.
        R=250px and D=30px are calibrated for standard CCTV resolution.
        All three are tunable for different camera setups.
    """

    def __init__(self, proximity_r: int = 250,
                 reaction_w: int = 20, yield_d: int = 30):
        self.R = proximity_r   # proximity radius in pixels
        self.W = reaction_w    # reaction window in frames
        self.D = yield_d       # minimum yield distance in pixels

        # Rolling distance history for each (non_ev_id, ev_id) pair
        # { (non_ev_track_id, ev_track_id): [dist_t0, dist_t1, ...] }
        self.distance_history = defaultdict(list)

        # Tracks which vehicle pairs have already been flagged this event
        # so we don't log the same violation repeatedly
        self.already_flagged = set()

    @staticmethod
    def _centre(x1, y1, x2, y2):
        """Returns the centre point (cx, cy) of a bounding box."""
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _dist(cx1, cy1, cx2, cy2):
        """Returns Euclidean pixel distance between two centre points."""
        return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    def check(self, tracks: list, frame_id: int,
              frame: np.ndarray) -> list:
        """
        Main method — call once per frame after DeepSORT tracking.

        Parameters:
            tracks   — list of (track_id, x1, y1, x2, y2, label)
                       returned by DeepSORTTracker.update()
            frame_id — integer frame counter
            frame    — raw BGR numpy array for evidence image saving

        Returns:
            violations — list of violation dicts, one per detected violation
        """
        violations = []

        # ── Step 1: Separate EV tracks from non-EV tracks ───────────────────
        ev_tracks = [
            (tid, x1, y1, x2, y2)
            for tid, x1, y1, x2, y2, label in tracks
            if label == "emergency"
        ]
        non_ev_tracks = [
            (tid, x1, y1, x2, y2)
            for tid, x1, y1, x2, y2, label in tracks
            if label == "non-emergency"
        ]

        # ── Step 2: If no EV in frame, nothing to assess ─────────────────────
        if not ev_tracks:
            # Clear history — no point keeping it if EV has left
            self.distance_history.clear()
            self.already_flagged.clear()
            return violations

        # ── Step 3: Check each non-EV against each EV ────────────────────────
        for non_tid, nx1, ny1, nx2, ny2 in non_ev_tracks:
            ncx, ncy = self._centre(nx1, ny1, nx2, ny2)

            for ev_tid, ex1, ey1, ex2, ey2 in ev_tracks:
                ecx, ecy = self._centre(ex1, ey1, ex2, ey2)

                # Current pixel distance between this vehicle and the EV
                current_dist = self._dist(ncx, ncy, ecx, ecy)
                pair_key     = (non_tid, ev_tid)

                # ── Step 4: Outside proximity zone → stop watching ───────────
                if current_dist > self.R:
                    if pair_key in self.distance_history:
                        del self.distance_history[pair_key]
                    self.already_flagged.discard(pair_key)
                    continue

                # ── Step 5: Inside zone → record distance this frame ─────────
                history = self.distance_history[pair_key]
                history.append(current_dist)

                # Keep only the last W measurements
                if len(history) > self.W:
                    history.pop(0)

                # ── Step 6: Need W frames before judging ─────────────────────
                # Give the driver the full reaction window before flagging
                if len(history) < self.W:
                    continue

                # ── Step 7: Apply compliance rule ────────────────────────────
                # yield_amount = how much the vehicle moved away from EV
                # positive = moved away (good)
                # negative or small positive = did not yield (bad)
                dist_w_frames_ago = history[0]
                yield_amount      = current_dist - dist_w_frames_ago

                if yield_amount <= self.D:
                    # Vehicle has not yielded sufficiently

                    # Don't log the same pair twice for one continuous event
                    if pair_key in self.already_flagged:
                        continue

                    # Severity: Major if vehicle moved closer, Minor if just
                    # didn't move far enough away
                    severity = "Major" if yield_amount < 0 else "Minor"

                    violations.append({
                        "vehicle_id"  : non_tid,
                        "ev_id"       : ev_tid,
                        "frame_id"    : frame_id,
                        "distance_px" : round(current_dist, 2),
                        "yield_amount": round(yield_amount, 2),
                        "severity"    : severity,
                        "timestamp"   : datetime.now().isoformat(),
                        "frame"       : frame,
                        "bbox"        : (int(nx1), int(ny1), int(nx2), int(ny2)),
                    })

                    # Mark as flagged to prevent duplicate logging
                    self.already_flagged.add(pair_key)

                else:
                    # Vehicle IS yielding — clear flag so it can be flagged
                    # again if it later stops yielding
                    self.already_flagged.discard(pair_key)

        return violations


# ================================================================================
# PART 14 — VIOLATION LOGGER WITH SQLITE AND EVIDENCE IMAGES
# ================================================================================

class ViolationLogger:
    """
    Saves all violation events to:
        1. SQLite database (violations.db) — structured, queryable records
        2. JPEG evidence images — cropped screenshot of the offending vehicle

    Database schema (3 tables, Third Normal Form):
        Vehicle    — one row per tracked vehicle
        VideoFrame — one row per frame that contains a violation
        Violation  — one row per violation event, FK to Vehicle and VideoFrame
    """

    def __init__(self, db_path: str = "violations.db",
                 evidence_dir: str = "evidence_frames"):
        self.db_path      = db_path
        self.evidence_dir = evidence_dir

        # Create evidence folder if it doesn't exist
        os.makedirs(evidence_dir, exist_ok=True)

        # Connect to SQLite — check_same_thread=False allows use in video loop
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
        print(f"[INFO] Violation database ready at: {db_path}")
        print(f"[INFO] Evidence images will be saved to: {evidence_dir}/")

    def _create_tables(self):
        """
        Creates the three database tables.
        IF NOT EXISTS means re-running won't delete old data.
        """
        cur = self.conn.cursor()

        # Vehicle table — stores metadata about each tracked vehicle
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Vehicle (
                vehicle_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id    INTEGER NOT NULL UNIQUE,
                type        TEXT,
                class       TEXT
            )
        """)

        # VideoFrame table — stores metadata for each frame with a violation
        cur.execute("""
            CREATE TABLE IF NOT EXISTS VideoFrame (
                frame_id   INTEGER PRIMARY KEY,
                timestamp  TEXT NOT NULL,
                source     TEXT
            )
        """)

        # Violation table — the main record of each compliance failure
        # References Vehicle and VideoFrame via foreign keys
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Violation (
                violation_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id     INTEGER NOT NULL,
                frame_id       INTEGER NOT NULL,
                distance_px    REAL,
                severity       TEXT,
                evidence_path  TEXT,
                timestamp      TEXT,
                FOREIGN KEY (vehicle_id) REFERENCES Vehicle(vehicle_id),
                FOREIGN KEY (frame_id)   REFERENCES VideoFrame(frame_id)
            )
        """)

        self.conn.commit()

    def log(self, violation: dict):
        """
        Saves one violation to the database and writes the evidence image.

        Parameters:
            violation — dict produced by ComplianceEngine.check()
        """
        cur = self.conn.cursor()

        # 1. Insert vehicle (ignore if track_id already exists)
        cur.execute(
            "INSERT OR IGNORE INTO Vehicle (track_id, type, class) VALUES (?,?,?)",
            (violation["vehicle_id"], "non-emergency", "unknown")
        )
        cur.execute(
            "SELECT vehicle_id FROM Vehicle WHERE track_id = ?",
            (violation["vehicle_id"],)
        )
        vehicle_id = cur.fetchone()[0]

        # 2. Insert video frame (ignore if already logged)
        cur.execute(
            "INSERT OR IGNORE INTO VideoFrame (frame_id, timestamp, source) "
            "VALUES (?,?,?)",
            (violation["frame_id"], violation["timestamp"], "video_stream")
        )

        # 3. Save evidence image — crop of the offending vehicle
        evidence_path = self._save_evidence(
            frame      = violation["frame"],
            bbox       = violation["bbox"],
            vehicle_id = violation["vehicle_id"],
            frame_id   = violation["frame_id"],
        )

        # 4. Insert violation record
        cur.execute("""
            INSERT INTO Violation
                (vehicle_id, frame_id, distance_px, severity,
                 evidence_path, timestamp)
            VALUES (?,?,?,?,?,?)
        """, (
            vehicle_id,
            violation["frame_id"],
            violation["distance_px"],
            violation["severity"],
            evidence_path,
            violation["timestamp"],
        ))

        self.conn.commit()

        # Print to console so violations are visible during video processing
        print(
            f"  [VIOLATION] Frame {violation['frame_id']:05d} | "
            f"Track {violation['vehicle_id']:03d} | "
            f"Dist: {violation['distance_px']:.1f}px | "
            f"Severity: {violation['severity']}"
        )

    def log_many(self, violations: list, frame: np.ndarray):
        """Logs a list of violations — call this from process_frame()."""
        for v in violations:
            self.log(v)

    def _save_evidence(self, frame: np.ndarray, bbox: tuple,
                       vehicle_id: int, frame_id: int) -> str:
        """
        Crops the offending vehicle from the frame and saves as JPEG.
        Returns the file path string.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Clamp to frame boundaries to avoid out-of-bounds crop
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))

        filename = f"viol_track{vehicle_id:03d}_frame{frame_id:05d}.jpg"
        filepath = os.path.join(self.evidence_dir, filename)

        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                cv2.imwrite(filepath, crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

        return filepath

    def get_all_violations(self) -> list:
        """
        Returns all violations as a list of dicts.
        Use this in Streamlit to populate the violation table:
            df = pd.DataFrame(logger.get_all_violations())
            st.dataframe(df)
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT v.violation_id, v.timestamp, vh.track_id,
                   v.distance_px, v.severity, v.evidence_path, v.frame_id
            FROM Violation v
            JOIN Vehicle vh ON v.vehicle_id = vh.vehicle_id
            ORDER BY v.violation_id DESC
        """)
        rows = cur.fetchall()
        keys = ["violation_id", "timestamp", "track_id",
                "distance_px", "severity", "evidence_path", "frame_id"]
        return [dict(zip(keys, row)) for row in rows]

    def get_violation_count(self) -> int:
        """Returns total number of violations logged."""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM Violation")
        return cur.fetchone()[0]

    def close(self):
        """Close database connection cleanly when video processing ends."""
        self.conn.close()
        print("[INFO] Violation database connection closed.")


# ================================================================================
# PART 15 — EMERGENCY VEHICLE SYSTEM (INFERENCE + COMPLIANCE)
# ================================================================================

class EmergencyVehicleSystem:
    """
    The main inference class that ties everything together.

    Per frame it:
        1. Runs YOLOv8 detector to find all vehicles
        2. Crops each detection and classifies it with the ensemble
        3. Feeds all detections to DeepSORT for persistent tracking
        4. Passes tracks to ComplianceEngine to check yielding behaviour
        5. Logs any violations to the database
        6. Draws annotated bounding boxes and overlays on the frame
        7. Returns the annotated frame

    Colour coding on output video:
        Red   = emergency vehicle
        Green = compliant non-emergency vehicle
        Cyan  = unknown/unconfirmed vehicle
    """

    ALERT_COLOR  = (0, 0, 255)    # Red   — emergency vehicle
    NORMAL_COLOR = (0, 200, 0)    # Green — non-emergency vehicle
    UNK_COLOR    = (200, 200, 0)  # Cyan  — unknown

    def __init__(self, det_weights: str, yolo_cls_weights: str,
                 resnet_weights: str, det_conf: float = 0.3,
                 proximity_r: int = 250, reaction_w: int = 20,
                 yield_d: int = 30, db_path: str = "violations.db",
                 evidence_dir: str = "evidence_frames"):
        """
        Parameters:
            det_weights       — YOLO detection model weights
            yolo_cls_weights  — YOLOv8-cls classification weights
            resnet_weights    — ResNet18 weights
            det_conf          — minimum detection confidence (0.3)
            proximity_r       — compliance engine proximity radius in pixels
            reaction_w        — compliance engine reaction window in frames
            yield_d           — compliance engine minimum yield distance
            db_path           — path to SQLite database file
            evidence_dir      — folder to save evidence JPEG images
        """
        from ultralytics import YOLO

        print("[INFO] Loading detection model …")
        self.detector = YOLO(det_weights)

        print("[INFO] Loading ensemble classifier …")
        self.classifier = EnsembleClassifier(yolo_cls_weights, resnet_weights)

        print("[INFO] Initialising DeepSORT tracker …")
        self.tracker = DeepSORTTracker()

        print("[INFO] Initialising compliance engine …")
        self.compliance = ComplianceEngine(
            proximity_r = proximity_r,
            reaction_w  = reaction_w,
            yield_d     = yield_d,
        )

        print("[INFO] Initialising violation logger …")
        self.logger   = ViolationLogger(db_path, evidence_dir)
        self.det_conf = det_conf

    def _color(self, label: str) -> tuple:
        """Returns BGR colour for a given label string."""
        if label == "emergency":     return self.ALERT_COLOR
        if label == "non-emergency": return self.NORMAL_COLOR
        return self.UNK_COLOR

    def process_frame(self, frame: np.ndarray,
                      frame_id: int = 0) -> tuple:
        """
        Processes one video frame end-to-end.

        Parameters:
            frame    — BGR numpy array from OpenCV
            frame_id — integer frame counter (used for logging)

        Returns:
            annotated — frame with bounding boxes and labels drawn
            alerts    — list of track IDs of active emergency vehicles
        """
        H, W = frame.shape[:2]

        # ── Step 1: Detect all vehicles in this frame ────────────────────────
        results = self.detector(frame, conf=self.det_conf, verbose=False)[0]
        raw     = []   # will hold [x1, y1, x2, y2, conf, label] per detection

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                det_conf        = float(box.conf[0])

                # Crop the detected vehicle and convert BGR to RGB for PIL
                crop = Image.fromarray(
                    cv2.cvtColor(
                        frame[max(0,y1):min(H,y2), max(0,x1):min(W,x2)],
                        cv2.COLOR_BGR2RGB
                    )
                )

                # Classify the crop as emergency or non-emergency
                label, cls_conf, _ = self.classifier.predict(crop)

                # Combined confidence = detection conf × classification conf
                raw.append([x1, y1, x2, y2, det_conf * cls_conf, label])

        # ── Step 2: Update DeepSORT tracker with detections ─────────────────
        tracks = self.tracker.update(raw, frame)

        # ── Step 3: Run compliance check on current tracks ───────────────────
        violations = self.compliance.check(tracks, frame_id, frame)

        # ── Step 4: Log any violations to database ───────────────────────────
        self.logger.log_many(violations, frame)

        # ── Step 5: Draw annotations on frame ────────────────────────────────
        annotated = frame.copy()
        alerts    = []   # track IDs of emergency vehicles in this frame

        for tid, x1, y1, x2, y2, label in tracks:
            color = self._color(label)

            # Draw bounding box
            cv2.rectangle(annotated,
                          (int(x1), int(y1)), (int(x2), int(y2)),
                          color, 2)

            # Draw label with track ID
            cv2.putText(annotated,
                        f"ID:{tid} {label}",
                        (int(x1), max(int(y1) - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if label == "emergency":
                alerts.append(tid)

        # ── Step 6: Draw violation overlays ──────────────────────────────────
        for v in violations:
            # Draw red X on the violating vehicle's bounding box
            bx1, by1, bx2, by2 = v["bbox"]
            cv2.putText(annotated, "VIOLATION",
                        (bx1, by2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)

        # ── Step 7: Draw alert banner if any EV is present ───────────────────
        if alerts:
            cv2.putText(annotated,
                        f"EMERGENCY VEHICLE DETECTED (IDs: {alerts})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        self.ALERT_COLOR, 2)

        # Show violation count in corner
        vcount = self.logger.get_violation_count()
        cv2.putText(annotated,
                    f"Violations logged: {vcount}",
                    (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)

        return annotated, alerts

    def process_video(self, video_path: str,
                      output_path: str = "output.mp4",
                      show: bool = False):
        """
        Processes an entire video file frame by frame.

        Parameters:
            video_path  — path to input video file (MP4, AVI, MOV)
            output_path — path to write annotated output video
            show        — if True, display video in a window while processing
                          (press Q to quit early)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        # Get video properties
        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        print(f"[INFO] Processing: {video_path}")
        print(f"[INFO] Resolution: {W}x{H}  FPS: {fps:.1f}")

        # Set up video writer for annotated output
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (W, H)
        )

        frame_id     = 0
        alert_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break   # end of video

            # Process this frame
            annotated, alerts = self.process_frame(frame, frame_id)

            if alerts:
                alert_frames += 1

            # Write annotated frame to output video
            writer.write(annotated)

            # Optionally display live
            if show:
                cv2.imshow("Emergency Vehicle Compliance System", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Stopped early by user.")
                    break

            frame_id += 1
            if frame_id % 100 == 0:
                print(f"  Processed {frame_id} frames | "
                      f"EV alert frames: {alert_frames} | "
                      f"Violations: {self.logger.get_violation_count()}")

        # Clean up
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

        # Close database connection
        self.logger.close()

        print(f"\n[INFO] Done.")
        print(f"[INFO]   Total frames processed : {frame_id}")
        print(f"[INFO]   EV alert frames         : {alert_frames}")
        print(f"[INFO]   Total violations logged : {self.logger.get_violation_count()}")
        print(f"[INFO]   Annotated video saved   : {output_path}")
        print(f"[INFO]   Violation database      : {self.logger.db_path}")
        print(f"[INFO]   Evidence images          : {self.logger.evidence_dir}/")


# ================================================================================
# PART 16 — MAIN TRAINING PIPELINE
# ================================================================================

def main_train(k: int = 5, yolo_epochs: int = 8,
               resnet_epochs: int = 4, batch: int = 32):
    """
    Runs the full training pipeline:
        1. Download dataset from Kaggle
        2. Pool all images into one flat list
        3. Run K-Fold cross-validation (trains K-1 folds)
        4. Evaluate best fold on held-out test set
        5. Save CV summary to JSON

    After this function completes you will have:
        - kfold_cv_results.png      — bar chart of fold accuracies
        - confusion_matrix_test.png — confusion matrix on held-out test set
        - cv_summary.json           — accuracy stats
        - resnet18_foldN.pt         — ResNet18 weights for best fold
        - runs/classify/emerg_cls_foldN/weights/best.pt — YOLOv8-cls weights
    """
    print("=" * 60)
    print("  EMERGENCY VEHICLE CLASSIFICATION — TRAINING")
    print(f"  YOLOv8-cls + ResNet18 Ensemble  |  K={k}-Fold CV")
    print("=" * 60)

    # Step 1 — Download dataset
    dataset_path = download_dataset()

    # Step 2 — Pool all images
    image_paths, labels = collect_all_samples(dataset_path)

    # Step 3 — K-Fold cross-validation
    cv_info = run_kfold_cv(
        image_paths,
        labels,
        k             = k,
        yolo_epochs   = yolo_epochs,
        resnet_epochs = resnet_epochs,
        batch         = batch,
    )

    # Step 4 — Final evaluation on held-out test fold
    evaluate_on_test_fold(cv_info, image_paths, labels)

    # Step 5 — Save summary
    summary = {
        "k"           : k,
        "mean_acc"    : cv_info["mean_acc"],
        "std_acc"     : cv_info["std_acc"],
        "best_fold"   : cv_info["best"]["fold"],
        "best_val_acc": cv_info["best"]["ensemble_val_acc"],
    }
    Path("cv_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[INFO] CV summary saved to cv_summary.json")
    print(f"[INFO] Pipeline complete.")
    print(f"[INFO] Mean CV accuracy = {cv_info['mean_acc']:.4f} "
          f"± {cv_info['std_acc']:.4f}")

    return cv_info


# ================================================================================
# PART 17 — RUN ON VIDEO (CALL AFTER TRAINING)
# ================================================================================

def run_on_video(det_weights: str, yolo_cls_weights: str,
                 resnet_weights: str, video_path: str,
                 output_path: str = "output.mp4"):
    """
    Runs the full inference + compliance pipeline on a video file.

    Call this function AFTER main_train() has completed and you have
    your trained weight files.

    Parameters:
        det_weights      — path to detection YOLO weights
                           e.g. "runs/detect/emerg_det/weights/best.pt"
        yolo_cls_weights — path to classification YOLO weights
                           e.g. "runs/classify/emerg_cls_fold0/weights/best.pt"
        resnet_weights   — path to ResNet18 weights
                           e.g. "resnet18_fold0.pt"
        video_path       — path to your input traffic video
                           e.g. "traffic.mp4"
        output_path      — where to save annotated output video
                           e.g. "output.mp4"

    Outputs:
        output.mp4          — annotated video with bounding boxes and violations
        violations.db       — SQLite database of all logged violations
        evidence_frames/    — folder of JPEG evidence images per violation
    """
    print("[INFO] Initialising system …")
    system = EmergencyVehicleSystem(
        det_weights      = det_weights,
        yolo_cls_weights = yolo_cls_weights,
        resnet_weights   = resnet_weights,
        det_conf         = 0.3,    # detection confidence threshold
        proximity_r      = 250,    # compliance: proximity radius in pixels
        reaction_w       = 20,     # compliance: reaction window in frames
        yield_d          = 30,     # compliance: minimum yield distance in pixels
        db_path          = "violations.db",
        evidence_dir     = "evidence_frames",
    )

    system.process_video(
        video_path  = video_path,
        output_path = output_path,
        show        = False,   # set True to watch live in a window
    )


# ================================================================================
# ENTRY POINT
# ================================================================================

if __name__ == "__main__":

    # ── STEP 1: RUN TRAINING ─────────────────────────────────────────────────
    # Uncomment the lines below to train the models.
    # This takes time — run once and save the weights.

    cv_info = main_train(
        k             = 5,   # number of folds
        yolo_epochs   = 8,   # epochs per fold for YOLOv8-cls
        resnet_epochs = 4,   # epochs per fold for ResNet18
        batch         = 32,  # batch size
    )

    # ── STEP 2: RUN ON VIDEO ─────────────────────────────────────────────────
    # After training completes, update these paths to your actual weight files
    # (Ultralytics prints the save path at the end of each training run)
    # then call run_on_video().

    # Example (update paths to match your actual output):
    # run_on_video(
    #     det_weights      = "runs/detect/emerg_det/weights/best.pt",
    #     yolo_cls_weights = "runs/classify/emerg_cls_fold0/weights/best.pt",
    #     resnet_weights   = "resnet18_fold0.pt",
    #     video_path       = "traffic.mp4",
    #     output_path      = "output.mp4",
    # )
