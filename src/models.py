"""
models.py
=========
PyTorch models for acoustic pipe inspection.

Two models are provided:

1. **BlockageDetector** (2-D CNN on spectrograms)
   Binary classification: blocked vs. clear.

2. **BlockageLocaliser** (1-D CNN on raw waveform)
   Regression: predict the normalised position of the primary blockage.

Both share a common training loop utility at the bottom.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# 1. Blockage Detector — 2-D CNN on spectrogram
# ---------------------------------------------------------------------------

class BlockageDetector(nn.Module):
    """
    Lightweight 2-D CNN that classifies a spectrogram as:
      0 = clear pipe
      1 = blockage present

    Input shape : (B, 1, F, T)  — single-channel spectrogram
    Output shape: (B, 2)         — logits for [clear, blocked]
    """

    def __init__(self, freq_bins: int = 193, time_bins: int = 64):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # → F/2, T/2

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # → F/4, T/4

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),              # fixed 4×4 output
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# 2. Blockage Localiser — 1-D CNN on raw waveform
# ---------------------------------------------------------------------------

class BlockageLocaliser(nn.Module):
    """
    1-D CNN that regresses the normalised position of the primary blockage
    from the raw received waveform.

    Input shape : (B, 1, N)  — single-channel raw signal
    Output shape: (B, 1)      — predicted normalised position in [0, 1]

    Only meaningful when a blockage is present; use DetectorModel first to
    gate predictions.
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            # Strided convolutions to compress the long time-series
            nn.Conv1d(1, 16, kernel_size=15, stride=4, padding=7),
            nn.ReLU(inplace=True),

            nn.Conv1d(16, 32, kernel_size=9, stride=4, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(16),   # → (B, 128, 16) regardless of input length
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),               # output in (0, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.encoder(x))


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification (Lin et al., RetinaNet 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy examples so training focuses on hard, misclassified ones.
    Particularly effective when class imbalance makes most samples trivially correct.

    Parameters
    ----------
    alpha : float
        Weighting factor for the positive (blocked/abnormal) class.
        Complement (1-alpha) applies to the negative class.
    gamma : float
        Focusing parameter. gamma=0 recovers standard cross-entropy.
        gamma=2.0 is the value used in the original paper.
    weight : Tensor, optional
        Additional per-class weights (same as CrossEntropyLoss weight).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Weighted cross-entropy per sample (loss magnitude includes class weight)
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        # p_t from raw logits — decoupled from class weights so focal modulation
        # measures genuine classification difficulty, not weight-inflated difficulty.
        log_pt = F.log_softmax(logits, dim=1).gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        pt = torch.exp(log_pt)
        # alpha_t: alpha for positive class, (1-alpha) for negative class
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(ce, self.alpha),
            torch.full_like(ce, 1.0 - self.alpha),
        )
        return (alpha_t * (1.0 - pt) ** self.gamma * ce).mean()


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_detector(
    spectrograms: np.ndarray,   # shape (N, 1, F, T)
    labels: np.ndarray,         # shape (N,)  — int64
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_split: float = 0.2,
    save_path: Optional[str] = None,
    device: Optional[str] = None,
    class_weights: Optional[np.ndarray] = None,
    focal_alpha: float = 0.25,
) -> Tuple[BlockageDetector, Dict]:
    """
    Train the BlockageDetector on labelled spectrograms.

    Returns
    -------
    model   : trained BlockageDetector
    history : dict with keys 'train_loss', 'val_loss', 'val_acc'
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.tensor(spectrograms, dtype=torch.float32)
    y = torch.tensor(labels,       dtype=torch.long)

    dataset = TensorDataset(X, y)
    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    # Infer freq/time dims from data
    _, _, F, T = spectrograms.shape
    model = BlockageDetector(freq_bins=F, time_bins=T).to(device)

    # Class-weighted Focal Loss to handle imbalanced datasets.
    # If not provided, compute inverse-frequency weights from training labels.
    if class_weights is None:
        counts = np.bincount(labels, minlength=2).astype(np.float32)
        weights = torch.tensor(counts.sum() / (2.0 * counts + 1e-8), dtype=torch.float32).to(device)
    else:
        weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=focal_alpha, gamma=2.0, weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: Dict = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        # --- Validate ---
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(xb)
                correct  += (logits.argmax(1) == yb).sum().item()
        val_loss /= n_val
        val_acc   = correct / n_val

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.3f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved -> {save_path}")

    return model, history


def train_localiser(
    signals: np.ndarray,                        # shape (N, signal_len)
    positions: np.ndarray,                      # shape (N,) — primary blockage normalised pos
    physics_positions: Optional[np.ndarray] = None,  # shape (N,) — DSP-derived pos; -1 where unavailable
    physics_lambda: float = 0.1,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_split: float = 0.2,
    save_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[BlockageLocaliser, Dict]:
    """
    Train the BlockageLocaliser on signals containing at least one blockage.

    Physics consistency loss
    ------------------------
    When `physics_positions` is provided, the training loss includes a term that
    penalises the CNN prediction for being inconsistent with what peak-detection
    DSP estimates from the same signal:

        L = L_mse + lambda * L_physics

    where L_physics = mean(|pred - pos_dsp|) over samples where DSP succeeded
    (physics_positions[i] >= 0). This grounds the CNN in the signal's echo
    timing rather than pattern-matching alone.

    Parameters
    ----------
    physics_positions : array, shape (N,)
        DSP-derived normalised position per sample. Set to -1 where peak
        detection found no echo (DSP failure or clear-pipe signal).
    physics_lambda : float
        Weight of the physics consistency term (default 0.1).

    Returns
    -------
    model   : trained BlockageLocaliser
    history : dict with keys 'train_loss', 'val_loss', 'val_mae', 'val_physics_mae'
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    X  = torch.tensor(signals[:, np.newaxis, :], dtype=torch.float32)   # (N,1,L)
    y  = torch.tensor(positions[:, np.newaxis],  dtype=torch.float32)   # (N,1)

    has_physics = physics_positions is not None
    if has_physics:
        yp = torch.tensor(physics_positions[:, np.newaxis], dtype=torch.float32)  # (N,1)
        dataset = TensorDataset(X, y, yp)
    else:
        dataset = TensorDataset(X, y)

    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model     = BlockageLocaliser().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_physics_train = 0
    if has_physics:
        # Count training samples where DSP succeeded
        train_idx = list(train_ds.indices)
        n_physics_train = int((physics_positions[train_idx] >= 0).sum())
        print(f"Physics consistency loss: lambda={physics_lambda}  "
              f"DSP-valid training samples={n_physics_train}/{n_train}")

    history: Dict = {"train_loss": [], "val_loss": [], "val_mae": [], "val_physics_mae": []}

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if has_physics:
                xb, yb, yp_b = batch
                yp_b = yp_b.to(device)
            else:
                xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)

            loss = criterion(preds, yb)

            if has_physics:
                # Physics consistency: penalise only where DSP succeeded (yp >= 0)
                valid = (yp_b >= 0).squeeze(1)
                if valid.any():
                    loss = loss + physics_lambda * torch.abs(
                        preds[valid] - yp_b[valid]
                    ).mean()

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        # --- Validate ---
        model.eval()
        val_loss, mae, physics_mae, physics_n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                if has_physics:
                    xb, yb, yp_b = batch
                    yp_b = yp_b.to(device)
                else:
                    xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                preds    = model(xb)
                val_loss += criterion(preds, yb).item() * len(xb)
                mae      += torch.abs(preds - yb).sum().item()
                if has_physics:
                    valid = (yp_b >= 0).squeeze(1)
                    if valid.any():
                        physics_mae += torch.abs(preds[valid] - yp_b[valid]).sum().item()
                        physics_n   += valid.sum().item()
        val_loss /= n_val
        mae       /= n_val
        p_mae = physics_mae / physics_n if physics_n > 0 else float('nan')

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(mae)
        history["val_physics_mae"].append(p_mae)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            phys_str = f"  physics_MAE={p_mae:.4f}" if has_physics else ""
            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:.5f}  "
                  f"val_loss={val_loss:.5f}  "
                  f"val_MAE={mae:.4f}{phys_str}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved -> {save_path}")

    return model, history
