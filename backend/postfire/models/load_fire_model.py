"""
load_fire_model.py
==================
Loader utility for lstm_fire_model.pth

Place this file in the SAME folder as:
  - lstm_fire_model.pth
  - lstm_scaler.pkl
  - lstm_imputer.pkl

Works regardless of whether the saved model is an LSTM or Transformer —
the correct architecture is reconstructed automatically from the config
stored inside the .pth checkpoint.

Usage
-----
    from load_fire_model import load_fire_model, predict_fire_risk

    model, scaler, imputer, threshold, meta = load_fire_model("path/to/models")

    # X_raw shape: (n_samples, sequence_length, n_features) — RAW unscaled values
    probs, labels = predict_fire_risk(model, scaler, imputer, X_raw, threshold,
                                      sequence_length=meta["sequence_length"])
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# ── Architecture definitions (must exactly match lstm_trainer.py) ─────────────

class AttentionLayer(nn.Module):
    """Additive attention over LSTM time steps."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        scores  = self.attn(lstm_out).squeeze(-1)            # (B, T)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1) # (B, T, 1)
        return (lstm_out * weights).sum(dim=1)               # (B, H)


class FireRiskLSTM(nn.Module):
    """Bidirectional LSTM with attention — outputs raw logits."""
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 dropout=0.3, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention  = AttentionLayer(out_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(out_size),
            nn.Linear(out_size, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        return self.classifier(context).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        import math
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class FireRiskTransformer(nn.Module):
    """Transformer encoder — outputs raw logits."""
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3,
                 dim_feedforward=512, dropout=0.3):
        super().__init__()
        self.proj    = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.pos_enc(self.proj(x))
        x = self.encoder(x)
        return self.classifier(x[:, -1]).squeeze(-1)


# ── Loader ────────────────────────────────────────────────────────────────────

def load_fire_model(model_dir=".", device="cpu"):
    """
    Load the best wildfire model from model_dir.

    The correct architecture (LSTM or Transformer) is reconstructed
    automatically from the 'config' key stored in the checkpoint —
    you do NOT need to know which model won at training time.

    Parameters
    ----------
    model_dir : str or Path
        Folder containing lstm_fire_model.pth, lstm_scaler.pkl, lstm_imputer.pkl
    device : str
        'cpu' or 'cuda'

    Returns
    -------
    model     : nn.Module   — in eval() mode, on `device`
    scaler    : StandardScaler
    imputer   : SimpleImputer
    threshold : float       — optimal classification threshold (max-F1)
    meta      : dict        — full checkpoint (feature_names, config, AUC, …)
    """
    model_dir   = Path(model_dir)
    ckpt_path   = model_dir / "lstm_fire_model.pth"
    scaler_path = model_dir / "lstm_scaler.pkl"
    imputer_path = model_dir / "lstm_imputer.pkl"

    for p in [ckpt_path, scaler_path, imputer_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                f"Make sure lstm_fire_model.pth, lstm_scaler.pkl, "
                f"and lstm_imputer.pkl are all in: {model_dir}"
            )

    ckpt   = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]   # ← the full architecture config saved by trainer

    # Reconstruct the correct architecture from saved config
    if config["model_type"] == "transformer":
        model = FireRiskTransformer(
            input_size      = ckpt["input_size"],
            d_model         = config["d_model"],
            nhead           = config["nhead"],
            num_layers      = config["num_layers"],
            dim_feedforward = config["dim_feedforward"],
            dropout         = config["dropout"],
        )
    elif config["model_type"] == "lstm":
        model = FireRiskLSTM(
            input_size    = ckpt["input_size"],
            hidden_size   = config["hidden_size"],
            num_layers    = config["num_layers"],
            dropout       = config["dropout"],
            bidirectional = config["bidirectional"],
        )
    else:
        raise ValueError(f"Unknown model_type in checkpoint: {config['model_type']!r}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    with open(scaler_path,  "rb") as f: scaler  = pickle.load(f)
    with open(imputer_path, "rb") as f: imputer = pickle.load(f)

    print(f"✓ Loaded model  : {ckpt['model_name']}")
    print(f"  Architecture  : {config}")
    print(f"  Features      : {ckpt['n_features']}")
    print(f"  Sequence len  : {ckpt['sequence_length']}")
    print(f"  Threshold     : {ckpt['optimal_threshold']:.4f}  ({ckpt.get('threshold_strategy','n/a')})")
    if "achieved_precision" in ckpt:
        print(f"  Precision     : {ckpt['achieved_precision']:.4f}")
        print(f"  Recall        : {ckpt['achieved_recall']:.4f}")
    if "test_roc_auc" in ckpt:
        print(f"  Test ROC-AUC  : {ckpt['test_roc_auc']:.4f}")
    if "test_pr_auc" in ckpt:
        print(f"  Test PR-AUC   : {ckpt['test_pr_auc']:.4f}")
    if "trained_on" in ckpt:
        print(f"  Trained on    : {ckpt['trained_on']}")

    return model, scaler, imputer, ckpt["optimal_threshold"], ckpt


# ── Inference helper ──────────────────────────────────────────────────────────

def predict_fire_risk(model, scaler, imputer, X_raw,
                      threshold, sequence_length, device="cpu"):
    """
    Preprocess and run inference on raw feature windows.

    Parameters
    ----------
    model           : loaded nn.Module (from load_fire_model)
    scaler          : loaded StandardScaler
    imputer         : loaded SimpleImputer
    X_raw           : np.ndarray, shape (n_samples, sequence_length, n_features)
                      Raw, UNSCALED feature values.
    threshold       : float — classification threshold (from load_fire_model)
    sequence_length : int   — must match training (use meta["sequence_length"])
    device          : str

    Returns
    -------
    probs  : np.ndarray (n_samples,) — fire probability 0–1
    labels : np.ndarray (n_samples,) — 1=FIRE, 0=no fire
    """
    if X_raw.ndim != 3:
        raise ValueError(f"X_raw must be 3-D (samples, seq_len, features), got {X_raw.shape}")

    n_samples, seq_len, n_feat = X_raw.shape

    if seq_len != sequence_length:
        raise ValueError(
            f"Sequence length mismatch: model expects {sequence_length}, "
            f"got {seq_len}. Adjust your sliding window."
        )

    # Flatten → impute → scale → reshape
    flat   = X_raw.reshape(-1, n_feat).astype(np.float32)
    flat   = imputer.transform(flat)
    flat   = scaler.transform(flat)
    X_proc = flat.reshape(n_samples, seq_len, n_feat)

    tensor = torch.FloatTensor(X_proc).to(device)
    model.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).cpu().numpy()

    labels = (probs >= threshold).astype(int)
    return probs, labels


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

    print(f"Loading model from: {model_dir}\n")
    model, scaler, imputer, threshold, meta = load_fire_model(model_dir)

    seq_len  = meta["sequence_length"]
    n_feat   = meta["n_features"]
    features = meta["feature_names"]

    print(f"\nExpected feature order (first 10):")
    for i, f in enumerate(features[:10]):
        print(f"  [{i:2d}] {f}")
    if len(features) > 10:
        print(f"  ... and {len(features)-10} more")

    # Fake batch of 4 sequences with random data
    print(f"\nRunning smoke-test on 4 random samples ...")
    X_dummy = np.random.randn(4, seq_len, n_feat).astype(np.float32)
    probs, labels = predict_fire_risk(
        model, scaler, imputer, X_dummy, threshold, seq_len
    )

    print("\nSmoke-test results:")
    for i, (p, l) in enumerate(zip(probs, labels)):
        tag = "🔥 FIRE" if l else "✓ no fire"
        print(f"  Sample {i+1}: prob={p:.4f}  →  {tag}")

    print("\n✓ Loader is working correctly.")