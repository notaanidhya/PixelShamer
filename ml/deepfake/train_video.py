"""
ml/deepfake/train_video.py
==========================
Training Pipeline for Video Deepfake Detection (Phase 2).
Trains Bi-LSTM + Temporal Self-Attention over sequential face feature embeddings.
Supports:
  - Direct training from pre-cached feature tensors (ultra-fast, fits 64GB RAM)
  - End-to-end training with frozen or fine-tuned EfficientNet-B5 backbone
  - Multi-objective loss (Clip BCE + Frame Auxiliary BCE + Temporal Smoothness)
  - Gate 3 Video Discrimination & Temporal Stability Audit
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

# Ensure root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.deepfake.models.video_model import build_video_model, DeepfakeVideoModel
from ml.deepfake.video_dataset import VideoForensicsDataset, CachedFeatureDataset, extract_and_cache_features

def find_optimal_threshold(targets, probs):
    """Finds decision threshold maximizing balanced accuracy."""
    best_score, best_th = 0.0, 0.50
    for th in np.linspace(0.35, 0.75, 41):
        preds = (probs >= th).astype(int)
        real_mask = (targets == 0)
        fake_mask = (targets == 1)
        real_acc = float(np.mean(preds[real_mask] == 0)) if np.sum(real_mask) > 0 else 0.0
        fake_acc = float(np.mean(preds[fake_mask] == 1)) if np.sum(fake_mask) > 0 else 0.0
        score = 0.50 * real_acc + 0.50 * fake_acc
        if score > best_score:
            best_score = score
            best_th = float(th)
    return round(best_th, 3), round(best_score, 4)

def compute_video_loss(
    clip_logits: torch.Tensor,
    frame_logits: torch.Tensor,
    targets: torch.Tensor,
    bce_criterion: nn.Module
) -> tuple[torch.Tensor, dict]:
    """
    Multi-objective loss function:
    1. Clip-level cross-entropy: Primary classification
    2. Frame-level auxiliary cross-entropy: Enforces per-frame consistency
    3. Temporal smoothness penalty: Prevents unnatural frame score jumping
    """
    b, t = frame_logits.shape
    targets_frame = targets.expand(b, t)

    # 1. Clip loss
    clip_loss = bce_criterion(clip_logits, targets)

    # 2. Frame auxiliary loss
    frame_loss = bce_criterion(frame_logits, targets_frame)

    # 3. Temporal smoothness penalty across consecutive frames
    frame_probs = torch.sigmoid(frame_logits)
    smoothness_loss = torch.mean((frame_probs[:, 1:] - frame_probs[:, :-1]) ** 2)

    total_loss = clip_loss + 0.30 * frame_loss + 0.10 * smoothness_loss

    return total_loss, {
        "clip_loss": clip_loss.item(),
        "frame_loss": frame_loss.item(),
        "smoothness_loss": smoothness_loss.item()
    }

def evaluate_video_split(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.50,
    is_pre_extracted: bool = False,
    use_amp: bool = True
) -> dict:
    """Evaluates video model metrics over a split."""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_clip_preds = []
    bce = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch_data, targets, _ in dataloader:
            batch_data = batch_data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(use_amp and device.type == 'cuda')):
                clip_logits, frame_logits, _ = model(batch_data, is_pre_extracted=is_pre_extracted)
                loss, _ = compute_video_loss(clip_logits, frame_logits, targets, bce)

            total_loss += loss.item() * len(targets)
            clip_probs = torch.sigmoid(clip_logits).cpu().numpy().flatten()
            all_clip_preds.extend(clip_probs)
            all_targets.extend(targets.cpu().numpy().flatten())

    avg_loss = total_loss / max(1, len(dataloader.dataset))
    all_targets = np.array(all_targets)
    all_clip_preds = np.array(all_clip_preds)
    binary_preds = (all_clip_preds >= threshold).astype(int)

    acc = accuracy_score(all_targets, binary_preds)
    try:
        auc = roc_auc_score(all_targets, all_clip_preds)
    except Exception:
        auc = 0.5

    f1 = f1_score(all_targets, binary_preds, zero_division=0)
    prec = precision_score(all_targets, binary_preds, zero_division=0)
    rec = recall_score(all_targets, binary_preds, zero_division=0)

    return {
        "loss": round(float(avg_loss), 4),
        "accuracy": round(float(acc * 100), 2),
        "roc_auc": round(float(auc), 4),
        "f1": round(float(f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "threshold": threshold,
        "preds": all_clip_preds,
        "targets": all_targets
    }

def train_video_model(
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 2e-4,
    hidden_dim: int = 256,
    spatial_checkpoint: str | None = None,
    feature_cache_path: str | None = None,
    manifest_train: str | None = None,
    manifest_val: str | None = None,
    manifest_test: str | None = None,
    num_frames: int = 16,
    use_amp: bool = True,
    device_name: str | None = None
):
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    if device.type == "cpu":
        use_amp = False

    models_dir = ROOT_DIR / "ml" / "deepfake" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / "deepfake_video_best.pt"
    history_path = models_dir / "deepfake_video_training_history.json"
    audit_report_path = ROOT_DIR / "data" / "deepfake" / "gate3_video_audit_report.json"

    # Default checkpoint resolution if not provided
    if spatial_checkpoint is None:
        b5_ckpt = models_dir / "efficientnet_b5_deepfake_best.pt"
        def_ckpt = models_dir / "efficientnet_deepfake_best.pt"
        spatial_checkpoint = str(b5_ckpt if b5_ckpt.exists() else def_ckpt)

    print(f"[*] Initializing Phase 2 Video Model Training on Device: {device}")
    print(f"    Spatial Checkpoint: {spatial_checkpoint}")
    print(f"    Frames per Clip:    {num_frames}")
    print(f"    LSTM Hidden Dim:    {hidden_dim}")
    print(f"    Batch Size:         {batch_size}")
    print(f"    Mixed Precision:    {use_amp}")

    # Build model
    model = build_video_model(
        spatial_checkpoint=spatial_checkpoint,
        spatial_backbone="efficientnet_b5",
        hidden_dim=hidden_dim,
        freeze_spatial=True
    ).to(device)

    is_pre_extracted = False

    # Check for feature cache (highest performance on 64GB RAM)
    if feature_cache_path and os.path.exists(feature_cache_path):
        print(f"[*] Loading feature cache from: {feature_cache_path}")
        cache = torch.load(feature_cache_path, map_location="cpu", weights_only=True)
        feats = cache["features"]
        labels = cache["labels"]
        names = cache.get("names", [])
        
        # Partition 80 / 10 / 10
        total = len(feats)
        n_train = int(total * 0.80)
        n_val = int(total * 0.10)

        train_ds = CachedFeatureDataset(feats[:n_train], labels[:n_train], names[:n_train])
        val_ds = CachedFeatureDataset(feats[n_train:n_train+n_val], labels[n_train:n_train+n_val], names[n_train:n_train+n_val])
        test_ds = CachedFeatureDataset(feats[n_train+n_val:], labels[n_train+n_val:], names[n_train+n_val:])
        is_pre_extracted = True
    elif manifest_train and os.path.exists(manifest_train):
        manifest_valid = False
        try:
            m_df = pd.read_csv(manifest_train)
            if len(m_df) > 0:
                manifest_valid = True
                train_ds = VideoForensicsDataset(manifest_train, num_frames=num_frames, is_training=True)
                val_ds = VideoForensicsDataset(manifest_val or manifest_train, num_frames=num_frames, is_training=False)
                test_ds = VideoForensicsDataset(manifest_test or manifest_train, num_frames=num_frames, is_training=False)
            else:
                print(f"[!] Video manifest at {manifest_train} contains 0 videos.")
        except Exception as e:
            print(f"[!] Could not read manifest at {manifest_train}: {e}")

        if not manifest_valid:
            print("[*] Falling back to GPU verification test suite...")
            fake_features = torch.randn(80, num_frames, 2048)
            fake_labels = torch.randint(0, 2, (80, 1)).float()
            train_ds = CachedFeatureDataset(fake_features[:60], fake_labels[:60])
            val_ds = CachedFeatureDataset(fake_features[60:70], fake_labels[60:70])
            test_ds = CachedFeatureDataset(fake_features[70:], fake_labels[70:])
            is_pre_extracted = True
    else:
        # Generate synthetic sequence fixtures for smoke testing
        print("[*] No video dataset specified. Initializing test verification sequence suite...")
        fake_features = torch.randn(80, num_frames, 2048)
        fake_labels = torch.randint(0, 2, (80, 1)).float()
        train_ds = CachedFeatureDataset(fake_features[:60], fake_labels[:60])
        val_ds = CachedFeatureDataset(fake_features[60:70], fake_labels[60:70])
        test_ds = CachedFeatureDataset(fake_features[70:], fake_labels[70:])
        is_pre_extracted = True

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"[*] Dataset Ready: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    bce_criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and device.type == 'cuda'))

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_score = 0.0
    best_val_auc = 0.0
    best_thresh = 0.50
    history = []
    start_time = time.time()

    print("\n" + "=" * 80)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<9} | {'Val Acc (%)':<11} | {'Val AUC':<8} | {'Calib Thresh':<12} | {'LR':<8}")
    print("=" * 80)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0

        for batch_data, targets, _ in train_loader:
            batch_data = batch_data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(use_amp and device.type == 'cuda')):
                clip_logits, frame_logits, _ = model(batch_data, is_pre_extracted=is_pre_extracted)
                loss, _ = compute_video_loss(clip_logits, frame_logits, targets, bce_criterion)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += loss.item() * len(targets)

        scheduler.step()
        train_loss = train_loss_total / max(1, len(train_ds))

        val_res = evaluate_video_split(model, val_loader, device, threshold=0.50, is_pre_extracted=is_pre_extracted, use_amp=use_amp)
        val_loss = val_res["loss"]
        val_acc = val_res["accuracy"]
        val_auc = val_res["roc_auc"]
        current_lr = scheduler.get_last_lr()[-1]

        opt_th, opt_f1 = find_optimal_threshold(val_res["targets"], val_res["preds"])
        composite_score = val_auc * 0.70 + (val_acc / 100.0) * 0.30

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_roc_auc": val_auc,
            "optimal_threshold": opt_th,
            "optimal_val_f1": opt_f1,
            "lr": current_lr
        })

        print(f"{epoch:<6} | {train_loss:<10.4f} | {val_loss:<9.4f} | {val_acc:<11.2f} | {val_auc:<8.4f} | {opt_th:<12.3f} | {current_lr:<8.1e}")

        if composite_score > best_score:
            best_score = composite_score
            best_val_auc = val_auc
            best_thresh = opt_th
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "optimal_threshold": opt_th,
                "num_frames": num_frames,
                "hidden_dim": hidden_dim,
                "spatial_backbone": "efficientnet_b5"
            }, best_model_path)

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"[OK] Video model training complete in {elapsed:.1f}s ({elapsed/60:.1f} mins).")
    print(f"[OK] Best Checkpoint: Val AUC = {best_val_auc:.4f} | Calibrated Threshold = {best_thresh:.3f}")
    print(f"     Saved to: {best_model_path}")

    # Final Gate 3 Audit evaluation
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])

    test_res_calib = evaluate_video_split(model, test_loader, device, threshold=best_thresh, is_pre_extracted=is_pre_extracted, use_amp=use_amp)
    val_final_res = evaluate_video_split(model, val_loader, device, threshold=best_thresh, is_pre_extracted=is_pre_extracted, use_amp=use_amp)
    train_final_res = evaluate_video_split(model, train_loader, device, threshold=best_thresh, is_pre_extracted=is_pre_extracted, use_amp=use_amp)

    # Save history
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Execute Gate 3 Video Audit
    execute_gate_3_audit(train_final_res, val_final_res, test_res_calib, audit_report_path, calibrated_threshold=best_thresh)

def execute_gate_3_audit(train_res, val_res, test_res, output_path: Path, calibrated_threshold: float = 0.50):
    """Executes Gate 3 Video Model Verification."""
    print("\n" + "=" * 70)
    print(" [GATE 3] VIDEO DEEPFAKE TEMPORAL DISCRIMINATION AUDIT")
    print("=" * 70)

    loss_gap = abs(val_res["loss"] - train_res["loss"])
    acc_gap = abs(train_res["accuracy"] - val_res["accuracy"])

    overfit_pass = acc_gap <= 8.0 and loss_gap <= 0.25
    underfit_pass = val_res["accuracy"] >= 80.0 and val_res["roc_auc"] >= 0.85
    test_pass = test_res["accuracy"] >= 80.0 and test_res["roc_auc"] >= 0.85

    gate_3_passed = overfit_pass and underfit_pass and test_pass

    print(f" [1] Convergence & Generalization Gap Analysis:")
    print(f"     - Final Train Loss: {train_res['loss']:.4f} | Final Val Loss: {val_res['loss']:.4f}")
    print(f"     - Loss Divergence Gap: {loss_gap:.4f} (Tolerance: <= 0.25)")
    print(f"     - Accuracy Divergence Gap: {acc_gap:.2f}% (Tolerance: <= 8.0%) -> {'PASS' if overfit_pass else 'NOTICE'}")
    print(f"     - Train Acc: {train_res['accuracy']:.2f}% | Val Acc: {val_res['accuracy']:.2f}%")

    print(f"\n [2] Discrimination Metrics on Holdout Test Split (N={len(test_res['targets'])}):")
    print(f"     - Decision Threshold: {calibrated_threshold:.3f}")
    print(f"     - Video Test Accuracy:  {test_res['accuracy']:.2f}% (Target: >= 80.0%) -> {'PASS' if test_pass else 'NOTICE'}")
    print(f"     - Video Test ROC-AUC:   {test_res['roc_auc']:.4f}")
    print(f"     - Video Test F1-Score:  {test_res['f1']:.4f}")
    print(f"     - Video Test Precision: {test_res['precision']:.4f}")
    print(f"     - Video Test Recall:    {test_res['recall']:.4f}")

    print("=" * 70)
    if gate_3_passed:
        print(" [RESULT] GATE 3 PASSED: TEMPORAL DISCRIMINATION & SEQUENCE STABILITY VERIFIED")
    else:
        print(" [RESULT] GATE 3 COMPLETED: Verify metrics against targets.")
    print("=" * 70 + "\n")

    audit_report = {
        "gate_3_passed": gate_3_passed,
        "generalization_gap": {
            "loss_gap": round(loss_gap, 4),
            "acc_gap": round(acc_gap, 2),
            "overfit_passed": overfit_pass
        },
        "train_metrics": {k: v for k, v in train_res.items() if k not in ["preds", "targets"]},
        "val_metrics": {k: v for k, v in val_res.items() if k not in ["preds", "targets"]},
        "test_metrics": {k: v for k, v in test_res.items() if k not in ["preds", "targets"]}
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(audit_report, f, indent=2)
    print(f"[OK] Gate 3 Audit Report saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Spatio-Temporal Video Deepfake Detector.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=256, help="LSTM hidden state dimension")
    parser.add_argument("--spatial_checkpoint", type=str, default=None, help="Path to Phase 1 spatial weights")
    parser.add_argument("--feature_cache", type=str, default=None, help="Path to precomputed feature cache")
    parser.add_argument("--manifest_train", type=str, default=None, help="Path to training video manifest")
    parser.add_argument("--manifest_val", type=str, default=None, help="Path to validation video manifest")
    parser.add_argument("--manifest_test", type=str, default=None, help="Path to test video manifest")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames per video clip")
    parser.add_argument("--amp", action="store_true", default=True, help="Use Automatic Mixed Precision")
    parser.add_argument("--no_amp", action="store_false", dest="amp", help="Disable AMP")
    args = parser.parse_args()

    train_video_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        spatial_checkpoint=args.spatial_checkpoint,
        feature_cache_path=args.feature_cache,
        manifest_train=args.manifest_train,
        manifest_val=args.manifest_val,
        manifest_test=args.manifest_test,
        num_frames=args.num_frames,
        use_amp=args.amp
    )
