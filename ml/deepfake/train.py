"""
ml/deepfake/train.py
===================
Modernized Deepfake Training Pipeline (Phase 1).
Features:
  - PyTorch Automatic Mixed Precision (AMP / FP16) for 8GB GPU memory efficiency
  - Configurable backbones (EfficientNet-B5, B2, ConvNeXt)
  - Gradual unfreezing (freeze early stages -> unfreeze end-to-end after warmup)
  - Label smoothing regularizer against overconfident boundary memorization
  - Multi-threaded DataLoader optimization for 64GB RAM workstation
  - Gate 2 Overfitting / Generalization Audit
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

from ml.deepfake.dataset import FaceForensicsDataset
from ml.deepfake.models.efficientnet_deepfake import build_model

def find_optimal_threshold(targets, probs):
    """Finds decision threshold maximizing balanced accuracy (equal real & fake weight)."""
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

def evaluate_split(model, dataloader, criterion, device, threshold: float = 0.50, use_amp: bool = True):
    """Evaluates loss, accuracy, and predictions over a dataset split."""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    all_subtypes = []

    with torch.no_grad():
        for tensors, targets, subtypes in dataloader:
            tensors = tensors.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(use_amp and device.type == 'cuda')):
                logits = model(tensors)
                loss = criterion(logits, targets)

            total_loss += loss.item() * len(targets)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy().flatten())
            all_subtypes.extend(subtypes)

    avg_loss = total_loss / max(1, len(dataloader.dataset))
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    binary_preds = (all_preds >= threshold).astype(int)

    acc = accuracy_score(all_targets, binary_preds)
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except Exception:
        auc = 0.5

    f1 = f1_score(all_targets, binary_preds, zero_division=0)
    prec = precision_score(all_targets, binary_preds, zero_division=0)
    rec = recall_score(all_targets, binary_preds, zero_division=0)

    # Sub-category breakdown
    sub_metrics = {}
    for st in set(all_subtypes):
        idx = [i for i, s in enumerate(all_subtypes) if s == st]
        st_targets = all_targets[idx]
        st_preds = binary_preds[idx]
        st_probs = all_preds[idx]
        st_acc = accuracy_score(st_targets, st_preds) if len(idx) > 0 else 0.0
        sub_metrics[st] = {
            "count": len(idx),
            "accuracy": round(float(st_acc * 100), 2),
            "mean_confidence": round(float(np.mean(st_probs)), 4) if len(st_probs) > 0 else 0.5
        }

    return {
        "loss": round(float(avg_loss), 4),
        "accuracy": round(float(acc * 100), 2),
        "roc_auc": round(float(auc), 4),
        "f1": round(float(f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "threshold": threshold,
        "sub_metrics": sub_metrics,
        "preds": all_preds,
        "targets": all_targets
    }

def train_model(
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 3e-4,
    backbone: str = "efficientnet_b5",
    img_size: int = 288,
    use_amp: bool = True,
    num_workers: int = 8,
    unfreeze_epoch: int = 4,
    label_smoothing: float = 0.05,
    device_name: str = None
):
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))
        use_amp = False

    data_dir = ROOT_DIR / "data" / "deepfake"
    train_csv = data_dir / "manifest_train.csv"
    val_csv = data_dir / "manifest_val.csv"
    test_csv = data_dir / "manifest_test.csv"

    if not train_csv.exists() or not val_csv.exists():
        print(f"[!] Manifests not found in {data_dir}. Generating now...")
        from ml.deepfake.prepare_dataset import main as prep_main
        prep_main()

    models_dir = ROOT_DIR / "ml" / "deepfake" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / f"{backbone}_deepfake_best.pt"
    default_link_path = models_dir / "efficientnet_deepfake_best.pt"
    history_path = models_dir / f"{backbone}_training_history.json"

    print(f"[*] Initializing Phase 1 Training on Device: {device}")
    print(f"    Backbone:         {backbone} (Resolution: {img_size}x{img_size})")
    print(f"    Batch Size:       {batch_size}")
    print(f"    Mixed Precision:  {use_amp} (torch.amp.autocast)")
    print(f"    DataLoader Wkrs:  {num_workers}")
    print(f"    Unfreeze Epoch:   {unfreeze_epoch}")
    print(f"    Train Split:      {train_csv}")

    train_ds = FaceForensicsDataset(train_csv, is_training=True, img_size=img_size)
    val_ds = FaceForensicsDataset(val_csv, is_training=False, img_size=img_size)
    test_ds = FaceForensicsDataset(test_csv, is_training=False, img_size=img_size)

    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_mem, persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem, persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem, persistent_workers=(num_workers > 0)
    )

    print(f"[*] Total Samples: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # Initialize model: freeze early stages initially for stable warm-up
    model = build_model(model_name=backbone, pretrained=True, freeze_early=True, freeze_up_to_stage=5).to(device)

    # Loss function with optional label smoothing
    criterion = nn.BCEWithLogitsLoss()
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
        # Gradual unfreezing: after warm-up epoch, unfreeze all layers for fine-tuning
        if epoch == unfreeze_epoch:
            print(f"\n[*] [Epoch {epoch}] Unfreezing entire {backbone} backbone for full end-to-end tuning...")
            model.unfreeze_all()
            # Backbone gets 10x smaller learning rate to prevent catastrophic forgetting
            backbone_params = []
            head_params = []
            for name, param in model.named_parameters():
                if "classifier" in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)
            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": lr * 0.10},
                {"params": head_params, "lr": lr}
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=(epochs - epoch + 1), eta_min=1e-6
            )

        model.train()
        train_loss_total = 0.0

        for tensors, targets, _ in train_loader:
            tensors = tensors.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Apply label smoothing if configured: y_smooth = y*(1-eps) + 0.5*eps
            if label_smoothing > 0.0:
                smooth_targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
            else:
                smooth_targets = targets

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(use_amp and device.type == 'cuda')):
                logits = model(tensors)
                loss = criterion(logits, smooth_targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += loss.item() * len(targets)

        scheduler.step()
        train_loss = train_loss_total / max(1, len(train_ds))

        # Validation evaluation
        val_res = evaluate_split(model, val_loader, criterion, device, threshold=0.50, use_amp=use_amp)
        val_loss = val_res["loss"]
        val_acc = val_res["accuracy"]
        val_auc = val_res["roc_auc"]
        current_lr = scheduler.get_last_lr()[-1]

        # Compute optimal decision threshold on validation set
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

        # Checkpoint based on composite discrimination metric
        if composite_score > best_score:
            best_score = composite_score
            best_val_auc = val_auc
            best_thresh = opt_th
            ckpt_payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "optimal_threshold": opt_th,
                "architecture": backbone,
                "img_size": img_size
            }
            torch.save(ckpt_payload, best_model_path)
            torch.save(ckpt_payload, default_link_path)

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"[OK] Training complete in {elapsed:.1f}s ({elapsed/60:.1f} mins).")
    print(f"[OK] Best Checkpoint: Val AUC = {best_val_auc:.4f} | Calibrated Threshold = {best_thresh:.3f}")
    print(f"     Saved to: {best_model_path}")

    # Load best checkpoint for Gate 2 final audit
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])

    # Final Evaluation at Calibrated Threshold
    test_res_calib = evaluate_split(model, test_loader, criterion, device, threshold=best_thresh, use_amp=use_amp)
    val_final_res = evaluate_split(model, val_loader, criterion, device, threshold=best_thresh, use_amp=use_amp)
    train_final_res = evaluate_split(model, train_loader, criterion, device, threshold=best_thresh, use_amp=use_amp)

    # Save history
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Execute Gate 2 Audit
    execute_gate_2_audit(train_final_res, val_final_res, test_res_calib, data_dir, calibrated_threshold=best_thresh)

def execute_gate_2_audit(train_res, val_res, test_res, output_dir: Path, calibrated_threshold: float = 0.50):
    """
    Executes Gate 2 Overfitting, Underfitting & Generalization Audit.
    """
    print("\n" + "=" * 70)
    print(" [GATE 2] OVERFITTING / UNDERFITTING & ACCURACY AUDIT")
    print("=" * 70)

    loss_gap = abs(val_res["loss"] - train_res["loss"])
    acc_gap = abs(train_res["accuracy"] - val_res["accuracy"])

    overfit_pass = acc_gap <= 7.0 and loss_gap <= 0.25
    underfit_pass = val_res["accuracy"] >= 80.0 and val_res["roc_auc"] >= 0.88
    test_pass = test_res["accuracy"] >= 80.0 and test_res["roc_auc"] >= 0.85

    gate_2_passed = overfit_pass and underfit_pass and test_pass

    print(f" [1] Convergence & Generalization Gap Analysis:")
    print(f"     - Final Train Loss: {train_res['loss']:.4f} | Final Val Loss: {val_res['loss']:.4f}")
    print(f"     - Loss Divergence Gap: {loss_gap:.4f} (Tolerance: <= 0.25)")
    print(f"     - Accuracy Divergence Gap: {acc_gap:.2f}% (Tolerance: <= 7.0%) -> {'PASS' if overfit_pass else 'NOTICE'}")
    print(f"     - Train Acc: {train_res['accuracy']:.2f}% | Val Acc: {val_res['accuracy']:.2f}%")

    print(f"\n [2] Discrimination Metrics on Holdout Test Split (N={len(test_res['targets'])}):")
    print(f"     - Decision Threshold: {calibrated_threshold:.3f}")
    print(f"     - Test Accuracy:  {test_res['accuracy']:.2f}% (Target: >= 80.0%) -> {'PASS' if test_pass else 'NOTICE'}")
    print(f"     - Test ROC-AUC:   {test_res['roc_auc']:.4f}")
    print(f"     - Test F1-Score:  {test_res['f1']:.4f}")
    print(f"     - Test Precision: {test_res['precision']:.4f}")
    print(f"     - Test Recall:    {test_res['recall']:.4f}")

    print(f"\n [3] Sub-Category Breakdown (Holdout Test Split):")
    for st, m in sorted(test_res["sub_metrics"].items()):
        print(f"     - {st:<14}: Acc = {m['accuracy']:>6.2f}% | Mean Conf = {m['mean_confidence']:>6.4f} (N={m['count']})")

    print("=" * 70)
    if gate_2_passed:
        print(" [RESULT] GATE 2 PASSED: HIGH ACCURACY & GENERALIZATION VERIFIED")
    else:
        print(" [RESULT] GATE 2 COMPLETED: Check metrics above against target tolerances.")
    print("=" * 70 + "\n")

    audit_report = {
        "gate_2_passed": gate_2_passed,
        "generalization_gap": {
            "loss_gap": round(loss_gap, 4),
            "acc_gap": round(acc_gap, 2),
            "overfit_passed": overfit_pass
        },
        "train_metrics": {k: v for k, v in train_res.items() if k not in ["preds", "targets"]},
        "val_metrics": {k: v for k, v in val_res.items() if k not in ["preds", "targets"]},
        "test_metrics": {k: v for k, v in test_res.items() if k not in ["preds", "targets"]}
    }

    audit_path = output_dir / "gate2_audit_report.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"[OK] Gate 2 Audit Report saved to: {audit_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train deepfake detector with AMP & gradual unfreezing.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--backbone", type=str, default="efficientnet_b5", help="Model backbone name (efficientnet_b5, efficientnet_b2)")
    parser.add_argument("--img_size", type=int, default=288, help="Image resolution for backbone")
    parser.add_argument("--amp", action="store_true", default=True, help="Use Automatic Mixed Precision")
    parser.add_argument("--no_amp", action="store_false", dest="amp", help="Disable Automatic Mixed Precision")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader worker count")
    parser.add_argument("--unfreeze_epoch", type=int, default=4, help="Epoch to unfreeze entire backbone")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing epsilon")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone=args.backbone,
        img_size=args.img_size,
        use_amp=args.amp,
        num_workers=args.num_workers,
        unfreeze_epoch=args.unfreeze_epoch,
        label_smoothing=args.label_smoothing
    )

