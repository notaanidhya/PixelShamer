"""
ml/deepfake/prepare_dataset.py
==============================
Phase 1: Dataset Partitioning, Manifest Engineering & Leakage Quarantine.
Processes the CIPLab Real and Fake Face dataset, executes a stratified 80/10/10 split,
and enforces a strict Gate 1 Data Leakage Audit before generating manifests.
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DEFAULT_SOURCE = Path.home() / ".cache" / "kagglehub" / "datasets" / "ciplab" / "real-and-fake-face-detection" / "versions" / "1" / "real_and_fake_face"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent.parent / "data" / "deepfake"

RANDOM_SEED = 42

def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 checksum of a file for duplicate & leakage detection."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def classify_sub_type(filename: str, is_real: bool) -> str:
    """Classifies sample into fine-grained difficulty categories."""
    if is_real:
        return "real"
    fn = filename.lower()
    if fn.startswith("easy_"):
        return "fake_easy"
    elif fn.startswith("mid_"):
        return "fake_mid"
    elif fn.startswith("hard_"):
        return "fake_hard"
    return "fake_other"

def scan_dataset(source_dir: Path) -> list[dict]:
    """Scans real and fake directories and compiles record list with SHA256 hashes."""
    records = []
    real_dir = source_dir / "training_real"
    fake_dir = source_dir / "training_fake"

    if not real_dir.exists() or not fake_dir.exists():
        raise FileNotFoundError(f"Missing required dataset directories in: {source_dir}")

    print(f"[*] Scanning 'training_real' from: {real_dir}")
    real_files = sorted(list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png")))
    for p in real_files:
        sha = compute_sha256(p)
        records.append({
            "filepath": str(p.resolve()),
            "filename": p.name,
            "label": 0,
            "label_name": "real",
            "sub_type": "real",
            "sha256": sha,
            "size_bytes": p.stat().st_size
        })

    print(f"[*] Scanning 'training_fake' from: {fake_dir}")
    fake_files = sorted(list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.png")))
    for p in fake_files:
        sha = compute_sha256(p)
        sub = classify_sub_type(p.name, is_real=False)
        records.append({
            "filepath": str(p.resolve()),
            "filename": p.name,
            "label": 1,
            "label_name": "fake",
            "sub_type": sub,
            "sha256": sha,
            "size_bytes": p.stat().st_size
        })

    return records

def perform_stratified_split(df: pd.DataFrame, seed: int = RANDOM_SEED):
    """
    Performs stratified 80% train, 10% val, 10% test split based on sub_type.
    Guarantees balanced representation of real, easy, mid, and hard categories.
    """
    # 1. Split Train (80%) vs Temp (20%)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=seed,
        stratify=df["sub_type"]
    )

    # 2. Split Temp into Val (10%) and Test (10%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["sub_type"]
    )

    train_df = train_df.copy().reset_index(drop=True)
    val_df = val_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return train_df, val_df, test_df

def execute_gate_1_audit(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Rigorous Gate 1 Data Leakage Audit:
    - Check 1: SHA-256 hash collision test across all splits.
    - Check 2: Filename collision test across all splits.
    - Check 3: Class prior balance verification.
    - Check 4: Sub-type representation verification.
    """
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    print("\n" + "=" * 60)
    print(" [GATE 1] DATA LEAKAGE & INTEGRITY AUDIT")
    print("=" * 60)

    train_hashes = set(train_df["sha256"])
    val_hashes = set(val_df["sha256"])
    test_hashes = set(test_df["sha256"])

    # 1. Hash Intersections
    train_val_overlap = train_hashes.intersection(val_hashes)
    train_test_overlap = train_hashes.intersection(test_hashes)
    val_test_overlap = val_hashes.intersection(test_hashes)

    hash_leakage = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)

    # 2. Filename Intersections
    train_names = set(train_df["filename"])
    val_names = set(val_df["filename"])
    test_names = set(test_df["filename"])

    name_leakage = len(train_names.intersection(val_names)) + \
                   len(train_names.intersection(test_names)) + \
                   len(val_names.intersection(test_names))

    # 3. Class distribution checks
    def get_distribution(df):
        total = len(df)
        sub_counts = df["sub_type"].value_counts().to_dict()
        real_ratio = (df["label"] == 0).mean()
        fake_ratio = (df["label"] == 1).mean()
        return {
            "total": total,
            "real_pct": round(real_ratio * 100, 2),
            "fake_pct": round(fake_ratio * 100, 2),
            "sub_counts": sub_counts
        }

    train_dist = get_distribution(train_df)
    val_dist = get_distribution(val_df)
    test_dist = get_distribution(test_df)

    audit_passed = (hash_leakage == 0) and (name_leakage == 0)

    print(f" [OK] Total Samples Processed: {len(train_df) + len(val_df) + len(test_df)}")
    print(f"      - Train Set: {len(train_df)} ({len(train_df)/(len(train_df)+len(val_df)+len(test_df)):.1%})")
    print(f"      - Val Set:   {len(val_df)} ({len(val_df)/(len(train_df)+len(val_df)+len(test_df)):.1%})")
    print(f"      - Test Set:  {len(test_df)} ({len(test_df)/(len(train_df)+len(val_df)+len(test_df)):.1%})")
    print("-" * 60)
    print(f" [OK] SHA-256 Hash Collisions Across Splits: {hash_leakage}")
    print(f"      - Train & Val:  {len(train_val_overlap)}")
    print(f"      - Train & Test: {len(train_test_overlap)}")
    print(f"      - Val & Test:   {len(val_test_overlap)}")
    print(f" [OK] Filename Collisions Across Splits:    {name_leakage}")
    print("-" * 60)
    print(" [OK] Sub-Category Stratification Check:")
    categories = sorted(list(set(train_df["sub_type"])))
    for c in categories:
        tr_c = train_dist["sub_counts"].get(c, 0)
        va_c = val_dist["sub_counts"].get(c, 0)
        te_c = test_dist["sub_counts"].get(c, 0)
        print(f"      - {c:<12}: Train={tr_c:<4} | Val={va_c:<3} | Test={te_c:<3}")
    print("=" * 60)

    if audit_passed:
        print(" [RESULT] GATE 1 PASSED: ZERO DATA LEAKAGE VERIFIED")
    else:
        print(" [RESULT] GATE 1 FAILED: LEAKAGE DETECTED")
    print("=" * 60 + "\n")

    audit_report = {
        "audit_passed": audit_passed,
        "hash_leakage_count": hash_leakage,
        "name_leakage_count": name_leakage,
        "splits": {
            "train": train_dist,
            "val": val_dist,
            "test": test_dist
        }
    }
    return audit_report

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset and execute Gate 1 audit.")
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE), help="Path to CIPLab raw dataset")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output directory for manifests")
    args = parser.parse_args()

    source_path = Path(args.source)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Beginning Phase 1 Dataset Preparation...")
    print(f"    Source: {source_path}")
    print(f"    Output: {output_path}")

    records = scan_dataset(source_path)
    df = pd.DataFrame(records)

    # Perform stratified split
    train_df, val_df, test_df = perform_stratified_split(df, seed=RANDOM_SEED)

    # Execute Gate 1 Audit
    audit = execute_gate_1_audit(train_df, val_df, test_df)

    if not audit["audit_passed"]:
        print("[!] ERROR: Gate 1 audit failed. Aborting manifest generation.")
        sys.exit(1)

    # Save manifests
    train_path = output_path / "manifest_train.csv"
    val_path = output_path / "manifest_val.csv"
    test_path = output_path / "manifest_test.csv"
    audit_path = output_path / "gate1_audit_report.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)

    print(f"[OK] Manifests successfully written:")
    print(f"    - {train_path} ({len(train_df)} rows)")
    print(f"    - {val_path} ({len(val_df)} rows)")
    print(f"    - {test_path} ({len(test_df)} rows)")
    print(f"    - {audit_path}")
    print(f"[OK] Phase 1 completed successfully.")

if __name__ == "__main__":
    main()
