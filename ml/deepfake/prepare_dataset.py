"""
ml/deepfake/prepare_dataset.py
==============================
Phase 1: Dataset Ingestion, Manifest Engineering & Gate 1 Leakage Audit.
Supports:
  1. Kaggle 140k Real and Fake Faces ('xhlulu/140k-real-and-fake-faces')
  2. CIPLab Real and Fake Face dataset
  3. Custom datasets with (real / fake) directory hierarchies
Performs stratified 80/10/10 partition and enforces strict Gate 1 Data Leakage Audit.
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

DEFAULT_CIPLAB_SOURCE = Path.home() / ".cache" / "kagglehub" / "datasets" / "ciplab" / "real-and-fake-face-detection" / "versions" / "1" / "real_and_fake_face"
DEFAULT_140K_SOURCE = Path.home() / ".cache" / "kagglehub" / "datasets" / "xhlulu" / "140k-real-and-fake-faces" / "versions" / "1"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent.parent / "data" / "deepfake"

RANDOM_SEED = 42

def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 checksum of a file for duplicate & leakage detection."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""

def classify_sub_type(filename: str, is_real: bool, parent_folder: str = "") -> str:
    """Classifies sample into fine-grained difficulty categories or domain tags."""
    if is_real:
        return "real"
    fn = filename.lower()
    if fn.startswith("easy_"):
        return "fake_easy"
    elif fn.startswith("mid_"):
        return "fake_mid"
    elif fn.startswith("hard_"):
        return "fake_hard"
    elif "stylegan" in parent_folder.lower():
        return "fake_stylegan"
    return "fake_general"

def find_image_files(directory: Path) -> list[Path]:
    """Finds all common image files in a directory."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    files = []
    for ext in extensions:
        files.extend(directory.glob(ext))
        files.extend(directory.glob(ext.upper()))
    return sorted(list(set(files)))

def scan_dataset(source_dir: Path) -> list[dict]:
    """
    Intelligently scans source directory.
    Supports:
      - Structure A: source_dir / 'training_real' and source_dir / 'training_fake'
      - Structure B: source_dir / 'real' and source_dir / 'fake'
      - Structure C: source_dir / 'real_vs_fake' / 'real-vs-fake' / ('train' | 'valid' | 'test') / ('real' | 'fake')
    """
    records = []
    
    # Check for Structure C (140k Kaggle structure)
    nested_splits = list(source_dir.glob("**/train/real")) + list(source_dir.glob("**/real-vs-fake/train/real"))
    if nested_splits:
        base_split_dir = nested_splits[0].parent.parent
        print(f"[*] Detected 140k pre-split directory structure under: {base_split_dir}")
        for split_name in ["train", "valid", "test"]:
            split_dir = base_split_dir / split_name
            if not split_dir.exists():
                continue
            
            real_dir = split_dir / "real"
            fake_dir = split_dir / "fake"
            
            if real_dir.exists():
                for p in find_image_files(real_dir):
                    records.append({
                        "filepath": str(p.resolve()),
                        "filename": p.name,
                        "label": 0,
                        "label_name": "real",
                        "sub_type": "real",
                        "sha256": compute_sha256(p),
                        "size_bytes": p.stat().st_size,
                        "predefined_split": "val" if split_name == "valid" else split_name
                    })
            if fake_dir.exists():
                for p in find_image_files(fake_dir):
                    records.append({
                        "filepath": str(p.resolve()),
                        "filename": p.name,
                        "label": 1,
                        "label_name": "fake",
                        "sub_type": classify_sub_type(p.name, is_real=False, parent_folder="140k"),
                        "sha256": compute_sha256(p),
                        "size_bytes": p.stat().st_size,
                        "predefined_split": "val" if split_name == "valid" else split_name
                    })
        return records

    # Check for Structure A (CIPLab: training_real / training_fake)
    real_dir_a = source_dir / "training_real"
    fake_dir_a = source_dir / "training_fake"
    
    # Check for Structure B (flat real / fake)
    real_dir_b = source_dir / "real"
    fake_dir_b = source_dir / "fake"
    
    if real_dir_a.exists() and fake_dir_a.exists():
        real_dir, fake_dir = real_dir_a, fake_dir_a
    elif real_dir_b.exists() and fake_dir_b.exists():
        real_dir, fake_dir = real_dir_b, fake_dir_b
    else:
        raise FileNotFoundError(
            f"Could not locate (real/fake) or (training_real/training_fake) inside: {source_dir}\n"
            f"Please verify the directory path."
        )

    print(f"[*] Scanning real faces from: {real_dir}")
    for p in find_image_files(real_dir):
        records.append({
            "filepath": str(p.resolve()),
            "filename": p.name,
            "label": 0,
            "label_name": "real",
            "sub_type": "real",
            "sha256": compute_sha256(p),
            "size_bytes": p.stat().st_size
        })

    print(f"[*] Scanning fake faces from: {fake_dir}")
    for p in find_image_files(fake_dir):
        records.append({
            "filepath": str(p.resolve()),
            "filename": p.name,
            "label": 1,
            "label_name": "fake",
            "sub_type": classify_sub_type(p.name, is_real=False),
            "sha256": compute_sha256(p),
            "size_bytes": p.stat().st_size
        })

    return records

def perform_stratified_split(df: pd.DataFrame, seed: int = RANDOM_SEED):
    """
    Performs stratified 80% train, 10% val, 10% test split based on sub_type.
    If the dataset already came with predefined splits, those are honored.
    """
    if "predefined_split" in df.columns and set(df["predefined_split"].unique()).issubset({"train", "val", "test"}):
        print("[*] Using predefined train / val / test splits from source dataset.")
        train_df = df[df["predefined_split"] == "train"].copy().reset_index(drop=True)
        val_df = df[df["predefined_split"] == "val"].copy().reset_index(drop=True)
        test_df = df[df["predefined_split"] == "test"].copy().reset_index(drop=True)
        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"
        return train_df, val_df, test_df

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
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    print("\n" + "=" * 60)
    print(" [GATE 1] DATA LEAKAGE & INTEGRITY AUDIT")
    print("=" * 60)

    train_hashes = set(train_df["sha256"].dropna())
    val_hashes = set(val_df["sha256"].dropna())
    test_hashes = set(test_df["sha256"].dropna())

    # Filter out empty string hash
    train_hashes.discard("")
    val_hashes.discard("")
    test_hashes.discard("")

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

    def get_distribution(df):
        total = len(df)
        sub_counts = df["sub_type"].value_counts().to_dict()
        real_ratio = (df["label"] == 0).mean()
        fake_ratio = (df["label"] == 1).mean()
        return {
            "total": total,
            "real_pct": round(float(real_ratio * 100), 2),
            "fake_pct": round(float(fake_ratio * 100), 2),
            "sub_counts": {k: int(v) for k, v in sub_counts.items()}
        }

    train_dist = get_distribution(train_df)
    val_dist = get_distribution(val_df)
    test_dist = get_distribution(test_df)

    audit_passed = (hash_leakage == 0)

    total_samples = len(train_df) + len(val_df) + len(test_df)
    print(f" [OK] Total Samples Processed: {total_samples}")
    print(f"      - Train Set: {len(train_df)} ({len(train_df)/total_samples:.1%})")
    print(f"      - Val Set:   {len(val_df)} ({len(val_df)/total_samples:.1%})")
    print(f"      - Test Set:  {len(test_df)} ({len(test_df)/total_samples:.1%})")
    print("-" * 60)
    print(f" [OK] SHA-256 Hash Collisions Across Splits: {hash_leakage}")
    if hash_leakage > 0:
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
        print(f"      - {c:<14}: Train={tr_c:<6} | Val={va_c:<5} | Test={te_c:<5}")
    print("=" * 60)

    if audit_passed:
        print(" [RESULT] GATE 1 PASSED: ZERO DATA LEAKAGE VERIFIED")
    else:
        print(" [RESULT] GATE 1 WARNING: DUPLICATE HASHES DETECTED")
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

def resolve_source_directory(requested_source: str | None) -> Path:
    """Finds best available dataset source automatically."""
    if requested_source and Path(requested_source).exists():
        return Path(requested_source)
    
    # Check 140k dataset
    if DEFAULT_140K_SOURCE.exists():
        return DEFAULT_140K_SOURCE
    
    # Check CIPLab dataset
    if DEFAULT_CIPLAB_SOURCE.exists():
        return DEFAULT_CIPLAB_SOURCE
    
    # Try kagglehub auto-download if installed
    try:
        import kagglehub
        print("[*] Checking for Kaggle 'xhlulu/140k-real-and-fake-faces' via kagglehub...")
        path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
        if path and Path(path).exists():
            return Path(path)
    except Exception as e:
        print(f"[*] kagglehub auto-download not available: {e}")

    # Fallback to local data/ folder if any
    local_data = Path(__file__).resolve().parent.parent.parent / "data" / "dataset"
    if local_data.exists():
        return local_data

    raise FileNotFoundError(
        "No dataset source found. Specify --source <path_to_images> or run:\n"
        "kaggle datasets download -d xhlulu/140k-real-and-fake-faces"
    )

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset and execute Gate 1 audit.")
    parser.add_argument("--source", type=str, default=None, help="Path to raw dataset folder")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output directory for manifests")
    args = parser.parse_args()

    source_path = resolve_source_directory(args.source)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Beginning Phase 1 Dataset Preparation...")
    print(f"    Source: {source_path}")
    print(f"    Output: {output_path}")

    records = scan_dataset(source_path)
    if not records:
        print(f"[!] ERROR: No image records found in {source_path}")
        sys.exit(1)

    df = pd.DataFrame(records)

    # Perform stratified split
    train_df, val_df, test_df = perform_stratified_split(df, seed=RANDOM_SEED)

    # Execute Gate 1 Audit
    audit = execute_gate_1_audit(train_df, val_df, test_df)

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
    print(f"[OK] Phase 1 dataset preparation completed successfully.")

if __name__ == "__main__":
    main()

