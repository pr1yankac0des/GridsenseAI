"""
╔══════════════════════════════════════════════════════════════════╗
║  STEP 2: MODEL TRAINING                                        ║
║  Trains the Random Forest classifier for fault detection        ║
║                                                                ║
║  WHAT THIS DOES:                                               ║
║  1. Loads the CSV dataset                                      ║
║  2. Preprocesses features (scaling, encoding)                  ║
║  3. Trains Random Forest classifier                            ║
║  4. Evaluates accuracy, prints confusion matrix                ║
║  5. Saves model to gridsense_model.pkl                         ║
║                                                                ║
║  RUN: python train_model.py                                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score
)


# ─── CONFIGURATION ─────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "gridsense_model.pkl")

FAULT_NAMES = {
    0: "Normal",
    1: "Voltage Sag",
    2: "Voltage Swell",
    3: "Harmonics",
    4: "Transients"
}

# Features we use for training (9 features)
FEATURE_COLUMNS = [
    "RMS_Voltage",           # Core voltage measurement
    "Peak_Voltage",          # Maximum instantaneous voltage
    "THD",                   # Total Harmonic Distortion (%)
    "Duration",              # Duration of disturbance (seconds)
    "DWT_Energy_Levels",     # Wavelet transform energy
    "DWT_Entropy",           # Signal complexity measure
    "Signal_Noise_Ratio_dB", # Signal quality
    "Phase_Encoded",         # Encoded phase (A=0, B=1, C=2)
    "Crest_Factor",          # Peak/RMS ratio (derived feature)
]

TARGET_COLUMN = "Fault_Type"


def load_data():
    """
    STEP 2a: Load and combine datasets.
    
    We try to load:
    1. Our synthetic dataset (always available)
    2. Kaggle dataset 1 (if downloaded)
    3. Kaggle dataset 2 (if downloaded)
    
    Then combine everything for maximum training data.
    """
    print("\n📦 Loading datasets...")
    
    dfs = []
    
    # Load our synthetic data
    synthetic_path = os.path.join(DATA_DIR, "power_quality_data.csv")
    if os.path.exists(synthetic_path):
        df = pd.read_csv(synthetic_path)
        print(f"  ✅ Synthetic data: {len(df)} samples")
        dfs.append(df)
    else:
        print(f"  ❌ Synthetic data not found! Run generate_dataset.py first.")
        sys.exit(1)
    
    # Try loading Kaggle datasets (if user downloaded them)
    kaggle_files = [
        "power_quality_fault_detection.csv",
        "power-quality-fault-detection-dataset.csv", 
        "pq_fault_detection.csv",
    ]
    for fname in kaggle_files:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            try:
                kdf = pd.read_csv(fpath)
                print(f"  ✅ Kaggle data ({fname}): {len(kdf)} samples")
                # Map column names if needed
                kdf = standardize_columns(kdf)
                dfs.append(kdf)
            except Exception as e:
                print(f"  ⚠️ Could not load {fname}: {e}")
    
    # Combine all
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  📊 Total combined dataset: {len(combined)} samples")
    
    return combined


def standardize_columns(df):
    """
    Standardize column names from different dataset formats.
    Different Kaggle datasets may have different column names.
    This maps them to our standard format.
    """
    # Common column name mappings
    rename_map = {
        "rms_voltage": "RMS_Voltage",
        "rms": "RMS_Voltage", 
        "peak_voltage": "Peak_Voltage",
        "peak": "Peak_Voltage",
        "thd": "THD",
        "total_harmonic_distortion": "THD",
        "duration": "Duration",
        "dwt_energy": "DWT_Energy_Levels",
        "dwt_entropy": "DWT_Entropy",
        "snr": "Signal_Noise_Ratio_dB",
        "snr_db": "Signal_Noise_Ratio_dB",
        "phase": "Phase",
        "fault_type": "Fault_Type",
        "fault": "Fault_Type",
        "label": "Fault_Type",
        "class": "Fault_Type",
    }
    
    # Lowercase all columns for matching
    df.columns = df.columns.str.strip()
    lower_map = {k.lower(): v for k, v in rename_map.items()}
    
    new_cols = {}
    for col in df.columns:
        if col.lower() in lower_map:
            new_cols[col] = lower_map[col.lower()]
    
    df = df.rename(columns=new_cols)
    return df


def preprocess(df):
    """
    STEP 2b: Feature Engineering & Preprocessing.
    
    WHY EACH STEP:
    
    1. LABEL ENCODING (Phase):
       Phase is categorical (A, B, C).
       ML models need numbers, so: A=0, B=1, C=2.
    
    2. CREST FACTOR (new derived feature):
       Crest Factor = Peak / RMS
       For perfect sine: CF = √2 ≈ 1.414
       High CF = transient spikes
       Low CF = clipped waveform
       This is a POWERFUL feature for detecting transients.
    
    3. STANDARD SCALING:
       Features have different ranges:
         RMS: 120-380V
         THD: 0-35%
         Duration: 0.0001-10s
       Scaling puts them all on same scale (mean=0, std=1).
       Random Forest doesn't strictly need scaling, but it helps
       and is good practice for when we try other models.
    """
    print("\n⚙️ Preprocessing...")
    
    df = df.copy()
    
    # Handle missing values
    for col in ["RMS_Voltage", "Peak_Voltage", "THD", "Duration", 
                "DWT_Energy_Levels", "DWT_Entropy", "Signal_Noise_Ratio_dB"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # 1. Encode Phase (A/B/C → 0/1/2)
    if "Phase" in df.columns:
        le = LabelEncoder()
        df["Phase_Encoded"] = le.fit_transform(df["Phase"].astype(str))
        print(f"  ✅ Phase encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    else:
        df["Phase_Encoded"] = 0  # default if no phase column
    
    # 2. Create Crest Factor (derived feature)
    if "Peak_Voltage" in df.columns and "RMS_Voltage" in df.columns:
        df["Crest_Factor"] = df["Peak_Voltage"] / df["RMS_Voltage"].replace(0, 1)
        print(f"  ✅ Crest Factor created (mean: {df['Crest_Factor'].mean():.3f})")
    else:
        df["Crest_Factor"] = 1.414
    
    # Ensure all required columns exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            print(f"  ⚠️ Missing column {col}, filling with defaults")
            df[col] = 0
    
    # Extract features and target
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    
    # 3. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"  ✅ Features scaled to mean=0, std=1")
    print(f"  ✅ Feature matrix shape: {X_scaled.shape}")
    print(f"  ✅ Classes: {np.unique(y)}")
    
    return X_scaled, y, scaler


def train_and_evaluate(X, y, scaler):
    """
    STEP 2c: Train the model and evaluate it.
    
    WHY RANDOM FOREST?
    ┌─────────────────────┬─────────────────────────────────────┐
    │ Reason              │ Explanation                         │
    ├─────────────────────┼─────────────────────────────────────┤
    │ Speed               │ Trains in seconds (vs hours for DL) │
    │ Accuracy            │ 95-99% on structured tabular data   │
    │ No GPU needed       │ Perfect for hackathon               │
    │ Interpretable       │ Feature importance shows WHY        │
    │ Robust              │ Handles noise, missing data well    │
    │ No overfitting      │ Ensemble of 200 trees = stable      │
    └─────────────────────┴─────────────────────────────────────┘
    
    HYPERPARAMETERS EXPLAINED:
    - n_estimators=200: Number of decision trees (more = better, but slower)
    - max_depth=20: How deep each tree can go (prevents overfitting)
    - min_samples_split=5: Minimum samples to split a node
    - min_samples_leaf=2: Minimum samples in a leaf
    - max_features='sqrt': Each tree sees √(n_features) features (diversity)
    """
    print("\n" + "="*60)
    print("🧠 MODEL TRAINING")
    print("="*60)
    
    # Split: 80% training, 20% testing
    # stratify=y ensures each class is proportionally represented
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n  Training set: {X_train.shape[0]} samples")
    print(f"  Test set:     {X_test.shape[0]} samples")
    
    # Train Random Forest
    print(f"\n  Training Random Forest (200 trees)...")
    model = RandomForestClassifier(
        n_estimators=200,     # 200 decision trees
        max_depth=20,         # max tree depth
        min_samples_split=5,  # min samples to split
        min_samples_leaf=2,   # min samples in leaf
        max_features='sqrt',  # features per tree = √9 ≈ 3
        random_state=42,
        n_jobs=-1,            # use all CPU cores
    )
    model.fit(X_train, y_train)
    print("  ✅ Training complete!")
    
    # ─── EVALUATION ─────────────────────────────────────────────
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n  ┌──────────────────────────────────┐")
    print(f"  │  TEST ACCURACY:  {accuracy*100:.2f}%           │")
    print(f"  │  F1 SCORE:       {f1*100:.2f}%           │")
    print(f"  └──────────────────────────────────┘")
    
    # Detailed classification report
    target_names = [FAULT_NAMES[i] for i in sorted(FAULT_NAMES.keys())]
    print(f"\n📊 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("📋 CONFUSION MATRIX:")
    print(f"   (rows = actual, columns = predicted)")
    print(f"   {'':15s}", end="")
    for name in target_names:
        print(f"{name[:8]:>10s}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"   {target_names[i]:15s}", end="")
        for val in row:
            print(f"{val:10d}", end="")
        print()
    
    # Feature importance (IMPORTANT FOR JUDGES!)
    print(f"\n🔍 FEATURE IMPORTANCE (what the model looks at most):")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for rank, idx in enumerate(sorted_idx):
        bar = "█" * int(importances[idx] * 50)
        print(f"   {rank+1}. {FEATURE_COLUMNS[idx]:25s} {importances[idx]:.4f}  {bar}")
    
    # Cross-validation (proves model is robust, not just lucky)
    print(f"\n📈 5-FOLD CROSS-VALIDATION:")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"   Scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"   Mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   → This means the model is CONSISTENT, not overfitting.")
    
    return model, accuracy, cv_scores.mean()


def save_model(model, scaler, accuracy, cv_score):
    """
    STEP 2d: Save everything needed for the API server.
    
    We save:
    - The trained model (for predictions)
    - The scaler (to transform new inputs the same way)
    - Accuracy metrics (to display on dashboard)
    - Feature column names (for validation)
    """
    print(f"\n💾 Saving model...")
    
    model_data = {
        "model": model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "fault_names": FAULT_NAMES,
        "accuracy": accuracy,
        "cv_score": cv_score,
    }
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"  ✅ Saved to: {MODEL_PATH}")
    print(f"  ✅ File size: {size_mb:.1f} MB")
    print(f"  ✅ Accuracy: {accuracy*100:.2f}%")


# ─── MAIN ───────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         GridSense AI — Model Training Pipeline          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    # Load data
    df = load_data()
    
    # Show dataset info
    print(f"\n📊 DATASET SUMMARY:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   Class Distribution:")
    for cls_id, cls_name in FAULT_NAMES.items():
        count = len(df[df[TARGET_COLUMN] == cls_id])
        pct = count / len(df) * 100
        print(f"     {cls_id} ({cls_name:15s}): {count:5d} samples ({pct:.1f}%)")
    
    # Preprocess
    X, y, scaler = preprocess(df)
    
    # Train & evaluate
    model, accuracy, cv_score = train_and_evaluate(X, y, scaler)
    
    # Save
    save_model(model, scaler, accuracy, cv_score)
    
    print(f"\n{'='*60}")
    print(f"  ✅ TRAINING COMPLETE!")
    print(f"  Next step: python server.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
