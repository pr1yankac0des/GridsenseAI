# GridSense AI — Technical Documentation

Complete technical reference for the GridSense AI power quality monitoring system.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Feature Engineering](#feature-engineering)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [Backend API](#backend-api)
6. [Dashboard Components](#dashboard-components)
7. [Cost Calculation Methodology](#cost-calculation-methodology)
8. [Glossary](#glossary)

---

## Problem Statement

### What Are Power Quality Disturbances?

Electricity should flow as a perfect sine wave at the nominal voltage and frequency (230V at 50Hz in India, 120V at 60Hz in the US). In reality, this wave gets distorted, drops, spikes, or fluctuates. These deviations are called **Power Quality Disturbances**.

### The Five Common Disturbances We Detect

| Type | Description | Real-World Cause |
|------|-------------|------------------|
| **Normal** | Clean sine wave at nominal voltage | Balanced load, no abnormalities |
| **Voltage Sag** | RMS drops below 90% of nominal | Motor startup, short circuits, sudden load surge |
| **Voltage Swell** | RMS rises above 110% of nominal | Sudden load disconnection, capacitor switching |
| **Harmonics** | Waveform distorted by extra frequencies | VFDs, rectifiers, UPS systems, LED drivers |
| **Transients** | Brief, high-energy voltage spikes | Lightning, switching surges, ESD |

### Why It Matters

Power quality issues cause **75% of all industrial electrical failures** and cost the global industry **$15-24 billion per year**. The frustrating part: traditional monitoring systems show operators cryptic numbers (THD, RMS, PF) that they cannot interpret or act upon. Problems go unnoticed until equipment fails — sometimes catastrophically.

---

## Dataset

### Source
**Kaggle Power Quality Fault Detection Dataset**

### Statistics
- **Total Samples:** 2,367
- **Classes:** 5 (balanced)
- **Missing Values:** 0
- **Features:** 11 columns (9 used + ID + Label)

### Class Distribution
| Class | Samples | Percentage |
|-------|---------|-----------|
| Normal | 492 | 20.8% |
| Harmonics | 485 | 20.5% |
| Transient | 473 | 20.0% |
| Sag | 467 | 19.7% |
| Swell | 450 | 19.0% |

### Original Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| RMS_Voltage | float | 114 - 301 V | Effective voltage value |
| Peak_Voltage | float | 161 - 466 V | Maximum instantaneous voltage |
| THD | float | -0.25 - 18.94 % | Total Harmonic Distortion |
| Duration_ms | int | 10 - 299 ms | Disturbance duration |
| DWT_Energy_Level1 | float | 10 - 99 | Wavelet detail level 1 energy |
| DWT_Energy_Level2 | float | 5 - 49 | Wavelet detail level 2 energy |
| DWT_Entropy | float | 1 - 5 | Signal complexity measure |
| Signal_Noise_Ratio_dB | float | 20 - 40 | Signal quality |
| Phase | string | A, B, C | 3-phase identifier |

---

## Feature Engineering

We engineered **3 additional features** based on electrical engineering physics. These features alone contribute **35.14% of model decision power**.

### Feature 1: Crest Factor

```python
df['Crest_Factor'] = df['Peak_Voltage'] / df['RMS_Voltage']
```

**Physics:** For a perfect sine wave, the mathematical relationship is:
$$\text{Peak} = \text{RMS} \times \sqrt{2}$$

Therefore: **Crest Factor = √2 ≈ 1.414** for any clean wave.

**Why it works:**
- Normal: 327V / 230V = 1.421 ✓
- Transient: 466V / 230V = 2.026 ⚠️ (peak disproportionate)

### Feature 2: Voltage Deviation

```python
df['Voltage_Deviation'] = abs(df['RMS_Voltage'] - 230)
```

**Logic:** Distance from 230V nominal standard. Both sags and swells produce high deviation values, allowing the model to detect bidirectional anomalies with a single feature.

- Normal (232V) → 2 (low) ✓
- Sag (147V) → 83 (high) ⚠️
- Swell (263V) → 33 (high) ⚠️

### Feature 3: Peak Excess

```python
df['Peak_Excess'] = df['Peak_Voltage'] - (df['RMS_Voltage'] * np.sqrt(2))
```

**Logic:** The residual between actual peak and theoretically expected peak. For clean waves, this is approximately zero. Positive values indicate distortion.

- Normal: 327 - (230 × 1.414) = +1.7 ✓ (near zero)
- Transient: 406 - (224 × 1.414) = +88.98 ⚠️ (large excess)

---

## Machine Learning Pipeline

### Step 1: Data Loading
```python
df = pd.read_csv("data/power_quality_fault_dataset.csv")
```

### Step 2: Feature Engineering
- Encode Phase categorically (A=0, B=1, C=2)
- Calculate Crest Factor, Voltage Deviation, Peak Excess

### Step 3: Normalization
**StandardScaler (Z-Score Normalization)**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Formula: `z = (x - μ) / σ` where μ is mean and σ is standard deviation.

### Step 4: Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
```
- Training: 1,893 samples (80%)
- Testing: 474 samples (20%)
- Stratified split ensures equal class proportions

### Step 5: Model Training
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=300,        # 300 decision trees
    max_depth=25,             # max tree depth
    min_samples_split=3,      # min samples to split
    min_samples_leaf=1,       # min samples in leaf
    max_features='sqrt',      # features per tree
    class_weight='balanced',  # handle imbalance
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
```

### Step 6: Validation
- **Test Accuracy:** 99.16%
- **10-Fold Cross-Validation:** 99.41% ± 0.51%
- **F1 Score:** 99.15%

### Step 7: Model Persistence
```python
import pickle
with open("ml/gridsense_model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "scaler": scaler,
        "features": FEATURES,
        "label_map": LABEL_MAP,
        "accuracy": accuracy
    }, f)
```

### Why Random Forest?

| Reason | Detail |
|--------|--------|
| Speed | Trains in 2 seconds (no GPU needed) |
| Accuracy | 99.16% on real tabular data |
| Interpretability | Feature importance is exposed |
| Robustness | Ensemble of 300 trees |
| Right size | Optimal for 2,367 samples |

---

## Backend API

### Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Server status check |
| POST | `/detect` | Detect fault from sensor readings |
| POST | `/simulate` | Generate random fault for demo |
| GET | `/fault-types` | List all fault types with explanations |

### Example Request
```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "RMS_Voltage": 150,
    "Peak_Voltage": 212,
    "THD": 3.0,
    "Duration_ms": 200,
    "DWT_Energy_Level1": 50,
    "DWT_Energy_Level2": 25,
    "DWT_Entropy": 2.5,
    "Signal_Noise_Ratio_dB": 30,
    "Phase": "A"
  }'
```

### Example Response
```json
{
  "fault": "Voltage Sag",
  "prediction": 1,
  "confidence": 99.2,
  "cause": "Large motor startup, short circuit on adjacent feeder...",
  "impact": "Equipment malfunction, PLC dropout, production stoppage...",
  "action": "1) Delay heavy load startup 5-10s\n2) Check capacitor banks...",
  "risk_level": "HIGH",
  "risk_score": 72,
  "energy_loss_pct": 8.5,
  "cost_per_hour_inr": 1020
}
```

---

## Dashboard Components

### Header
- Project logo and title
- Real-time fault status badge with pulsing indicator
- Model accuracy and dataset size display

### Simulate Buttons
Five buttons to manually trigger each fault type for demonstration.

### Metrics Row (6 Cards)
1. **RMS Voltage** — color-coded (green/amber/red)
2. **THD** — with IEEE-519 limit warning
3. **Power Factor** — efficiency indicator
4. **Frequency** — grid stability
5. **Peak Voltage** — instantaneous maximum
6. **Crest Factor** — distortion indicator

### Chart Panel (3 Tabs)
- **Waveform** — Time-domain voltage signal (Canvas-rendered)
- **FFT Spectrum** — Frequency-domain harmonic analysis (H1-H13)
- **24h Trends** — Historical THD and RMS tracking

### Operator Action Engine
Three-column display showing:
- **Detected Issue** — Fault name with severity badge
- **Root Cause** — Most likely physical cause
- **Recommended Action** — Numbered steps for the operator

### AI Chatbot
- Text input for natural language queries
- Web Speech API voice input
- Rule-based response engine with context-aware replies

### Live Alerts Panel
Real-time fault notifications with timestamps and severity.

### Bottom Cards
- **Energy Loss** — Percentage and ₹/hour cost
- **Risk Score** — 0-100 visual gauge
- **Predictive Insights** — Maintenance recommendations

---

## Cost Calculation Methodology

### Formula
```javascript
const energyLoss = (THD * 0.5) + ((1 - PF) * 15);
const costPerHour = energyLoss * 120;
```

### Coefficient Explanations

**THD × 0.5:**  
Harmonic-induced losses (I²R heating in cables and transformers). For every 1% of THD, approximately 0.5% additional energy is lost as heat.

**(1 - PF) × 15:**  
Power factor penalty modeling. Indian utilities apply roughly 15% extra cost for every 0.1 drop below acceptable PF thresholds.

**× 120:**  
Cost conversion factor calibrated to a typical 15kW industrial sub-load at ₹8/kWh. This produces conservative ₹/hour estimates suitable for operator awareness.

### Example Calculation (Harmonics scenario)
- THD = 17%, PF = 0.77
- THD losses: 17 × 0.5 = 8.5%
- PF losses: (1 - 0.77) × 15 = 3.45%
- Total loss: 11.95%
- Cost: 11.95 × 120 ≈ **₹1,434/hour wasted**

---

## Glossary

| Term | Definition |
|------|-----------|
| **RMS** | Root Mean Square — the "effective" value of an AC signal |
| **THD** | Total Harmonic Distortion — measures wave distortion |
| **FFT** | Fast Fourier Transform — converts time signal to frequency components |
| **DWT** | Discrete Wavelet Transform — captures both time and frequency info |
| **PF** | Power Factor — ratio of real to apparent power (1.0 = perfect) |
| **Crest Factor** | Peak/RMS ratio (ideal = √2 ≈ 1.414) |
| **VFD** | Variable Frequency Drive — common harmonic source |
| **SPD** | Surge Protection Device — protects against transients |
| **DVR** | Dynamic Voltage Restorer — corrects voltage sags |
| **IEEE-519** | International standard limiting harmonic distortion to 5% |
| **PLC** | Programmable Logic Controller — industrial automation device |

---

## References

- IEEE-519 Standard: IEEE Recommended Practice for Harmonic Control
- IEEE-1159 Standard: IEEE Recommended Practice on Power Quality Monitoring
- Kaggle Dataset: programmer3/power-quality-fault-detection-dataset
- Scikit-learn Documentation: https://scikit-learn.org/
- FastAPI Documentation: https://fastapi.tiangolo.com/

---

*GridSense AI — Industry 5.0 Power Quality Intelligence*
