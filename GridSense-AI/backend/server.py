"""
╔══════════════════════════════════════════════════════════════════╗
║  STEP 3: FastAPI SERVER                                         ║
║  The API that connects ML model to dashboard                    ║
║                                                                ║
║  ENDPOINTS:                                                     ║
║  GET  /health          → Check if server is running             ║
║  POST /detect          → Detect fault from sensor readings      ║
║  POST /simulate        → Generate random signal + detect        ║
║  GET  /fault-types     → List all fault types with explanations ║
║                                                                ║
║  RUN: python server.py (starts at http://localhost:8000)        ║
║  DOCS: http://localhost:8000/docs (auto-generated API docs)     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import math
import numpy as np
import pickle
import os
from typing import Optional, Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── LOAD MODEL ─────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "gridsense_model.pkl")

model_data = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    print(f"✅ Model loaded (accuracy: {model_data['accuracy']*100:.1f}%)")
else:
    print("⚠️ No model found. Run train_model.py first!")


# ─── EXPLANATION ENGINE ─────────────────────────────────────────
# This is what makes our system HUMAN-UNDERSTANDABLE
# Instead of "Class 3", operators see plain English explanations

EXPLANATIONS = {
    0: {
        "fault": "Normal",
        "what_happened": "Power supply is operating within normal parameters. The voltage waveform is clean and stable.",
        "cause": "Normal balanced load conditions on the electrical network.",
        "impact": "No negative impact. Equipment is operating safely and efficiently.",
        "risk_level": "LOW",
        "risk_score": 5,
        "action": "No action required. Continue standard monitoring.",
        "energy_loss_pct": 0.5,
        "cost_per_hour_inr": 60,
        "industry_example": "All machines running normally in an industrial manufacturing facility.",
    },
    1: {
        "fault": "Voltage Sag",
        "what_happened": "Voltage has dropped significantly below the normal 230V level. This is like a brief 'brown-out' in the power supply.",
        "cause": "Most likely: a large motor just started up, a short circuit occurred on a nearby power line, or there was a sudden spike in power demand.",
        "impact": "HIGH RISK — Sensitive equipment may malfunction or shut down. PLCs and relays can trip. Production lines may stop. Data on servers could be lost.",
        "risk_level": "HIGH",
        "risk_score": 72,
        "action": "1. Check if heavy machinery just started\n2. Inspect capacitor bank status\n3. Verify transformer tap settings\n4. Delay starting heavy loads by 5-10 seconds\n5. Consider installing a Dynamic Voltage Restorer (DVR)",
        "energy_loss_pct": 8.5,
        "cost_per_hour_inr": 1020,
        "industry_example": "In a semiconductor fab, a single voltage sag can halt production for hours, costing millions.",
    },
    2: {
        "fault": "Voltage Swell",
        "what_happened": "Voltage has risen above the safe limit (above 253V for a 230V system). This is the opposite of a sag — too much voltage.",
        "cause": "A large electrical load was suddenly disconnected (like a big motor turning off), or capacitor banks switched on unexpectedly.",
        "impact": "MEDIUM-HIGH RISK — Can damage electronic equipment, break down wire insulation, cause overheating, and reduce the lifespan of connected devices.",
        "risk_level": "MEDIUM",
        "risk_score": 58,
        "action": "1. Check if a large load was recently disconnected\n2. Verify voltage regulator is working properly\n3. Review capacitor bank switching schedule\n4. Install surge protection devices on sensitive equipment",
        "energy_loss_pct": 5.2,
        "cost_per_hour_inr": 624,
        "industry_example": "Voltage swells can reduce transformer life by 30-50% if not addressed.",
    },
    3: {
        "fault": "Harmonics",
        "what_happened": "The voltage waveform is distorted — instead of a clean sine wave, it contains extra frequency components (3rd, 5th, 7th harmonics). Think of it like audio distortion in a speaker.",
        "cause": "Non-linear electrical loads are injecting distortion: Variable Frequency Drives (VFDs), rectifiers, UPS systems, LED lighting, switch-mode power supplies, or arc furnaces.",
        "impact": "HIGH RISK — Causes transformer and cable overheating (FIRE HAZARD), motor vibration and bearing damage, capacitor bank failure, electricity metering errors, and interference with communication systems.",
        "risk_level": "HIGH",
        "risk_score": 68,
        "action": "1. Measure THD at point of common coupling (should be <5% per IEEE-519)\n2. Install active or passive harmonic filters\n3. Redistribute single-phase non-linear loads evenly across phases\n4. Check neutral conductor sizing (3rd harmonics add up in neutral!)\n5. Consider upgrading to 12-pulse or 18-pulse VFDs",
        "energy_loss_pct": 12.0,
        "cost_per_hour_inr": 1440,
        "industry_example": "Harmonics cause 30% of all unexplained equipment failures and 40% of transformer overheating incidents in industry.",
    },
    4: {
        "fault": "Transients",
        "what_happened": "A sudden, extremely fast voltage spike was detected — like a lightning bolt on the power line. This is a brief but powerful disturbance.",
        "cause": "Lightning strikes near power lines, switching of large electrical equipment, electrostatic discharge, or capacitor bank energization.",
        "impact": "CRITICAL RISK — Can INSTANTLY destroy: circuit boards, MOSFET/IGBT power modules, communication equipment, PLCs, and control systems. This is the #1 cause of electronic equipment failure.",
        "risk_level": "CRITICAL",
        "risk_score": 92,
        "action": "1. URGENT: Inspect all surge protection devices (SPDs) immediately\n2. Check all sensitive electronic equipment for damage\n3. Verify grounding system integrity\n4. Install coordinated surge protection (Type 1 at main panel + Type 2 at sub-panels + Type 3 at equipment)",
        "energy_loss_pct": 3.0,
        "cost_per_hour_inr": 360,
        "industry_example": "A single transient spike can cause ₹8,00,000+ in equipment damage in a data center.",
    },
}


# ─── FASTAPI APP ────────────────────────────────────────────────

app = FastAPI(
    title="GridSense AI",
    description="Industry 5.0 Power Quality Intelligence API",
    version="2.0.0",
)

# Allow dashboard to connect from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── REQUEST/RESPONSE MODELS ───────────────────────────────────

class DetectRequest(BaseModel):
    """
    What the dashboard sends to us.
    These are the sensor readings from the power monitoring system.
    """
    RMS_Voltage: float = 230.0          # Volts
    Peak_Voltage: float = 325.0         # Volts  
    THD: float = 2.0                    # Percentage
    Duration: float = 0.0               # Seconds
    DWT_Energy_Levels: float = 0.5      # Energy units
    DWT_Entropy: float = 0.3            # Entropy units
    Signal_Noise_Ratio_dB: float = 35.0 # Decibels
    Phase: str = "A"                    # A, B, or C


class DetectResponse(BaseModel):
    fault: str
    prediction: int
    confidence: float
    cause: str
    impact: str
    action: str
    risk_level: str
    risk_score: int
    energy_loss_pct: float
    cost_per_hour_inr: float
    what_happened: str
    industry_example: str
    metrics: Dict


# ─── ENDPOINTS ──────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Check if server and model are ready."""
    return {
        "status": "online",
        "model_loaded": model_data is not None,
        "accuracy": model_data["accuracy"] if model_data else None,
        "version": "2.0.0",
    }


@app.post("/detect", response_model=DetectResponse)
def detect_fault(data: DetectRequest):
    """
    MAIN ENDPOINT: Detect power quality fault.
    
    Takes sensor readings → returns fault type + explanation.
    """
    if not model_data:
        raise HTTPException(500, "Model not loaded. Run train_model.py first.")

    # Validate for NaN before any numeric operations
    if math.isnan(data.RMS_Voltage) or math.isnan(data.Peak_Voltage):
        raise HTTPException(422, "RMS_Voltage and Peak_Voltage must be valid numbers, not NaN.")

    # Encode phase
    phase_map = {"A": 0, "B": 1, "C": 2}
    phase_encoded = phase_map.get(data.Phase.upper(), 0)

    # Calculate crest factor (derived feature)
    crest_factor = data.Peak_Voltage / max(data.RMS_Voltage, 0.01)
    
    # Build feature vector (must match training order!)
    features = np.array([[
        data.RMS_Voltage,
        data.Peak_Voltage,
        data.THD,
        data.Duration,
        data.DWT_Energy_Levels,
        data.DWT_Entropy,
        data.Signal_Noise_Ratio_dB,
        phase_encoded,
        crest_factor,
    ]])
    
    # Scale features
    features_scaled = model_data["scaler"].transform(features)
    
    # Predict
    prediction = int(model_data["model"].predict(features_scaled)[0])
    probabilities = model_data["model"].predict_proba(features_scaled)[0]
    confidence = float(probabilities[prediction]) * 100
    
    # Get explanation
    exp = EXPLANATIONS.get(prediction, EXPLANATIONS[0])
    
    # Build response
    return DetectResponse(
        fault=exp["fault"],
        prediction=prediction,
        confidence=round(confidence, 1),
        cause=exp["cause"],
        impact=exp["impact"],
        action=exp["action"],
        risk_level=exp["risk_level"],
        risk_score=exp["risk_score"],
        energy_loss_pct=exp["energy_loss_pct"],
        cost_per_hour_inr=exp["cost_per_hour_inr"],
        what_happened=exp["what_happened"],
        industry_example=exp["industry_example"],
        metrics={
            "rms_voltage": f"{data.RMS_Voltage:.1f} V",
            "thd": f"{data.THD:.1f}%",
            "peak_voltage": f"{data.Peak_Voltage:.1f} V",
            "crest_factor": f"{crest_factor:.3f}",
            "phase": data.Phase,
            "model_accuracy": f"{model_data['accuracy']*100:.1f}%",
        },
    )


@app.post("/simulate")
def simulate_and_detect():
    """
    Generate random sensor readings and detect.
    Used for live demo when no hardware is connected.
    """
    # Randomly pick a fault type (weighted toward normal)
    fault_type = np.random.choice([0, 1, 2, 3, 4], p=[0.4, 0.15, 0.1, 0.2, 0.15])
    
    # Generate realistic readings for that fault type
    if fault_type == 0:  # Normal
        data = DetectRequest(
            RMS_Voltage=round(np.random.normal(230, 4), 1),
            THD=round(np.random.uniform(0.5, 3.0), 1),
            Signal_Noise_Ratio_dB=round(np.random.uniform(30, 45), 1),
            DWT_Energy_Levels=round(np.random.uniform(0.1, 1.0), 2),
            DWT_Entropy=round(np.random.uniform(0.1, 0.5), 2),
        )
    elif fault_type == 1:  # Sag
        data = DetectRequest(
            RMS_Voltage=round(np.random.uniform(130, 200), 1),
            THD=round(np.random.uniform(1.5, 5.0), 1),
            DWT_Energy_Levels=round(np.random.uniform(1.5, 5.0), 2),
            DWT_Entropy=round(np.random.uniform(0.5, 2.0), 2),
            Duration=round(np.random.uniform(0.02, 0.8), 3),
            Signal_Noise_Ratio_dB=round(np.random.uniform(20, 33), 1),
        )
    elif fault_type == 2:  # Swell
        data = DetectRequest(
            RMS_Voltage=round(np.random.uniform(260, 360), 1),
            THD=round(np.random.uniform(1.5, 5.5), 1),
            DWT_Energy_Levels=round(np.random.uniform(2.0, 6.0), 2),
            DWT_Entropy=round(np.random.uniform(0.5, 1.8), 2),
            Duration=round(np.random.uniform(0.01, 0.6), 3),
            Signal_Noise_Ratio_dB=round(np.random.uniform(22, 34), 1),
        )
    elif fault_type == 3:  # Harmonics
        data = DetectRequest(
            RMS_Voltage=round(np.random.normal(228, 8), 1),
            THD=round(np.random.uniform(10, 30), 1),
            DWT_Energy_Levels=round(np.random.uniform(3.0, 8.0), 2),
            DWT_Entropy=round(np.random.uniform(1.5, 3.5), 2),
            Duration=round(np.random.uniform(1, 8), 2),
            Signal_Noise_Ratio_dB=round(np.random.uniform(15, 26), 1),
        )
    else:  # Transients
        rms = round(np.random.normal(232, 8), 1)
        data = DetectRequest(
            RMS_Voltage=rms,
            Peak_Voltage=round(rms * 1.414 * np.random.uniform(1.5, 3.5), 1),
            THD=round(np.random.uniform(3, 8), 1),
            DWT_Energy_Levels=round(np.random.uniform(6, 14), 2),
            DWT_Entropy=round(np.random.uniform(2.5, 5.0), 2),
            Duration=round(np.random.uniform(0.0001, 0.005), 5),
            Signal_Noise_Ratio_dB=round(np.random.uniform(12, 22), 1),
        )
    
    # Set peak voltage if not already set
    if data.Peak_Voltage == 325.0:
        data.Peak_Voltage = round(data.RMS_Voltage * 1.414 * (1 + np.random.normal(0, 0.03)), 1)
    
    data.Phase = np.random.choice(["A", "B", "C"])
    
    # Detect
    result = detect_fault(data)
    
    # Add raw input data
    return {
        "input": data.dict(),
        "result": result.dict(),
    }


@app.get("/fault-types")
def get_fault_types():
    """List all detectable fault types with full explanations."""
    return EXPLANATIONS


# ─── START SERVER ───────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         GridSense AI — API Server Starting              ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Endpoints:                                             ║")
    print("║  GET  /health       → Server status                     ║")
    print("║  POST /detect       → Detect fault from readings        ║")
    print("║  POST /simulate     → Random simulation + detection     ║")
    print("║  GET  /fault-types  → All fault explanations            ║")
    print("║  GET  /docs         → Auto-generated API documentation  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
