# GridSense AI — Deployment Guide

## 🎯 Current Status

✅ **Model trained** (99.37% accuracy)  
✅ **API running** at `http://127.0.0.1:8000`  
✅ **Dashboard integrated** with backend  
✅ **Auto-fallback enabled** (API → local simulation)

---

## 🚀 Local Testing

### 1. Start the API (if not already running)
```bash
cd c:\Users\DELL\OneDrive\Desktop\gridsense\GridSense-AI
python backend/server.py
```
Expected output:
```
✅ Model loaded from ... (accuracy: 99.4%)
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 2. Open the Dashboard
- **Option A:** Open in VS Code Explorer → Right-click `dashboard/GridSense_AI.html` → Open with Live Server
- **Option B:** Double-click `dashboard/GridSense_AI.html` directly
- **Option C:** Browser: `file:///c:/Users/DELL/OneDrive/Desktop/gridsense/GridSense-AI/dashboard/GridSense_AI.html`

### 3. Test Live API Integration
The dashboard will automatically:
- 🔍 Detect your API at `http://127.0.0.1:8000`
- 📊 Call `/simulate` every 5 seconds to get real ML predictions
- 🎯 Display live fault detection with metrics from your trained model
- 📱 Fall back to local simulation if API is unavailable

Check browser console (F12) for:
```
✅ API available at http://127.0.0.1:8000
```

### 4. Test Specific Endpoints Manually

**Try a detection:**
```powershell
$body = @{
    RMS_Voltage = 180
    Peak_Voltage = 250
    THD = 3.5
    Duration = 0.2
    DWT_Energy_Levels = 2.0
    DWT_Entropy = 0.8
    Signal_Noise_Ratio_dB = 28.0
    Phase = "A"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/detect" -ContentType "application/json" -Body $body
```

**View all fault explanations:**
```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/fault-types" | ConvertTo-Json
```

**View API docs (interactive):**
```
http://127.0.0.1:8000/docs
```

---

## ☁️ Cloud Deployment (Render)

### Prerequisites
- GitHub account
- Render account (free tier available)

### Step 1: Push to GitHub
```bash
cd c:\Users\DELL\OneDrive\Desktop\gridsense
git add .
git commit -m "GridSense AI deployment ready"
git push origin main
```

### Step 2: Deploy on Render
1. Go to [render.com](https://render.com)
2. Sign in with GitHub
3. Click **New +** → **Blueprint**
4. Select your `gridsense` repository
5. Render detects `render.yaml` automatically
6. Click **Deploy**

Render will:
- Build Docker image
- Train the ML model during build (~2-3 min)
- Start the API at `https://gridsense-ai-api.onrender.com`

### Step 3: Test Deployed API
```powershell
# Health check
Invoke-RestMethod -Method Get -Uri "https://gridsense-ai-api.onrender.com/health"

# Simulate
Invoke-RestMethod -Method Post -Uri "https://gridsense-ai-api.onrender.com/simulate"

# Docs
# https://gridsense-ai-api.onrender.com/docs
```

### Step 4: Connect Dashboard to Cloud
Update `dashboard/GridSense_AI.html` if needed — it auto-detects based on hostname:
- Local: Uses `http://127.0.0.1:8000`
- Cloud: Uses `https://gridsense-ai-api.onrender.com`

Or manually set in `GridSense_AI.html`:
```javascript
API_URL = 'https://gridsense-ai-api.onrender.com';
```

---

## 🐳 Docker Build Locally (Optional)

```bash
# Build image
docker build -t gridsense-ai .

# Run container
docker run -p 8000:8000 gridsense-ai

# Test
curl http://localhost:8000/health
```

---

## 📊 API Endpoints Reference

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Server status & model info |
| POST | `/detect` | Detect fault from sensor readings |
| POST | `/simulate` | Generate random signal + detect |
| GET | `/fault-types` | All fault explanations |
| GET | `/docs` | Interactive API documentation |

---

## 🎮 Dashboard Features Now Live

✅ **Real-time Detection** — Uses your ML model inference  
✅ **Live Metrics** — RMS, THD, Peak Voltage, Crest Factor  
✅ **Risk Score Gauge** — Real-time equipment risk (0-100)  
✅ **Waveform Visualization** — Canvas-based signal display  
✅ **Energy Loss Calculator** — ₹/hour cost impact  
✅ **Fault History** — Recent alerts timeline  
✅ **Chat Interface** — Natural language queries  
✅ **Voice Commands** — "What is the current fault?" (Chrome/Edge)  
✅ **FFT Spectrum** — Harmonic analysis (H1-H13)  
✅ **Trend Analysis** — 24-hour THD and voltage trends  

---

## 🔧 Environment Variables (Cloud Deployment)

When deploying, you can optionally set:
- `PORT` — API port (default: 8000)
- `MODEL_PATH` — Path to trained model (auto-detected)

Render automatically exposes your service to the internet with HTTPS.

---

## 📈 Next Steps

1. **Test with real sensor data** — Replace `/simulate` with hardware PDU readings
2. **Add authentication** — JWT tokens for secure API access
3. **Mobile app** — React Native frontend connecting to this API
4. **IoT integration** — MQTT broker for distributed monitoring
5. **Database** — Store predictions and alerts for analytics
6. **Multi-site clustering** — Monitor multiple feeders/facilities

---

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| API not responding | Check if `python backend/server.py` is running |
| Dashboard shows "Using local simulation" | API is unreachable; check firewall or port 8000 |
| CORS errors | Browser blocks cross-origin; API handles this with CORSMiddleware |
| Model not loading | Run `python ml/train_model.py` first |
| Docker build fails | Ensure `requirements.txt` and `data/power_quality_fault_dataset.csv` exist |

---

## 📞 Support

For issues or questions:
- Check `/health` endpoint to verify API status
- Review `/docs` for API contract
- Check browser console (F12) for JavaScript errors
- Verify Python version: `python --version` (3.10+)

---

**⚡ GridSense AI is now deployment-ready. Choose your platform and launch! ⚡**
