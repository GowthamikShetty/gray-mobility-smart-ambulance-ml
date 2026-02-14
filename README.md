🚑 Smart Ambulance AI – Artifact-Aware Time-Series Monitoring

🧠 Project Context

This project simulates a Smart Ambulance real-time patient monitoring system designed to operate under noisy, vibration-heavy transport conditions.

Unlike ICU environments, ambulance signals are:
Corrupted by motion artifacts
Interrupted by sensor dropouts
Non-stationary
Safety-critical

The goal is early deterioration detection with controlled false alerts, not perfect classification accuracy.

📊 System Overview
🔁 End-to-End Pipeline
Diagram
flowchart TD
    A[Synthetic Vitals] --> B[Artifact Detection]
    B --> C[Signal Cleaning]
    C --> D[Feature Extraction]
    D --> E[Anomaly Detection]
    E --> F[Risk Score + Confidence]
    F --> G[API Output]

📈 Example Signal Behavior
1️⃣ Raw Heart Rate (With Motion Artifacts)
| Stable 75 bpm ----
|     ^ spike
|     ^ spike
| Gradual rise during distress

2️⃣ SpO₂ During Motion vs True Distress

Sudden drop + high motion → artifact

Gradual drop + low motion → real deterioration

3️⃣ Motion Signal

Mostly low baseline

Short sharp spikes (road bumps)

🏗 Repository Structure
gray-mobility-smart-ambulance-ml/
│
├── data/                  # Generated vitals
├── plots/                 # Before/after cleaning visuals
├── src/
│   ├── data_gen.py
│   ├── artifact_detection.py
│   ├── anomaly_model.py
│   ├── risk_logic.py
│   ├── evaluate.py
│
├── api/
│   └── app.py             # FastAPI service
│
├── run_pipeline.py
├── report.md
├── requirements.txt
└── README.md


Modular structure ensures reproducibility and avoids notebook-only submission.

⚙️ How to Run
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run Full Pipeline
python run_pipeline.py


This will:

Generate synthetic data
Clean artifacts
Detect anomalies
Compute risk scores
Evaluate alerts
Save plots

3️⃣ Launch API
uvicorn api.app:app --reload


Open:

http://127.0.0.1:8000/docs

📥 Sample API Output
{
  "anomaly": true,
  "risk_score": 0.76,
  "confidence": 0.84
}

Field	Meaning
anomaly	Whether early deterioration is detected
risk_score	Combined multi-vital instability score
confidence	Signal reliability estimate
🧪 Evaluation Metrics

The system reports:

✅ Precision
✅ Recall
✅ False Alert Rate
✅ Alert Latency

Ambulance Context Trade-off
False negatives (missed deterioration) are most dangerous
Some false positives are acceptable

Alerts must be explainable
⚠️ Failure Analysis (Key Insight)

Three analyzed failure cases:
Motion masking early deterioration
Slow physiological drift detection delay
Sensor dropout lowering confidence excessively
Each failure includes mitigation suggestions in report.md.

🛑 Safety-Critical Principles

This system is designed as decision support, not medical automation.
It should NEVER:
Replace clinicians
Trigger treatment automatically
Make final medical decisions
AI assists. Humans decide.

💡 Design Philosophy

✔ Explainability over black-box models
✔ Robust trend detection over threshold hacks
✔ Engineering discipline over notebook experiments
✔ Safety-first thinking

📌 Technical Stack

Python

NumPy / Pandas
Scikit-learn (statistical modeling)
Matplotlib
FastAPI
Uvicorn

🙌 Final Note

Building ML for safety-critical environments requires thinking beyond accuracy metrics.
This project emphasizes artifact awareness, engineering structure, and risk reasoning — aligning with real-world ambulance constraints.
