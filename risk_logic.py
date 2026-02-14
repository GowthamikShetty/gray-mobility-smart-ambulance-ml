import pandas as pd
import numpy as np

def calculate_risk_and_alerts(anomaly_df, risk_threshold=0.6, confidence_threshold=0.7):
    """
    Computes real-time risk scores and triggers alerts based on safety-critical logic.
    """
    risk_df = anomaly_df.copy()
    
    # 1. Normalized Risk Score (0-1)
    # Combination of physiological instability (slopes and means)
    # We use a weighted sum of normalized deviations
    
    # Normalize HR (Base 75, High 120)
    hr_risk = (risk_df['heart_rate_mean'] - 75) / (120 - 75)
    hr_risk = hr_risk.clip(0, 1) * 0.4 # 40% weight
    
    # Normalize SpO2 (Base 98, Low 90)
    spo2_risk = (98 - risk_df['spo2_mean']) / (98 - 90)
    spo2_risk = spo2_risk.clip(0, 1) * 0.4 # 40% weight
    
    # Normalize BP (Base 120, High 160)
    bp_risk = (risk_df['bp_systolic_mean'] - 120) / (160 - 120)
    bp_risk = bp_risk.clip(0, 1) * 0.2 # 20% weight
    
    risk_df['risk_score'] = (hr_risk + spo2_risk + bp_risk).clip(0, 1)
    
    # Increase risk if symptoms are worsening (positive slopes for HR/BP, negative for SpO2)
    slope_multiplier = 1.0
    if 'heart_rate_slope' in risk_df:
        # If HR is rising AND SpO2 is falling, it's a critical synergy
        critical_synergy = (risk_df['heart_rate_slope'] > 0.02) & (risk_df['spo2_slope'] < -0.005)
        risk_df.loc[critical_synergy, 'risk_score'] *= 1.2
        risk_df['risk_score'] = risk_df['risk_score'].clip(0, 1)

    # 2. Confidence Score (Already mostly calculated as artifact_confidence)
    # We can further penalize if variance is extremely high (unstable sensors)
    risk_df['sensor_stability'] = 1.0 - (risk_df['heart_rate_var'] / 100).clip(0, 0.5)
    risk_df['final_confidence'] = (risk_df['confidence'] * 0.7 + risk_df['sensor_stability'] * 0.3)

    # 3. Alert Rules
    # Alert triggers ONLY when:
    # - Risk is high
    # - Confidence is acceptable
    # - Trend persists (implemented here as risk > threshold for 2 consecutive windows)
    
    risk_df['risk_threshold_breached'] = risk_df['risk_score'] > risk_threshold
    risk_df['confidence_acceptable'] = risk_df['final_confidence'] > confidence_threshold
    
    # Persistence check (current and previous window)
    risk_df['persistent_risk'] = risk_df['risk_threshold_breached'] & risk_df['risk_threshold_breached'].shift(1).fillna(False)
    
    # Alert Trigger
    risk_df['alert_triggered'] = risk_df['persistent_risk'] & risk_df['confidence_acceptable']
    
    # Explainability comments
    def get_alert_comment(row):
        if row['alert_triggered']:
            return f"CRITICAL: High risk ({row['risk_score']:.2f}) with stable sensor ({row['final_confidence']:.2f}). {row['reasons']}"
        elif row['risk_threshold_breached'] and not row['confidence_acceptable']:
            return f"SUPPRESSED: High risk ({row['risk_score']:.2f}) but low confidence ({row['final_confidence']:.2f}) due to motion/artifacts."
        elif row['risk_threshold_breached'] and not row['persistent_risk']:
            return "WAITING: Risk threshold breached but awaiting trend persistence."
        return "Normal status."

    risk_df['alert_comment'] = risk_df.apply(get_alert_comment, axis=1)
    
    return risk_df

if __name__ == "__main__":
    try:
        anomaly_df = pd.read_csv("anomaly_results.csv")
        risk_results = calculate_risk_and_alerts(anomaly_df)
        risk_results.to_csv("risk_results.csv", index=False)
        print("Risk scoring complete. Results saved to risk_results.csv")
        
        # Show some active alerts
        alerts = risk_results[risk_results['alert_triggered']]
        if not alerts.empty:
            print(f"\nDetected {len(alerts)} alert windows.")
            print(alerts[['timestamp', 'risk_score', 'final_confidence', 'alert_comment']].head())
        else:
            print("\nNo alerts triggered in this dataset.")
            
    except FileNotFoundError:
        print("Error: anomaly_results.csv not found. Run anomaly_model.py first.")
