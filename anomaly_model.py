import pandas as pd
import numpy as np
from scipy.stats import linregress

def extract_features(window):
    """
    Extracts mean, slope, and variance from a temporal window of vitals.
    """
    features = {}
    for col in ['heart_rate', 'spo2', 'bp_systolic']:
        v = window[col].values
        t = np.arange(len(v))
        
        features[f'{col}_mean'] = np.mean(v)
        features[f'{col}_var'] = np.var(v)
        
        # Slope using linear regression
        if len(v) > 1 and not np.any(np.isnan(v)):
            slope, _, _, _, _ = linregress(t, v)
            features[f'{col}_slope'] = slope
        else:
            features[f'{col}_slope'] = 0.0
            
    # Artifact confidence mean in this window
    features['confidence_mean'] = window['artifact_confidence'].mean()
    
    return features

def detect_anomalies(df, window_size=30, step_size=10):
    """
    Sliding window anomaly detection.
    """
    anomalies = []
    
    for start in range(0, len(df) - window_size, step_size):
        end = start + window_size
        window = df.iloc[start:end]
        
        feats = extract_features(window)
        timestamp = window['timestamp'].iloc[-1]
        
        # DETECTION LOGIC (Rule-based)
        is_anomaly = False
        reasons = []
        
        # 1. Tachycardia Trend: Increasing HR slope + High mean HR
        if feats['heart_rate_slope'] > 0.05 and feats['heart_rate_mean'] > 100:
            is_anomaly = True
            reasons.append("Rising HR trend")
            
        # 2. Desaturation Trend: Decreasing SpO2 slope
        if feats['spo2_slope'] < -0.01 and feats['spo2_mean'] < 95:
            is_anomaly = True
            reasons.append("Declining SpO2 trend")
            
        # 3. Hypertension Trend: Rising BP
        if feats['bp_systolic_slope'] > 0.1 and feats['bp_systolic_mean'] > 140:
            is_anomaly = True
            reasons.append("Rising Systolic BP")

        # Suppression Logic: If confidence is low, we flag but mark as 'low confidence anomaly'
        confidence = feats['confidence_mean']
        
        # Risk persistence (simplified here as per-window logic, 
        # but in risk_logic.py we will aggregate)
        
        anomalies.append({
            'timestamp': timestamp,
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'reasons': "; ".join(reasons),
            **feats
        })
        
    return pd.DataFrame(anomalies)

if __name__ == "__main__":
    df = pd.read_csv("cleaned_vitals.csv")
    anomaly_results = detect_anomalies(df)
    anomaly_results.to_csv("anomaly_results.csv", index=False)
    print("Anomaly detection complete. Results saved to anomaly_results.csv")
    print(anomaly_results[anomaly_results['is_anomaly'] == True].head())
