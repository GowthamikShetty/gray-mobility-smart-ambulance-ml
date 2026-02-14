import pandas as pd
import numpy as np

def evaluate_performance(vitals_df, risk_df):
    """
    Evaluates the alert system against ground truth distress labels.
    """
    # Merge ground truth into risk_df based on timestamp
    # We take the max distress label in the window preceding the alert timestamp
    risk_eval = risk_df.copy()
    
    gt_labels = []
    for ts in risk_eval['timestamp']:
        # Find distress labels in the 30s window ending at ts
        window_gt = vitals_df[(vitals_df['timestamp'] <= ts) & (vitals_df['timestamp'] > ts - 30)]['distress_label']
        gt_labels.append(1 if window_gt.max() > 0.5 else 0)
    
    risk_eval['ground_truth'] = gt_labels
    
    # Metrics
    tp = len(risk_eval[(risk_eval['alert_triggered'] == True) & (risk_eval['ground_truth'] == 1)])
    fp = len(risk_eval[(risk_eval['alert_triggered'] == True) & (risk_eval['ground_truth'] == 0)])
    fn = len(risk_eval[(risk_eval['alert_triggered'] == False) & (risk_eval['ground_truth'] == 1)])
    tn = len(risk_eval[(risk_eval['alert_triggered'] == False) & (risk_eval['ground_truth'] == 0)])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # False Alert Rate (FAR) - Percentage of alerts that are false
    far = fp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Alert Latency
    # Time from first GT=1 to first Alert=True
    gt_start = vitals_df[vitals_df['distress_label'] == 1]['timestamp'].min()
    alert_start = risk_eval[risk_eval['alert_triggered'] == True]['timestamp'].min()
    latency = alert_start - gt_start if not np.isnan(alert_start) and not np.isnan(gt_start) else None

    print("--- EVALUATION REPORT ---")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1:.2f}")
    print(f"False Alert Rate: {far:.2f}")
    if latency is not None:
        print(f"Alert Latency: {latency:.1f} seconds")
    else:
        print("Alert Latency: N/A (No alerts or no distress detected)")
    print("-------------------------")

    # Failure Case Analysis
    print("\n--- FAILURE CASE ANALYSIS ---")
    print("1. Motion-Heavy Deterioration")
    print("   - WHAT: Alert suppressed during genuine deterioration because of high vibration.")
    print("   - WHY: Safety logic prioritizes confidence; high motion makes sensor data untrustworthy.")
    print("   - IMPROVEMENT: Use redundant sensors or more robust signal processing (e.g., adaptive filtering) to maintain confidence.")
    
    print("\n2. Sudden Event Latency")
    print("   - WHAT: The 2-window persistence requirement delays alerts by ~20-30 seconds.")
    print("   - WHY: To prevent false alarms from transient spikes.")
    print("   - IMPROVEMENT: Implementation of a 'Fast-Track' alert for extreme, immediate breaches (e.g., Asystole).")
    
    print("\n3. Gradual Drift Under Threshold")
    print("   - WHAT: If a patient deteriorates very slowly and stays just below thresholds, no alert triggers.")
    print("   - WHY: Thresholds are static and rule-based.")
    print("   - IMPROVEMENT: Context-aware thresholds based on patient baseline and medical history.")

if __name__ == "__main__":
    try:
        vitals_df = pd.read_csv("ambulance_vitals.csv")
        risk_df = pd.read_csv("risk_results.csv")
        evaluate_performance(vitals_df, risk_df)
    except FileNotFoundError:
        print("Error: Files missing. Ensure data_gen, anomaly_model, and risk_logic have run.")
