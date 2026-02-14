import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

def generate_ambulance_data(duration_sec=1800, sampling_rate=1):
    """
    Generates synthetic physiological time-series data for a smart ambulance system.
    
    Parameters:
        duration_sec (int): Total duration in seconds.
        sampling_rate (int): Samples per second.
        
    Returns:
        pd.DataFrame: Generated data.
    """
    t = np.arange(0, duration_sec, 1/sampling_rate)
    n = len(t)
    
    # Base Vitals (Normal ranges)
    hr = np.full(n, 75.0)
    spo2 = np.full(n, 98.0)
    bp_sys = np.full(n, 120.0)
    bp_dia = np.full(n, 80.0)
    motion = np.random.normal(0.1, 0.05, n) # Background vibration
    distress_ground_truth = np.zeros(n)
    
    # 1. Normal transport noise (small fluctuations)
    hr += np.random.normal(0, 1, n)
    spo2 += np.random.normal(0, 0.2, n)
    bp_sys += np.random.normal(0, 2, n)
    bp_dia += np.random.normal(0, 1.5, n)
    
    # 2. Gradual patient deterioration (e.g., from 15 min to 25 min)
    deterioration_start = 900
    deterioration_end = 1500
    det_mask = (t >= deterioration_start) & (t <= deterioration_end)
    
    # Linear deterioration trends
    hr[det_mask] += np.linspace(0, 40, np.sum(det_mask)) # HR increases to 115+
    spo2[det_mask] -= np.linspace(0, 8, np.sum(det_mask)) # SpO2 drops to ~90%
    bp_sys[det_mask] += np.linspace(0, 30, np.sum(det_mask)) # BP increases (stress)
    distress_ground_truth[t >= deterioration_start + 100] = 1 # Label distress after trend is established
    
    # 3. Motion-induced artifacts
    # Scenario: Large bump/vibration episodes
    artifact_windows = [(300, 330), (700, 720), (1200, 1230)]
    for start, end in artifact_windows:
        mask = (t >= start) & (t <= end)
        motion[mask] += np.random.uniform(0.5, 1.2, np.sum(mask)) # High vibration
        
        # Coupled artifacts
        # SpO2 drops during high motion (sensor decoupling)
        spo2[mask] -= np.random.uniform(5, 15, np.sum(mask))
        
        # HR spikes during bumps
        hr[mask] += np.random.uniform(10, 20, np.sum(mask))
        
    # 4. Sensor dropout (missing HR and SpO2 for short intervals)
    dropout_windows = [(1000, 1010), (1600, 1605)]
    for start, end in dropout_windows:
        mask = (t >= start) & (t <= end)
        hr[mask] = np.nan
        spo2[mask] = np.nan
        
    # Clip values to physiological limits
    hr = np.clip(hr, 40, 200)
    spo2 = np.clip(spo2, 60, 100)
    bp_sys = np.clip(bp_sys, 60, 220)
    bp_dia = np.clip(bp_dia, 40, 130)
    
    df = pd.DataFrame({
        'timestamp': t,
        'heart_rate': hr,
        'spo2': spo2,
        'bp_systolic': bp_sys,
        'bp_diastolic': bp_dia,
        'vibration': motion,
        'distress_label': distress_ground_truth
    })
    
    return df

if __name__ == "__main__":
    print("Generating synthetic physiological data...")
    df = generate_ambulance_data()
    
    output_path = "ambulance_vitals.csv"
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    
    # Basic statistics
    print("\nData Summary:")
    print(df.describe())
    
    # Basic Plotting for verification
    plt.figure(figsize=(12, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(df['timestamp'], df['heart_rate'], label='Heart Rate (bpm)')
    plt.plot(df['timestamp'], df['distress_label']*100, 'r--', alpha=0.3, label='Ground Truth Distress')
    plt.legend()
    plt.ylabel('HR / Distress')
    
    plt.subplot(4, 1, 2)
    plt.plot(df['timestamp'], df['spo2'], label='SpO2 (%)', color='green')
    plt.legend()
    plt.ylabel('SpO2')
    
    plt.subplot(4, 1, 3)
    plt.plot(df['timestamp'], df['bp_systolic'], label='BP Systolic', color='purple')
    plt.plot(df['timestamp'], df['bp_diastolic'], label='BP Diastolic', color='orange')
    plt.legend()
    plt.ylabel('BP')
    
    plt.subplot(4, 1, 4)
    plt.plot(df['timestamp'], df['vibration'], label='Vibration (Motion)', color='gray')
    plt.legend()
    plt.ylabel('Motion')
    plt.xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig('vitals_simulation.png')
    print("Verification plot saved as vitals_simulation.png")
