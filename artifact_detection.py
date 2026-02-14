import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

def detect_and_clean_artifacts(df, motion_threshold=0.6):
    """
    Cleans physiological signals by detecting motion-induced artifacts.
    
    Logic:
    1. Identify 'artifact windows' where vibration exceeds threshold.
    2. Within these windows, if SpO2 drops or HR spikes abruptly, flag them as artifacts.
    3. Interpolate flagged artifacts and existing NaNs.
    4. Calculate artifact confidence score (1 - ratio of artifact windows).
    """
    df_clean = df.copy()
    
    # 1. Identify windows with high vibration
    df_clean['is_motion'] = df_clean['vibration'] > motion_threshold
    
    # Calculate rolling motion to smooth artifact windows
    df_clean['motion_risk'] = df_clean['is_motion'].rolling(window=5, center=True).max().fillna(0)
    
    # 2. Flag artifacts coupled with motion
    # logic: if motion_risk is high, we are skeptical of sudden vital changes
    
    # Initialize artifact flags
    df_clean['hr_artifact'] = False
    df_clean['spo2_artifact'] = False
    
    # SpO2 artifact: sudden drop (>2% in 1s) coupled with motion
    spo2_diff = df_clean['spo2'].diff().abs()
    df_clean.loc[(df_clean['motion_risk'] > 0.5) & (spo2_diff > 2.0), 'spo2_artifact'] = True
    
    # HR artifact: sudden spike (>5 bpm in 1s) coupled with motion
    hr_diff = df_clean['heart_rate'].diff().abs()
    df_clean.loc[(df_clean['motion_risk'] > 0.5) & (hr_diff > 5.0), 'hr_artifact'] = True

    # Also handle extremely unrealistic values during motion
    df_clean.loc[(df_clean['motion_risk'] > 0.8) & (df_clean['spo2'] < 85), 'spo2_artifact'] = True
    
    # 3. Suppress artifacts and Interpolate
    # Replace flagged artifacts with NaN
    df_clean.loc[df_clean['hr_artifact'], 'heart_rate'] = np.nan
    df_clean.loc[df_clean['spo2_artifact'], 'spo2'] = np.nan
    
    # Linear interpolation for all NaNs (including sensor dropouts)
    df_clean['heart_rate'] = df_clean['heart_rate'].interpolate(method='linear', limit=30)
    df_clean['spo2'] = df_clean['spo2'].interpolate(method='linear', limit=30)
    
    # 4. Artifact Confidence Score per window (rolling 60s window)
    # Higher vibration = Lower confidence
    df_clean['artifact_confidence'] = 1.0 - (df_clean['vibration'].rolling(60, center=True).mean().clip(0, 1))
    
    return df_clean

def plot_cleanup_results(original_df, cleaned_df):
    plt.figure(figsize=(15, 10))
    
    t = original_df['timestamp']
    
    # HR comparison
    plt.subplot(3, 1, 1)
    plt.plot(t, original_df['heart_rate'], 'r', alpha=0.3, label='Original HR')
    plt.plot(t, cleaned_df['heart_rate'], 'b', label='Cleaned HR')
    plt.legend()
    plt.title('Heart Rate Cleaning')
    
    # SpO2 comparison
    plt.subplot(3, 1, 2)
    plt.plot(t, original_df['spo2'], 'g', alpha=0.3, label='Original SpO2')
    plt.plot(t, cleaned_df['spo2'], 'b', label='Cleaned SpO2')
    plt.legend()
    plt.title('SpO2 Cleaning')
    
    # Vibration and Confidence
    plt.subplot(3, 1, 3)
    plt.plot(t, original_df['vibration'], color='gray', alpha=0.5, label='Vibration')
    plt.plot(t, cleaned_df['artifact_confidence'], color='orange', label='Artifact Confidence')
    plt.legend()
    plt.title('Vibration vs Confidence Score')
    
    plt.tight_layout()
    plt.savefig('cleaning_results.png')
    print("Cleanup visualization saved as cleaning_results.png")

if __name__ == "__main__":
    input_file = "ambulance_vitals.csv"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run data_gen.py first.")
    else:
        df = pd.read_csv(input_file)
        cleaned_df = detect_and_clean_artifacts(df)
        
        output_file = "cleaned_vitals.csv"
        cleaned_df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        
        plot_cleanup_results(df, cleaned_df)
