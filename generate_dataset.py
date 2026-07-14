"""
Enhanced dataset generator — 18 features for better BP prediction accuracy.
"""
import numpy as np
import pandas as pd
import random

random.seed(42)
np.random.seed(42)

N = 2000

def simulate():
    data = []
    for _ in range(N):
        age_grp    = np.random.choice([1, 2, 3, 4], p=[0.18, 0.30, 0.30, 0.22])
        gender     = np.random.randint(0, 2)
        fam_hist   = np.random.choice([0, 1], p=[0.45, 0.55])
        bmi_cat    = np.random.choice([0, 1, 2, 3], p=[0.05, 0.35, 0.35, 0.25])
        diabetes   = np.random.choice([0, 1], p=[0.75, 0.25])
        cholesterol= np.random.choice([0, 1, 2], p=[0.50, 0.30, 0.20])
        heart_rate = np.random.choice([0, 1, 2, 3], p=[0.05, 0.60, 0.25, 0.10])
        exercise   = np.random.randint(0, 6)       # 0–5 days/week

        # Compute a base risk score (0–12)
        risk = (age_grp - 1) + fam_hist + bmi_cat + diabetes*2 + cholesterol + \
               (3 - exercise // 2) + max(0, heart_rate - 1)

        # Map risk to systolic BP range
        sys_raw = np.clip(int(risk / 3) + np.random.randint(-1, 2), 0, 4)
        dia_raw = np.clip(int(sys_raw * 0.75) + np.random.randint(-1, 2), 0, 3)

        # Stage target
        if sys_raw == 0 and dia_raw == 0:
            stage = 0
        elif sys_raw <= 1 and dia_raw == 0:
            stage = np.random.choice([0, 1], p=[0.45, 0.55])
        elif sys_raw <= 2 or dia_raw <= 1:
            stage = np.random.choice([1, 2], p=[0.55, 0.45])
        elif sys_raw <= 3 or dia_raw <= 2:
            stage = np.random.choice([2, 3], p=[0.60, 0.40])
        else:
            stage = 3

        # Clinical fields derived from stage
        take_med  = int(stage >= 2 and np.random.rand() > 0.25) or int(np.random.rand() > 0.7)
        severity  = min(stage, 2)
        sob       = int(stage >= 2 and np.random.rand() > 0.3)
        visual    = int(stage >= 2 and np.random.rand() > 0.5)
        nosebleed = int(stage >= 1 and np.random.rand() > 0.5)
        diet      = int(np.random.rand() > 0.5 - stage * 0.08)
        pat_stat  = int(stage >= 1 and np.random.rand() > 0.2)
        time_diag = np.random.choice([0, 1, 2, 3]) if pat_stat else 0

        data.append({
            'Gender':              gender,
            'Age_Group':           age_grp,
            'Family_History':      fam_hist,
            'Patient_Status':      pat_stat,
            'Take_Medication':     take_med,
            'Time_Since_Diagnosis':time_diag,
            'Symptom_Severity':    severity,
            'Shortness_of_Breath': sob,
            'Visual_Changes':      visual,
            'Nosebleeds':          nosebleed,
            'Systolic_BP':         sys_raw,
            'Diastolic_BP':        dia_raw,
            'Controlled_Diet':     diet,
            # NEW
            'BMI_Category':        bmi_cat,
            'Diabetes':            diabetes,
            'Cholesterol_Level':   cholesterol,
            'Heart_Rate_Category': heart_rate,
            'Exercise_Frequency':  exercise,
            'Stage':               stage,
        })
    return pd.DataFrame(data)

df = simulate()
df.to_csv('data/hypertension_data.csv', index=False)
print(f"Dataset: {len(df)} rows, {df.shape[1]-1} features")
print("Stage distribution:\n", df['Stage'].value_counts().rename({0:'Normal',1:'Stage-1',2:'Stage-2',3:'Crisis'}).sort_index())
