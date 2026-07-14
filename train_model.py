"""
Enhanced model trainer — 7 algorithms + Voting Ensemble.
Saves best model and model comparison stats as JSON.
"""
import json
import pickle
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ─── Load ───────────────────────────────────────────────
df = pd.read_csv('data/hypertension_data.csv')
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]-1} features")

FEATURES = [
    'Gender','Age_Group','Family_History','Patient_Status',
    'Take_Medication','Time_Since_Diagnosis','Symptom_Severity',
    'Shortness_of_Breath','Visual_Changes','Nosebleeds',
    'Systolic_BP','Diastolic_BP','Controlled_Diet',
    'BMI_Category','Diabetes','Cholesterol_Level',
    'Heart_Rate_Category','Exercise_Frequency'
]
X = df[FEATURES]
y = df['Stage']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ─── Individual models ──────────────────────────────────
lr  = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
dt  = DecisionTreeClassifier(max_depth=8, random_state=42)
rf  = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
svm = SVC(kernel='rbf', probability=True, C=2.0, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
rc  = RidgeClassifier()
gnb = GaussianNB()
gb  = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                  learning_rate=0.08, random_state=42)

# Voting Ensemble (soft = average probabilities)
voting = VotingClassifier(
    estimators=[('lr', lr), ('gb', gb), ('rf', rf)],
    voting='soft'
)

# RidgeClassifier can't do predict_proba, handle separately
individual_models = {
    'Logistic Regression': lr,
    'Decision Tree':       dt,
    'Random Forest':       rf,
    'SVM':                 svm,
    'KNN':                 knn,
    'Ridge Classifier':    rc,
    'Gaussian Naive Bayes':gnb,
    'Gradient Boosting':   gb,
    'Voting Ensemble':     voting,
}

print("\n" + "="*65)
print("  MODEL COMPARISON (18 Features)")
print("="*65)

results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in individual_models.items():
    pipe = Pipeline([('scaler', MinMaxScaler()), ('clf', clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    try:
        cv  = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy').mean()
    except Exception:
        cv  = acc
    results[name] = {'pipeline': pipe, 'test_acc': acc, 'f1': f1, 'cv_acc': cv}
    print(f"  {name:<25} Test={acc*100:.1f}%  CV={cv*100:.1f}%  F1={f1:.2f}")

print("="*65)

# ─── Select best generalizing model (highest CV) ────────
best_name = max(results, key=lambda n: results[n]['cv_acc'])
print(f"\n✅ Best Model: {best_name}")
best_pipe = results[best_name]['pipeline']

# ─── Detailed Classification Report ─────────────────────
print(f"\n── {best_name} Classification Report ──")
y_pred_best = best_pipe.predict(X_test)
print(classification_report(y_test, y_pred_best,
      target_names=['Normal','Stage-1','Stage-2','Crisis']))

# ─── Save best model ─────────────────────────────────────
with open('logreg_model.pkl', 'wb') as f:
    pickle.dump(best_pipe, f)
with open('model_features.pkl', 'wb') as f:
    pickle.dump(FEATURES, f)

# ─── Save model comparison stats for the UI chart ────────
comparison = {
    name: {
        'test_acc': round(v['test_acc'] * 100, 1),
        'cv_acc':   round(v['cv_acc'] * 100, 1),
        'f1':       round(v['f1'], 3),
    }
    for name, v in results.items()
}
with open('static/model_comparison.json', 'w') as f:
    json.dump({'models': comparison, 'best': best_name,
               'features': len(FEATURES)}, f, indent=2)

print(f"\n📊 Saved model_comparison.json")
print(f"   Best CV Accuracy: {results[best_name]['cv_acc']*100:.1f}%")
print(f"   Features used: {len(FEATURES)}")
