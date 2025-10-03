#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Landslide-Hazard-Prediction-using-Geospatial-ML-Models (Synthetic Data)
------------------------------------------------------------------------
This script generates a reasonably populated synthetic geospatial dataset and
trains multiple ML models (Logistic Regression, Random Forest, Gradient Boosting)
to predict landslide hazard. It saves:
  - synthetic_landslide_data.csv
  - roc_curve.png
  - pr_curve.png
  - confusion_matrix.png
  - feature_importance.png
Usage:
  python landslide_hazard_geospatial_ml.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, confusion_matrix, classification_report)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -----------------------------
# 1) Generate synthetic geospatial dataset
# -----------------------------
# Grid of coordinates (approximate degrees) – generic mountainous region
n_points = 5000  # reasonably populated
lat_min, lat_max = 6.0, 7.5
lon_min, lon_max = 5.0, 7.0

lats = np.random.uniform(lat_min, lat_max, n_points)
lons = np.random.uniform(lon_min, lon_max, n_points)

# Elevation model: base gradient + smooth noise
base_elev = 100 + 900 * (lats - lat_min) / (lat_max - lat_min)  # higher in the north
noise = np.random.normal(0, 30, n_points)
elevation = base_elev + noise  # meters

# Approximate slope as function of local gradients + noise (0–60 degrees)
slope = np.clip(0.05 * (elevation - elevation.min()) + np.random.normal(5, 5, n_points), 0, 60)

# Aspect (0–360 degrees)
aspect = np.mod(np.random.vonmises(mu=0, kappa=0.1, size=n_points) * 180/np.pi, 360)

# Curvature (concave negative, convex positive), scaled small
curvature = np.random.normal(0, 0.5, n_points) - 0.002 * (slope - slope.mean())

# Rainfall (mm/yr): higher in south-east corner + noise
rain_grad = 1500 + 800 * ((lat_max - lats) / (lat_max - lat_min)) + 400 * ((lons - lon_min) / (lon_max - lon_min))
rainfall = rain_grad + np.random.normal(0, 50, n_points)

# Simulate "river" and "fault" linear features; compute planar distance
def min_distance_to_lines(x, y, lines):
    # lines: list of dicts with 'a, b, c' for ax + by + c = 0
    d_all = []
    for ln in lines:
        a, b, c = ln['a'], ln['b'], ln['c']
        d = np.abs(a*x + b*y + c) / np.sqrt(a*a + b*b)
        d_all.append(d)
    return np.min(np.vstack(d_all), axis=0)

# Define 2 river lines and 2 fault lines in lon-lat space
river_lines = [
    {'a': 1.0, 'b': -1.0, 'c': -1.0},   # roughly diagonal
    {'a': 0.5, 'b': 1.0, 'c': -8.0}
]
fault_lines = [
    {'a': -1.0, 'b': -0.5, 'c': 9.0},
    {'a': 0.3, 'b': -1.0, 'c': 4.0}
]

dist_to_river = min_distance_to_lines(lons, lats, river_lines)
dist_to_fault = min_distance_to_lines(lons, lats, fault_lines)

# Lithology (categorical): weak sedimentary, metamorphic, igneous, unconsolidated
lithologies = np.random.choice(['sedimentary', 'metamorphic', 'igneous', 'unconsolidated'],
                               size=n_points, p=[0.35, 0.25, 0.2, 0.2])

# Landcover (categorical): forest, agriculture, urban, grassland
landcover = np.random.choice(['forest', 'agriculture', 'urban', 'grassland'],
                             size=n_points, p=[0.35, 0.35, 0.15, 0.15])

# Soil clay content (%) and NDVI (–0.1 to 0.9)
soil_clay = np.clip(np.random.normal(30, 10, n_points) - 0.05 * slope + 0.02 * rainfall/100.0, 5, 70)
ndvi = np.clip(0.7 - 0.007 * slope + np.random.normal(0, 0.05, n_points), -0.1, 0.9)

# Road distance: create random road centrelines by points; use nearest distance to a few road points
n_roads = 8
road_pts = np.column_stack([np.random.uniform(lon_min, lon_max, n_roads),
                            np.random.uniform(lat_min, lat_max, n_roads)])
dist_to_road = np.min(np.sqrt((lons[:, None] - road_pts[None, :, 0])**2 +
                              (lats[:, None] - road_pts[None, :, 1])**2), axis=1)

# -----------------------------
# 2) Construct hazard probability and labels (logistic model + noise)
# -----------------------------
# Map lithology vulnerability
litho_weight = {'unconsolidated': 0.8, 'sedimentary': 0.4, 'metamorphic': 0.2, 'igneous': 0.1}
litho_v = np.array([litho_weight[l] for l in lithologies])

# Landcover weight (less vegetation -> higher hazard)
lc_weight = {'urban': 0.6, 'agriculture': 0.4, 'grassland': 0.3, 'forest': 0.1}
lc_v = np.array([lc_weight[l] for l in landcover])

# Standardize key continuous predictors to build a logit
def z(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-9)

logit = (
    1.2 * z(slope) +
    0.6 * z(rainfall) +
    0.5 * z(-ndvi) +
    0.7 * z(-dist_to_fault) +
    0.5 * z(-dist_to_river) +
    0.3 * z(curvature) +
    0.4 * z(soil_clay) +
    1.0 * z(litho_v) +
    0.6 * z(lc_v) +
    0.3 * z(-dist_to_road) +
    np.random.normal(0, 0.5, n_points)  # irreducible noise
)

# Convert logit to probability (sigmoid) and sample labels
prob = 1.0 / (1.0 + np.exp(-logit))
labels = (np.random.rand(n_points) < prob).astype(int)

# -----------------------------
# 3) Build DataFrame and save CSV
# -----------------------------
df = pd.DataFrame({
    'lat': lats,
    'lon': lons,
    'elevation_m': elevation,
    'slope_deg': slope,
    'aspect_deg': aspect,
    'curvature': curvature,
    'rainfall_mm': rainfall,
    'dist_to_river_deg': dist_to_river,
    'dist_to_fault_deg': dist_to_fault,
    'dist_to_road_deg': dist_to_road,
    'soil_clay_pct': soil_clay,
    'ndvi': ndvi,
    'lithology': lithologies,
    'landcover': landcover,
    'landslide': labels
})
df.to_csv('synthetic_landslide_data.csv', index=False)

# -----------------------------
# 4) Train/test split
# -----------------------------
X = df.drop(columns=['landslide'])
y = df['landslide']

num_cols = ['elevation_m', 'slope_deg', 'aspect_deg', 'curvature', 'rainfall_mm',
            'dist_to_river_deg', 'dist_to_fault_deg', 'dist_to_road_deg',
            'soil_clay_pct', 'ndvi', 'lat', 'lon']
cat_cols = ['lithology', 'landcover']

preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

models = {
    'LogisticRegression': LogisticRegression(max_iter=200, n_jobs=None, random_state=RANDOM_STATE),
    'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2,
                                           n_jobs=-1, random_state=RANDOM_STATE),
    'GradientBoosting': GradientBoostingClassifier(random_state=RANDOM_STATE)
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

# -----------------------------
# 5) Cross-validate and pick best by ROC-AUC
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}

for name, clf in models.items():
    pipe = Pipeline(steps=[('prep', preprocess), ('clf', clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_results[name] = scores.mean()

best_model_name = max(cv_results, key=cv_results.get)
best_clf = models[best_model_name]
best_pipe = Pipeline(steps=[('prep', preprocess), ('clf', best_clf)])
best_pipe.fit(X_train, y_train)

# -----------------------------
# 6) Evaluate on test set
# -----------------------------
y_prob = best_pipe.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

roc_auc = roc_auc_score(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

report = classification_report(y_test, y_pred, digits=3)

print("Cross-validated ROC-AUC (train) per model:")
for k, v in cv_results.items():
    print(f"  {k}: {v:.3f}")
print("\nBest model:", best_model_name)
print(f"Test ROC-AUC: {roc_auc:.3f}")
print(f"Test Average Precision (PR-AUC): {ap:.3f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# -----------------------------
# 7) Plots: ROC, PR, Confusion Matrix, Feature Importance
# -----------------------------
# ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Landslide Hazard')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300)
plt.close()

# PR
prec, rec, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(rec, prec, label=f'{best_model_name} (AP = {ap:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Landslide Hazard')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('pr_curve.png', dpi=300)
plt.close()

# Confusion Matrix
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['No LS', 'LS'])
plt.yticks(tick_marks, ['No LS', 'LS'])
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()

# Feature importance (support both tree-based and linear models)
# Retrieve fitted classifier from pipeline
fitted_clf = best_pipe.named_steps['clf']

# Get feature names after preprocessing
ohe = best_pipe.named_steps['prep'].named_transformers_['cat']
num_features = num_cols
cat_features = list(ohe.get_feature_names_out(cat_cols))
all_features = num_features + cat_features

# Compute importances
if hasattr(fitted_clf, 'feature_importances_'):
    importances = fitted_clf.feature_importances_
elif hasattr(fitted_clf, 'coef_'):
    importances = np.abs(fitted_clf.coef_).ravel()
else:
    # Fallback: permutation-like proxy using standard deviation of SHAP-like scores (not available)
    importances = np.zeros(len(all_features))

# Align length in case of any mismatch (rare, but safe)
n_min = min(len(importances), len(all_features))
fi = pd.Series(importances[:n_min], index=all_features[:n_min]).sort_values(ascending=False)[:20]

plt.figure(figsize=(8,6))
plt.barh(fi.index[::-1], fi.values[::-1])
plt.xlabel('Importance')
plt.title(f'Feature Importance ({best_model_name})')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.close()

print("\nSaved files:")
print("  synthetic_landslide_data.csv")
print("  roc_curve.png")
print("  pr_curve.png")
print("  confusion_matrix.png")
print("  feature_importance.png")
