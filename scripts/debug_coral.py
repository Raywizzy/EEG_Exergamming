#!/usr/bin/env python3
"""
Debug CORAL data type issue
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('src')
from domain_adaptation.coral import CORAL

# Load datasets
df2778 = pd.read_csv('data/features/core5/ds002778_core5_features.csv')
df4584 = pd.read_csv('data/features/core5/ds004584_core5_features.csv')

core5_features = ['duration_cv', 'duty_cycle', 'mean_duration_ms', 'median_duration_ms', 'motor_posterior_duty_ratio']

print("=== Extracting features ===")
X_source = df2778[core5_features].values  # ds002778 as source
X_target = df4584[core5_features].values  # ds004584 as target

print(f"Source shape: {X_source.shape}, dtype: {X_source.dtype}")
print(f"Target shape: {X_target.shape}, dtype: {X_target.dtype}")

print("\n=== Testing StandardScaler ===")
try:
    scaler_source = StandardScaler()
    scaler_target = StandardScaler()

    source_norm = scaler_source.fit_transform(X_source)
    target_norm = scaler_target.fit_transform(X_target)

    print(f"Source normalized: {source_norm.shape}, dtype: {source_norm.dtype}")
    print(f"Target normalized: {target_norm.shape}, dtype: {target_norm.dtype}")

    print("\n=== Testing CORAL ===")
    coral = CORAL(reg_param=1e-6)

    print("Fitting CORAL...")
    coral.fit(source_norm, target_norm)

    print("Transforming source...")
    adapted_source = coral.transform_source(source_norm)
    print(f"Adapted source: {adapted_source.shape}, dtype: {adapted_source.dtype}")

    print("✅ CORAL test successful!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Detailed data inspection ===")
print("Source data sample:")
print(X_source[:2])
print("Source data types per column:")
for i, col in enumerate(core5_features):
    print(f"  {col}: {type(X_source[0, i])}")

print("\nTarget data sample:")
print(X_target[:2])
print("Target data types per column:")
for i, col in enumerate(core5_features):
    print(f"  {col}: {type(X_target[0, i])}")