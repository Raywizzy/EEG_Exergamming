#!/usr/bin/env python3
"""
Ensemble Models for Phase V Optimization
Random Forest + XGBoost + Voting Classifier
Target: 49% → 65%+ balanced accuracy
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Machine learning imports
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.utils import class_weight
import joblib

try:
    import xgboost as XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost not available, using RandomForest + LogisticRegression ensemble")
    XGBOOST_AVAILABLE = False

# Domain adaptation
from src.domain_adaptation.coral import CORAL

def load_core5_features():
    """Load existing Core5 features from all 3 sites"""

    features_dir = Path("results/features")

    datasets = {
        'ds004584': 'Iowa',
        'ds002778': 'UCSD',
        'ds003490': 'UNM/Iowa'
    }

    all_data = []

    for dataset_id, site_name in datasets.items():
        feature_file = features_dir / f"{dataset_id}_core5.csv"

        if not feature_file.exists():
            print(f"❌ Feature file not found: {feature_file}")
            continue

        df = pd.read_csv(feature_file)
        df['dataset'] = dataset_id
        df['site'] = site_name

        print(f"✅ Loaded {len(df)} subjects from {site_name} ({dataset_id})")
        all_data.append(df)

    if not all_data:
        raise ValueError("No feature files found")

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"📊 Total: {len(combined_df)} subjects from {len(all_data)} sites")

    return combined_df

def prepare_features_and_labels(df):
    """Prepare feature matrix and labels for training"""

    # Feature columns (exclude metadata)
    feature_cols = [col for col in df.columns if col not in [
        'subject_id', 'dataset', 'site', 'group', 'session', 'label'
    ]]

    X = df[feature_cols].values

    # Labels - use 'group' column
    if 'group' in df.columns:
        y = df['group'].values
    else:
        # Try to infer from subject IDs or create dummy labels for testing
        print("⚠️  No 'group' column found, creating dummy labels for testing")
        # Create dummy binary labels (50/50 split)
        n_subjects = len(df)
        y = np.array(['PD'] * (n_subjects // 2) + ['Control'] * (n_subjects - n_subjects // 2))
        np.random.shuffle(y)

    print(f"📊 Features: {X.shape}")
    print(f"📊 Labels: {len(y)} ({np.unique(y, return_counts=True)})")

    return X, y, feature_cols

def create_ensemble_models():
    """Create ensemble of Random Forest + XGBoost + Logistic Regression"""

    # Base models
    models = {}

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    models['RandomForest'] = rf

    # Logistic Regression
    lr = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    )
    models['LogisticRegression'] = lr

    # XGBoost (if available)
    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        models['XGBoost'] = xgb

        # Voting Classifier (all three)
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('lr', lr),
                ('xgb', xgb)
            ],
            voting='soft'
        )
        models['VotingClassifier'] = voting_clf
    else:
        # Voting Classifier (RF + LR only)
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('lr', lr)
            ],
            voting='soft'
        )
        models['VotingClassifier'] = voting_clf

    return models

def evaluate_model_cross_site(model, X, y, sites, model_name, n_folds=5):
    """Evaluate model using cross-validation within sites"""

    print(f"\n🔄 Evaluating {model_name}")
    print("-" * 50)

    # Stratified K-fold within each site to maintain class balance
    cv_scores = []
    cv_predictions = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    try:
        # Cross-validation
        scores = cross_val_score(model, X, y, cv=skf, scoring='balanced_accuracy', n_jobs=-1)
        cv_scores.extend(scores)

        print(f"  Cross-validation scores: {scores}")
        print(f"  Mean CV Balanced Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

        # Fit model for feature importance (if available)
        model.fit(X, y)

        results = {
            'model_name': model_name,
            'cv_scores': scores,
            'mean_ba': scores.mean(),
            'std_ba': scores.std(),
            'fitted_model': model
        }

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            results['feature_importance'] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            results['feature_importance'] = np.abs(model.coef_[0])

        return results

    except Exception as e:
        print(f"  ❌ Error evaluating {model_name}: {str(e)}")
        return None

def perform_loso_validation(models, X, y, sites, feature_cols):
    """Perform Leave-One-Site-Out validation"""

    print("\n" + "=" * 80)
    print("LEAVE-ONE-SITE-OUT (LOSO) VALIDATION")
    print("=" * 80)

    unique_sites = np.unique(sites)

    if len(unique_sites) < 3:
        print(f"⚠️  Only {len(unique_sites)} sites available. LOSO requires 3+ sites.")
        print(f"Available sites: {unique_sites}")
        return None

    loso_results = {}

    for model_name, model in models.items():
        print(f"\n🔍 LOSO Validation: {model_name}")
        print("-" * 50)

        fold_scores = []
        fold_details = []

        for test_site in unique_sites:
            print(f"  Test Site: {test_site}")

            # Split data
            test_mask = sites == test_site
            train_mask = ~test_mask

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            sites_train = sites[train_mask]

            print(f"    Train: {len(X_train)} subjects from {np.unique(sites_train)}")
            print(f"    Test: {len(X_test)} subjects from {test_site}")

            # Harmonize features using CORAL
            harmonizer = CORAL()
            X_train_harm = harmonizer.fit_transform(X_train, X_train)  # Source=target for training
            X_test_harm = harmonizer.transform(X_test)

            # Train model
            try:
                model.fit(X_train_harm, y_train)

                # Predict
                y_pred = model.predict(X_test_harm)
                y_pred_proba = model.predict_proba(X_test_harm)

                # Calculate balanced accuracy
                ba = balanced_accuracy_score(y_test, y_pred)
                fold_scores.append(ba)

                print(f"    Balanced Accuracy: {ba:.3f}")

                fold_details.append({
                    'test_site': test_site,
                    'n_test': len(X_test),
                    'balanced_accuracy': ba,
                    'y_true': y_test,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                })

            except Exception as e:
                print(f"    ❌ Error: {str(e)}")
                continue

        if fold_scores:
            mean_ba = np.mean(fold_scores)
            std_ba = np.std(fold_scores)

            print(f"\n  📊 {model_name} LOSO Results:")
            print(f"    Mean Balanced Accuracy: {mean_ba:.3f} ± {std_ba:.3f}")
            print(f"    Individual fold scores: {fold_scores}")

            loso_results[model_name] = {
                'fold_scores': fold_scores,
                'mean_ba': mean_ba,
                'std_ba': std_ba,
                'fold_details': fold_details
            }

    return loso_results

def main():
    """Main ensemble model evaluation pipeline"""

    print("=" * 80)
    print("PHASE V: ENSEMBLE MODELS (RANDOM FOREST + XGBOOST)")
    print("=" * 80)
    print(f"Target: 49% → 65%+ balanced accuracy")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Load Core5 features
    print("📂 Loading Core5 features...")
    df = load_core5_features()
    print()

    # Prepare features and labels
    print("🔧 Preparing features and labels...")
    X, y, feature_cols = prepare_features_and_labels(df)
    sites = df['site'].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✅ Features standardized")
    print()

    # Create ensemble models
    print("🤖 Creating ensemble models...")
    models = create_ensemble_models()
    print(f"Models created: {list(models.keys())}")
    print()

    # Evaluate models with cross-validation
    print("=" * 80)
    print("CROSS-VALIDATION EVALUATION")
    print("=" * 80)

    cv_results = {}
    for model_name, model in models.items():
        result = evaluate_model_cross_site(model, X_scaled, y, sites, model_name)
        if result:
            cv_results[model_name] = result

    # Compare CV results
    print("\n📊 CROSS-VALIDATION SUMMARY:")
    print("-" * 50)
    for model_name, result in cv_results.items():
        mean_ba = result['mean_ba']
        std_ba = result['std_ba']
        print(f"  {model_name}: {mean_ba:.3f} ± {std_ba:.3f}")

    # Find best CV model
    if cv_results:
        best_cv_model = max(cv_results.keys(), key=lambda k: cv_results[k]['mean_ba'])
        best_cv_score = cv_results[best_cv_model]['mean_ba']
        print(f"\n🏆 Best CV Model: {best_cv_model} ({best_cv_score:.3f})")

    # LOSO validation
    loso_results = perform_loso_validation(models, X_scaled, y, sites, feature_cols)

    if loso_results:
        print("\n📊 LOSO VALIDATION SUMMARY:")
        print("-" * 50)
        for model_name, result in loso_results.items():
            mean_ba = result['mean_ba']
            std_ba = result['std_ba']
            improvement = (mean_ba - 0.49) * 100  # vs baseline 49%
            print(f"  {model_name}: {mean_ba:.3f} ± {std_ba:.3f} (+{improvement:.1f}%)")

        # Find best LOSO model
        best_loso_model = max(loso_results.keys(), key=lambda k: loso_results[k]['mean_ba'])
        best_loso_score = loso_results[best_loso_model]['mean_ba']
        print(f"\n🏆 Best LOSO Model: {best_loso_model} ({best_loso_score:.3f})")

        # Check if target reached
        target_ba = 0.65
        if best_loso_score >= target_ba:
            print(f"🎯 TARGET ACHIEVED! {best_loso_score:.3f} ≥ {target_ba:.3f}")
        else:
            gap = (target_ba - best_loso_score) * 100
            print(f"📈 Target gap: {gap:.1f}% (need {target_ba:.3f}, got {best_loso_score:.3f})")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save models
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    if cv_results:
        for model_name, result in cv_results.items():
            model_file = model_dir / f"ensemble_{model_name.lower()}_{timestamp}.joblib"
            joblib.dump(result['fitted_model'], model_file)
            print(f"💾 Saved model: {model_file}")

    # Save results summary
    results_file = f"results/validation/ensemble_models_results_{timestamp}.json"
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'cv_results': {k: {
            'mean_ba': float(v['mean_ba']),
            'std_ba': float(v['std_ba']),
            'cv_scores': [float(x) for x in v['cv_scores']]
        } for k, v in cv_results.items()},
        'loso_results': {k: {
            'mean_ba': float(v['mean_ba']),
            'std_ba': float(v['std_ba']),
            'fold_scores': [float(x) for x in v['fold_scores']]
        } for k, v in loso_results.items()} if loso_results else {},
        'best_cv_model': best_cv_model if cv_results else None,
        'best_loso_model': best_loso_model if loso_results else None,
        'target_achieved': best_loso_score >= target_ba if loso_results else False
    }

    import json
    Path("results/validation").mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"💾 Results saved: {results_file}")

    print("\n" + "=" * 80)
    print("ENSEMBLE MODELS EVALUATION COMPLETE")
    print("=" * 80)

    if loso_results and best_loso_score >= target_ba:
        print("🎉 SUCCESS: Target 65%+ balanced accuracy achieved!")
    else:
        print("📈 Next steps: Core15+ features + deep learning optimization")

if __name__ == "__main__":
    main()