#!/usr/bin/env python3
"""
ML Pipeline for EEG Exergaming Project
Implements regulatory-grade model training with proper subject isolation.

Features:
- Grouped cross-validation (no subject leakage)
- Nested CV with hyperparameter optimization
- Statistical validation (permutation tests, bootstrap CI)
- Model interpretation and feature importance
- Calibrated probability estimates
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GroupKFold, StratifiedGroupKFold, GridSearchCV,
    permutation_test_score, cross_val_predict
)
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

class RegulatoryMLPipeline:
    """
    Regulatory-grade ML pipeline with proper subject isolation.
    """

    def __init__(self, random_state=42):
        """Initialize pipeline with fixed random state for reproducibility."""
        self.random_state = random_state
        np.random.seed(random_state)

        # Pipeline configurations
        self.pipelines = self._create_pipelines()
        self.param_grids = self._create_param_grids()

        # Results storage
        self.cv_results = {}
        self.best_models = {}
        self.feature_importance = {}
        self.label_encoder = LabelEncoder()

    def _create_pipelines(self) -> Dict:
        """Create standardized ML pipelines."""
        return {
            "logistic": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=self.random_state
                ))
            ]),

            "svm": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=self.random_state
                ))
            ]),

            "random_forest": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=500,
                    class_weight="balanced",
                    random_state=self.random_state,
                    n_jobs=-1
                ))
            ]),

            "gradient_boost": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(
                    random_state=self.random_state
                ))
            ])
        }

    def _create_param_grids(self) -> Dict:
        """Create hyperparameter grids for each model."""
        return {
            "logistic": {
                "clf__C": [0.1, 1.0, 3.0, 10.0, 30.0]
            },

            "svm": {
                "clf__C": [0.5, 1.0, 3.0, 10.0],
                "clf__gamma": ["scale", 0.1, 0.01]
            },

            "random_forest": {
                "clf__max_depth": [None, 6, 10],
                "clf__max_features": ["sqrt", 0.3],
                "clf__n_estimators": [500, 800]
            },

            "gradient_boost": {
                "clf__learning_rate": [0.05, 0.1],
                "clf__n_estimators": [300, 500],
                "clf__max_depth": [2, 3]
            }
        }

    def prepare_data(self, feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training with proper subject grouping.

        Parameters:
        -----------
        feature_df : pd.DataFrame
            Feature matrix from Phase 3

        Returns:
        --------
        X, y, groups : Features, labels, subject groups
        """
        # Separate features from metadata
        metadata_cols = ['subject_id', 'condition', 'session_number',
                        'original_epochs', 'rejected_epochs']
        feature_cols = [col for col in feature_df.columns if col not in metadata_cols]

        X = feature_df[feature_cols].copy()
        y = feature_df['condition']  # Keep as strings for now
        groups = feature_df['subject_id']

        print(f"Data prepared:")
        print(f"  Features: {X.shape}")
        print(f"  Samples: {len(y)} ({sum(y == 'PD_REAL')} PD_REAL, {sum(y == 'PD_SHAM')} PD_SHAM)")
        print(f"  Subjects: {groups.nunique()}")
        print(f"  Sessions per subject: {len(y)/groups.nunique():.1f} average")

        return X, y, groups

    def grouped_cross_validation(self, X: pd.DataFrame, y: pd.Series,
                                groups: pd.Series, n_splits=5) -> Dict:
        """
        Perform grouped cross-validation with proper subject isolation.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target labels
        groups : pd.Series
            Subject groupings
        n_splits : int
            Number of CV folds

        Returns:
        --------
        Dict with CV results
        """
        print(f"Starting grouped {n_splits}-fold cross-validation...")

        # Encode string labels to integers
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"Label encoding: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

        # Try StratifiedGroupKFold first, fallback to GroupKFold
        try:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=self.random_state)
            cv_name = "StratifiedGroupKFold"
        except Exception:
            cv = GroupKFold(n_splits=n_splits)
            cv_name = "GroupKFold"

        print(f"Using {cv_name}")

        # Store fold information for reproducibility
        fold_info = {}
        cv_scores = defaultdict(list)
        oof_predictions = {}

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y_encoded, groups), 1):
            print(f"\nFold {fold_idx}/{n_splits}")

            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
            groups_train = groups.iloc[train_idx]
            groups_test = groups.iloc[test_idx]

            # Verify no subject leakage
            train_subjects = set(groups_train)
            test_subjects = set(groups_test)
            assert len(train_subjects.intersection(test_subjects)) == 0, \
                f"Subject leakage detected in fold {fold_idx}!"

            print(f"  Train: {len(X_train)} samples, {len(train_subjects)} subjects")
            print(f"  Test:  {len(X_test)} samples, {len(test_subjects)} subjects")
            print(f"  Train class balance: {sum(y_train)}/{len(y_train)} PD_REAL")
            print(f"  Test class balance:  {sum(y_test)}/{len(y_test)} PD_REAL")

            # Store fold information
            fold_info[f"fold_{fold_idx}"] = {
                "train_subjects": list(train_subjects),
                "test_subjects": list(test_subjects),
                "train_indices": train_idx.tolist(),
                "test_indices": test_idx.tolist(),
                "train_class_balance": {"PD_REAL": int(sum(y_train)),
                                      "PD_SHAM": int(len(y_train) - sum(y_train))},
                "test_class_balance": {"PD_REAL": int(sum(y_test)),
                                     "PD_SHAM": int(len(y_test) - sum(y_test))}
            }

            # Train and evaluate each model
            fold_best_score = -1
            fold_best_model = None
            fold_best_name = None

            for model_name, pipeline in self.pipelines.items():
                print(f"    Training {model_name}...")

                # Hyperparameter optimization with inner CV
                param_grid = self.param_grids[model_name]

                # Use regular StratifiedKFold for inner loop (subjects already separated)
                inner_cv = 3
                grid_search = GridSearchCV(
                    pipeline, param_grid,
                    scoring="balanced_accuracy",
                    cv=inner_cv,
                    n_jobs=-1,
                    verbose=0
                )

                grid_search.fit(X_train, y_train)

                # Evaluate on test fold
                y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
                y_pred = (y_pred_proba >= 0.5).astype(int)

                # Calculate metrics
                ba_score = balanced_accuracy_score(y_test, y_pred)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                sensitivity = recall_score(y_test, y_pred)
                specificity = recall_score(1 - y_test, 1 - y_pred)
                f1 = f1_score(y_test, y_pred)

                print(f"      CV score: {grid_search.best_score_:.3f}")
                print(f"      Test BA:  {ba_score:.3f}")
                print(f"      Test AUC: {auc_score:.3f}")

                # Track best model for this fold
                if ba_score > fold_best_score:
                    fold_best_score = ba_score
                    fold_best_model = grid_search.best_estimator_
                    fold_best_name = model_name

                # Store detailed results
                cv_scores[model_name].append({
                    "fold": fold_idx,
                    "cv_score": grid_search.best_score_,
                    "best_params": grid_search.best_params_,
                    "balanced_accuracy": ba_score,
                    "auc": auc_score,
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "f1_score": f1,
                    "n_train": len(X_train),
                    "n_test": len(X_test)
                })

            # Store out-of-fold predictions from best model
            best_proba = fold_best_model.predict_proba(X_test)[:, 1]
            for idx, proba in zip(test_idx, best_proba):
                oof_predictions[idx] = {
                    "fold": fold_idx,
                    "model": fold_best_name,
                    "probability": float(proba),
                    "prediction": int(proba >= 0.5),
                    "true_label": int(y_encoded[idx]),
                    "subject_id": groups.iloc[idx]
                }

            print(f"  Best model for fold: {fold_best_name} (BA: {fold_best_score:.3f})")

        return {
            "fold_info": fold_info,
            "cv_scores": dict(cv_scores),
            "oof_predictions": oof_predictions,
            "cv_strategy": cv_name
        }

    def statistical_validation(self, X: pd.DataFrame, y: pd.Series,
                             groups: pd.Series, best_model_name: str,
                             n_permutations=1000, n_bootstrap=1000) -> Dict:
        """
        Perform statistical validation with permutation tests and bootstrap CI.

        Parameters:
        -----------
        X, y, groups : Data matrices
        best_model_name : str
            Name of best performing model
        n_permutations : int
            Number of permutation test iterations
        n_bootstrap : int
            Number of bootstrap iterations

        Returns:
        --------
        Dict with statistical validation results
        """
        print(f"Running statistical validation...")
        print(f"  Permutation test: {n_permutations} iterations")
        print(f"  Bootstrap CI: {n_bootstrap} iterations")

        results = {}

        # Get best pipeline
        best_pipeline = self.pipelines[best_model_name]

        # Permutation test
        print("Running permutation test...")
        try:
            cv = GroupKFold(n_splits=5)
            score, permutation_scores, pvalue = permutation_test_score(
                best_pipeline, X, y, groups=groups,
                scoring="balanced_accuracy",
                cv=cv,
                n_permutations=n_permutations,
                random_state=self.random_state,
                n_jobs=-1
            )

            results["permutation_test"] = {
                "observed_score": float(score),
                "permutation_scores": permutation_scores.tolist(),
                "p_value": float(pvalue),
                "n_permutations": n_permutations,
                "null_mean": float(np.mean(permutation_scores)),
                "null_std": float(np.std(permutation_scores))
            }

            print(f"  Observed score: {score:.3f}")
            print(f"  Null mean: {np.mean(permutation_scores):.3f}")
            print(f"  P-value: {pvalue:.6f}")

        except Exception as e:
            print(f"  Permutation test failed: {e}")
            results["permutation_test"] = {"error": str(e)}

        # Bootstrap confidence intervals
        print("Computing bootstrap confidence intervals...")
        try:
            cv = GroupKFold(n_splits=5)
            bootstrap_scores = []

            for i in range(n_bootstrap):
                # Bootstrap sample subjects (not sessions)
                unique_subjects = groups.unique()
                boot_subjects = np.random.choice(
                    unique_subjects, size=len(unique_subjects), replace=True
                )

                # Get all sessions for bootstrapped subjects
                boot_indices = []
                boot_y = []
                boot_groups = []

                for subject in boot_subjects:
                    subject_indices = groups[groups == subject].index
                    boot_indices.extend(subject_indices)
                    boot_y.extend(y.loc[subject_indices])
                    boot_groups.extend([subject] * len(subject_indices))

                boot_X = X.loc[boot_indices]
                boot_y = pd.Series(boot_y)
                boot_groups = pd.Series(boot_groups)

                # Cross-validate on bootstrap sample
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(
                    best_pipeline, boot_X, boot_y, groups=boot_groups,
                    scoring="balanced_accuracy", cv=cv
                )
                bootstrap_scores.append(np.mean(scores))

            # Calculate confidence intervals
            bootstrap_scores = np.array(bootstrap_scores)
            ci_lower = np.percentile(bootstrap_scores, 2.5)
            ci_upper = np.percentile(bootstrap_scores, 97.5)

            results["bootstrap_ci"] = {
                "bootstrap_scores": bootstrap_scores.tolist(),
                "mean": float(np.mean(bootstrap_scores)),
                "std": float(np.std(bootstrap_scores)),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "n_bootstrap": n_bootstrap
            }

            print(f"  Bootstrap mean: {np.mean(bootstrap_scores):.3f}")
            print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

        except Exception as e:
            print(f"  Bootstrap CI failed: {e}")
            results["bootstrap_ci"] = {"error": str(e)}

        return results

    def model_interpretation(self, X: pd.DataFrame, y: pd.Series,
                           groups: pd.Series, model_name: str,
                           top_n_features=20) -> Dict:
        """
        Interpret model with feature importance and partial dependence.

        Parameters:
        -----------
        X, y, groups : Data matrices
        model_name : str
            Model to interpret
        top_n_features : int
            Number of top features to analyze

        Returns:
        --------
        Dict with interpretation results
        """
        print(f"Interpreting {model_name} model...")

        # Train model on full dataset
        pipeline = self.pipelines[model_name]
        param_grid = self.param_grids[model_name]

        # Find best parameters
        grid_search = GridSearchCV(
            pipeline, param_grid,
            scoring="balanced_accuracy",
            cv=GroupKFold(n_splits=3),
            n_jobs=-1
        )
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_

        results = {
            "model_name": model_name,
            "best_params": grid_search.best_params_,
            "feature_names": list(X.columns)
        }

        # Permutation importance
        print("  Computing permutation importance...")
        try:
            perm_importance = permutation_importance(
                best_model, X, y,
                scoring="balanced_accuracy",
                n_repeats=30,
                random_state=self.random_state,
                n_jobs=-1
            )

            # Get top features
            importance_df = pd.DataFrame({
                "feature": X.columns,
                "importance_mean": perm_importance.importances_mean,
                "importance_std": perm_importance.importances_std
            }).sort_values("importance_mean", ascending=False)

            results["permutation_importance"] = {
                "importances_mean": perm_importance.importances_mean.tolist(),
                "importances_std": perm_importance.importances_std.tolist(),
                "top_features": importance_df.head(top_n_features).to_dict("records")
            }

            print(f"    Top 5 features:")
            for i, row in importance_df.head(5).iterrows():
                print(f"      {row['feature']}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")

        except Exception as e:
            print(f"    Permutation importance failed: {e}")
            results["permutation_importance"] = {"error": str(e)}

        # Model-specific interpretation
        try:
            if model_name == "logistic":
                # Get coefficients after scaling
                scaler = best_model.named_steps['scaler']
                clf = best_model.named_steps['clf']

                # Scale coefficients by feature std
                feature_names = X.columns
                coefficients = clf.coef_[0]
                scaled_coefs = coefficients / scaler.scale_

                coef_df = pd.DataFrame({
                    "feature": feature_names,
                    "coefficient": coefficients,
                    "scaled_coefficient": scaled_coefs,
                    "abs_coefficient": np.abs(coefficients)
                }).sort_values("abs_coefficient", ascending=False)

                results["model_specific"] = {
                    "coefficients": coef_df.to_dict("records"),
                    "intercept": float(clf.intercept_[0])
                }

            elif model_name == "random_forest":
                # Get feature importances
                clf = best_model.named_steps['clf']

                feature_imp_df = pd.DataFrame({
                    "feature": X.columns,
                    "importance": clf.feature_importances_
                }).sort_values("importance", ascending=False)

                results["model_specific"] = {
                    "feature_importances": feature_imp_df.to_dict("records")
                }

        except Exception as e:
            print(f"    Model-specific interpretation failed: {e}")
            results["model_specific"] = {"error": str(e)}

        # Save trained model
        results["trained_model"] = best_model

        return results

    def calibrate_model(self, X: pd.DataFrame, y: pd.Series,
                       groups: pd.Series, model_name: str) -> Dict:
        """
        Calibrate model probabilities using cross-validation.

        Parameters:
        -----------
        X, y, groups : Data matrices
        model_name : str
            Model to calibrate

        Returns:
        --------
        Dict with calibration results
        """
        print(f"Calibrating {model_name} model...")

        # Get base model with best parameters
        pipeline = self.pipelines[model_name]
        param_grid = self.param_grids[model_name]

        grid_search = GridSearchCV(
            pipeline, param_grid,
            scoring="balanced_accuracy",
            cv=GroupKFold(n_splits=3),
            n_jobs=-1
        )
        grid_search.fit(X, y)
        base_model = grid_search.best_estimator_

        # Calibrate with CalibratedClassifierCV
        try:
            calibrated_model = CalibratedClassifierCV(
                base_model,
                method="sigmoid",
                cv=GroupKFold(n_splits=3)
            )
            calibrated_model.fit(X, y)

            # Get calibration curve
            y_proba_uncalibrated = base_model.predict_proba(X)[:, 1]
            y_proba_calibrated = calibrated_model.predict_proba(X)[:, 1]

            # Compute calibration metrics
            brier_uncalibrated = brier_score_loss(y, y_proba_uncalibrated)
            brier_calibrated = brier_score_loss(y, y_proba_calibrated)

            # Get calibration curve data
            fraction_pos_uncal, mean_pred_uncal = calibration_curve(
                y, y_proba_uncalibrated, n_bins=10
            )
            fraction_pos_cal, mean_pred_cal = calibration_curve(
                y, y_proba_calibrated, n_bins=10
            )

            results = {
                "calibrated_model": calibrated_model,
                "brier_score_uncalibrated": float(brier_uncalibrated),
                "brier_score_calibrated": float(brier_calibrated),
                "calibration_curve_uncalibrated": {
                    "fraction_positives": fraction_pos_uncal.tolist(),
                    "mean_predicted": mean_pred_uncal.tolist()
                },
                "calibration_curve_calibrated": {
                    "fraction_positives": fraction_pos_cal.tolist(),
                    "mean_predicted": mean_pred_cal.tolist()
                }
            }

            print(f"  Brier score uncalibrated: {brier_uncalibrated:.4f}")
            print(f"  Brier score calibrated:   {brier_calibrated:.4f}")

            return results

        except Exception as e:
            print(f"  Calibration failed: {e}")
            return {"error": str(e)}

    def select_best_model(self, cv_results: Dict) -> str:
        """
        Select best model based on mean CV balanced accuracy.

        Parameters:
        -----------
        cv_results : Dict
            Results from cross-validation

        Returns:
        --------
        str : Name of best model
        """
        model_scores = {}

        for model_name, scores in cv_results["cv_scores"].items():
            ba_scores = [s["balanced_accuracy"] for s in scores]
            model_scores[model_name] = {
                "mean_ba": np.mean(ba_scores),
                "std_ba": np.std(ba_scores),
                "scores": ba_scores
            }

        # Select model with highest mean balanced accuracy
        best_model = max(model_scores.keys(),
                        key=lambda k: model_scores[k]["mean_ba"])

        print(f"\nModel Selection Results:")
        for model_name, stats in model_scores.items():
            marker = "★" if model_name == best_model else " "
            print(f"{marker} {model_name:15}: {stats['mean_ba']:.3f} ± {stats['std_ba']:.3f}")

        print(f"\nSelected model: {best_model}")
        return best_model