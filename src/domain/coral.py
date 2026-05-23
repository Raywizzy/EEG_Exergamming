"""
CORAL (Correlation Alignment) Domain Adaptation
Implementation for Phase III multi-site validation

Reference: Sun, B., & Saenko, K. (2016). Deep coral: Correlation alignment
for deep domain adaptation. ECCV Workshop.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class CORALTransformer:
    """CORAL domain adaptation transformer for EEG cross-site validation"""

    def __init__(self, reg_param: float = 1e-6):
        """
        Initialize CORAL transformer

        Parameters:
        -----------
        reg_param : float
            Regularization parameter for numerical stability
        """
        self.reg_param = reg_param
        self.source_stats = None
        self.target_stats = None
        self.transform_matrix = None
        self.is_fitted = False

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> 'CORALTransformer':
        """
        Fit CORAL transformation between source and target domains

        Parameters:
        -----------
        X_source : np.ndarray, shape (n_samples_source, n_features)
            Source domain data
        X_target : np.ndarray, shape (n_samples_target, n_features)
            Target domain data

        Returns:
        --------
        self : CORALTransformer
        """
        logger.info(f"Fitting CORAL: source {X_source.shape}, target {X_target.shape}")

        # Compute domain statistics
        self.source_stats = self._compute_domain_stats(X_source, 'source')
        self.target_stats = self._compute_domain_stats(X_target, 'target')

        # Compute transformation matrix
        self.transform_matrix = self._compute_transform_matrix()

        self.is_fitted = True
        logger.info("CORAL transformation fitted successfully")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply CORAL transformation to source domain data

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Source domain data to transform

        Returns:
        --------
        X_transformed : np.ndarray, shape (n_samples, n_features)
            CORAL-aligned data
        """
        if not self.is_fitted:
            raise ValueError("CORAL transformer must be fitted before transform")

        # Center data using source statistics
        X_centered = X - self.source_stats['mean']

        # Apply CORAL transformation
        X_transformed = X_centered @ self.transform_matrix

        # Shift to target domain mean
        X_transformed += self.target_stats['mean']

        return X_transformed

    def fit_transform(self, X_source: np.ndarray, X_target: np.ndarray) -> np.ndarray:
        """
        Fit CORAL and transform source data in one step

        Parameters:
        -----------
        X_source : np.ndarray
            Source domain data
        X_target : np.ndarray
            Target domain data

        Returns:
        --------
        X_transformed : np.ndarray
            CORAL-aligned source data
        """
        return self.fit(X_source, X_target).transform(X_source)

    def _compute_domain_stats(self, X: np.ndarray, domain_name: str) -> Dict:
        """Compute domain statistics (mean, covariance)"""
        mean = np.mean(X, axis=0, keepdims=True)
        X_centered = X - mean

        # Compute covariance with regularization
        cov = np.cov(X_centered.T) + self.reg_param * np.eye(X.shape[1])

        # Numerical stability check
        try:
            # Test matrix decomposition
            eigenvals, _ = np.linalg.eigh(cov)
            min_eigenval = np.min(eigenvals)

            if min_eigenval <= 0:
                logger.warning(f"{domain_name} covariance not positive definite, "
                             f"min eigenvalue: {min_eigenval:.2e}")
                # Add more regularization
                cov += (abs(min_eigenval) + 1e-6) * np.eye(X.shape[1])

        except np.linalg.LinAlgError:
            logger.error(f"Failed to decompose {domain_name} covariance matrix")
            raise

        stats = {
            'mean': mean,
            'cov': cov,
            'n_samples': X.shape[0],
            'n_features': X.shape[1]
        }

        logger.debug(f"{domain_name} stats: mean shape {mean.shape}, "
                    f"cov shape {cov.shape}, condition number: {np.linalg.cond(cov):.2e}")

        return stats

    def _compute_transform_matrix(self) -> np.ndarray:
        """
        Compute CORAL transformation matrix using eigendecomposition

        Transform = C_source^(-1/2) @ C_target^(1/2)
        """
        C_source = self.source_stats['cov']
        C_target = self.target_stats['cov']

        try:
            # Source: C_source^(-1/2)
            w_source, v_source = np.linalg.eigh(C_source)
            w_source = np.maximum(w_source, 1e-8)  # Numerical stability
            C_source_neg_sqrt = v_source @ np.diag(w_source**(-0.5)) @ v_source.T

            # Target: C_target^(1/2)
            w_target, v_target = np.linalg.eigh(C_target)
            w_target = np.maximum(w_target, 1e-8)  # Numerical stability
            C_target_sqrt = v_target @ np.diag(w_target**0.5) @ v_target.T

            # Transformation matrix
            A = C_source_neg_sqrt @ C_target_sqrt

            # Verify transformation is reasonable
            transform_norm = np.linalg.norm(A)
            if transform_norm > 100 or transform_norm < 0.01:
                logger.warning(f"CORAL transform matrix norm unusual: {transform_norm:.2e}")

            logger.debug(f"Transform matrix computed: norm={transform_norm:.3f}, "
                        f"condition={np.linalg.cond(A):.2e}")

            return A

        except np.linalg.LinAlgError as e:
            logger.error(f"Failed to compute CORAL transformation: {e}")
            logger.info("Falling back to identity transformation")
            return np.eye(C_source.shape[0])

    def get_alignment_metrics(self) -> Dict:
        """
        Compute metrics to assess alignment quality

        Returns:
        --------
        metrics : dict
            Alignment quality metrics
        """
        if not self.is_fitted:
            raise ValueError("CORAL transformer must be fitted first")

        C_source = self.source_stats['cov']
        C_target = self.target_stats['cov']

        # Frobenius norm of covariance difference
        cov_diff_norm = np.linalg.norm(C_source - C_target, 'fro')

        # Maximum eigenvalue ratio
        w_source = np.linalg.eigvals(C_source)
        w_target = np.linalg.eigvals(C_target)

        eigenval_ratio = np.max(w_source) / np.max(w_target)

        # Condition numbers
        cond_source = np.linalg.cond(C_source)
        cond_target = np.linalg.cond(C_target)

        metrics = {
            'covariance_diff_norm': float(cov_diff_norm),
            'eigenvalue_ratio': float(eigenval_ratio),
            'condition_source': float(cond_source),
            'condition_target': float(cond_target),
            'transform_norm': float(np.linalg.norm(self.transform_matrix)),
            'regularization': self.reg_param
        }

        return metrics

def coral_transform(X_source: np.ndarray, X_target: np.ndarray,
                   reg_param: float = 1e-6) -> Tuple[np.ndarray, CORALTransformer]:
    """
    Convenience function for CORAL domain adaptation

    Parameters:
    -----------
    X_source : np.ndarray
        Source domain data
    X_target : np.ndarray
        Target domain data
    reg_param : float
        Regularization parameter

    Returns:
    --------
    X_transformed : np.ndarray
        CORAL-aligned source data
    transformer : CORALTransformer
        Fitted transformer for future use
    """
    transformer = CORALTransformer(reg_param=reg_param)
    X_transformed = transformer.fit_transform(X_source, X_target)

    return X_transformed, transformer