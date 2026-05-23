"""CORAL (CORrelation ALignment) Domain Adaptation Implementation.

Phase IV Multi-Site EEG Biomarker Validation
Locked implementation for regulatory compliance.
"""

import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class CORAL:
    """CORAL (CORrelation ALignment) domain adaptation.

    Implementation based on:
    Sun, B., Feng, J., & Saenko, K. (2016). Return of frustratingly easy domain adaptation.
    In Proceedings of the AAAI Conference on Artificial Intelligence.

    Locked for Phase IV clinical validation.
    """

    def __init__(self, reg_param: float = 1e-6):
        """Initialize CORAL adapter.

        Args:
            reg_param: Regularization parameter for numerical stability
        """
        self.reg_param = reg_param
        self.source_mean_ = None
        self.target_mean_ = None
        self.transformation_matrix_ = None

        logger.info(f"Initialized CORAL with regularization: {reg_param}")

    def _compute_covariance(self, X: np.ndarray) -> np.ndarray:
        """Compute covariance matrix with regularization.

        Args:
            X: Input data matrix (n_samples, n_features)

        Returns:
            Regularized covariance matrix
        """
        # Center the data
        X_centered = X - np.mean(X, axis=0)

        # Compute covariance
        n_samples = X_centered.shape[0]
        cov = np.dot(X_centered.T, X_centered) / (n_samples - 1)

        # Add regularization for numerical stability
        cov += self.reg_param * np.eye(cov.shape[0])

        return cov

    def _matrix_sqrt(self, A: np.ndarray) -> np.ndarray:
        """Compute matrix square root using eigendecomposition.

        Args:
            A: Symmetric positive definite matrix

        Returns:
            Matrix square root
        """
        # Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(A)

        # Ensure positive eigenvalues (numerical stability)
        eigenvals = np.maximum(eigenvals, self.reg_param)

        # Compute square root
        sqrt_eigenvals = np.sqrt(eigenvals)
        A_sqrt = eigenvecs @ np.diag(sqrt_eigenvals) @ eigenvecs.T

        return A_sqrt

    def _matrix_inv_sqrt(self, A: np.ndarray) -> np.ndarray:
        """Compute matrix inverse square root using eigendecomposition.

        Args:
            A: Symmetric positive definite matrix

        Returns:
            Matrix inverse square root
        """
        # Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(A)

        # Ensure positive eigenvalues (numerical stability)
        eigenvals = np.maximum(eigenvals, self.reg_param)

        # Compute inverse square root
        inv_sqrt_eigenvals = 1.0 / np.sqrt(eigenvals)
        A_inv_sqrt = eigenvecs @ np.diag(inv_sqrt_eigenvals) @ eigenvecs.T

        return A_inv_sqrt

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> 'CORAL':
        """Fit CORAL transformation.

        Args:
            X_source: Source domain data (n_source, n_features)
            X_target: Target domain data (n_target, n_features)

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting CORAL: {X_source.shape[0]} source, {X_target.shape[0]} target samples")

        # Store means for later use
        self.source_mean_ = np.mean(X_source, axis=0)
        self.target_mean_ = np.mean(X_target, axis=0)

        # Compute covariance matrices
        C_source = self._compute_covariance(X_source)
        C_target = self._compute_covariance(X_target)

        # Compute CORAL transformation matrix
        # T = C_source^(-1/2) * C_target^(1/2)
        C_source_inv_sqrt = self._matrix_inv_sqrt(C_source)
        C_target_sqrt = self._matrix_sqrt(C_target)

        self.transformation_matrix_ = C_source_inv_sqrt @ C_target_sqrt

        logger.info("CORAL transformation fitted successfully")
        return self

    def transform_source(self, X_source: np.ndarray) -> np.ndarray:
        """Transform source domain data to target domain.

        Args:
            X_source: Source domain data to transform

        Returns:
            Transformed source domain data
        """
        if self.transformation_matrix_ is None:
            raise ValueError("CORAL not fitted. Call fit() first.")

        # Center source data
        X_centered = X_source - self.source_mean_

        # Apply CORAL transformation
        X_transformed = X_centered @ self.transformation_matrix_.T

        # Add target domain mean
        X_transformed += self.target_mean_

        return X_transformed

    def fit_transform(self, X_source: np.ndarray, X_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit CORAL and transform both domains.

        Args:
            X_source: Source domain data
            X_target: Target domain data

        Returns:
            Tuple of (transformed_source, original_target)
        """
        # Fit CORAL transformation
        self.fit(X_source, X_target)

        # Transform source domain
        X_source_transformed = self.transform_source(X_source)

        # Target domain remains unchanged in standard CORAL
        X_target_transformed = X_target.copy()

        logger.info(f"CORAL adaptation complete: {X_source.shape} -> {X_source_transformed.shape}")

        return X_source_transformed, X_target_transformed

    def get_alignment_loss(self, X_source: np.ndarray, X_target: np.ndarray) -> float:
        """Compute CORAL alignment loss for monitoring.

        Args:
            X_source: Source domain data
            X_target: Target domain data

        Returns:
            CORAL alignment loss (Frobenius norm of covariance difference)
        """
        C_source = self._compute_covariance(X_source)
        C_target = self._compute_covariance(X_target)

        # Frobenius norm of covariance difference
        coral_loss = np.linalg.norm(C_source - C_target, 'fro') ** 2

        return coral_loss