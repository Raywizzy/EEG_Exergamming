"""
Grad-CAM Implementation for CNN Spectrogram Analysis
Provides per-subject, per-prediction visualization of CNN attention on EEG spectrograms
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class GradCAMConfig:
    """Configuration for Grad-CAM analysis"""
    target_layer: str = "conv_layers.4"  # Last conv layer before GAP
    guided_backprop: bool = True
    smooth_grad: bool = True
    smooth_grad_samples: int = 50
    smooth_grad_noise: float = 0.15
    alpha: float = 0.5  # Overlay transparency
    colormap: str = 'jet'
    save_intermediate: bool = True

class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNNs"""

    def __init__(self, model: torch.nn.Module, target_layer: str, config: GradCAMConfig):
        self.model = model
        self.target_layer = target_layer
        self.config = config
        self.gradients = None
        self.activations = None
        self.hooks = []

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on target layer"""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Find target layer
        target_module = self._find_target_layer()
        if target_module is None:
            raise ValueError(f"Target layer '{self.target_layer}' not found in model")

        # Register hooks
        self.hooks.append(target_module.register_forward_hook(forward_hook))
        self.hooks.append(target_module.register_full_backward_hook(backward_hook))

    def _find_target_layer(self):
        """Find target layer in model"""
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                return module
        return None

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """Generate Class Activation Map"""
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)

        # Use predicted class if target not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()

        # Generate CAM
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy()

    def generate_smooth_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """Generate SmoothGrad-CAM for reduced noise"""
        if not self.config.smooth_grad:
            return self.generate_cam(input_tensor, target_class)

        input_tensor = input_tensor.clone()
        smooth_cam = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        for _ in range(self.config.smooth_grad_samples):
            # Add noise
            noise = torch.randn_like(input_tensor) * self.config.smooth_grad_noise
            noisy_input = input_tensor + noise

            # Generate CAM
            cam = self.generate_cam(noisy_input, target_class)
            smooth_cam += cam

        return smooth_cam / self.config.smooth_grad_samples

    def overlay_cam_on_spectrogram(self, spectrogram: np.ndarray, cam: np.ndarray) -> np.ndarray:
        """Overlay CAM on original spectrogram"""
        # Resize CAM to match spectrogram
        cam_resized = cv2.resize(cam, (spectrogram.shape[1], spectrogram.shape[0]))

        # Apply colormap
        cam_colored = plt.cm.get_cmap(self.config.colormap)(cam_resized)[:, :, :3]

        # Normalize spectrogram
        spec_norm = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        spec_colored = plt.cm.gray(spec_norm)[:, :, :3]

        # Blend
        overlay = self.config.alpha * cam_colored + (1 - self.config.alpha) * spec_colored

        return overlay

    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

class CNNExplainer:
    """High-level CNN explainability interface"""

    def __init__(self, model_path: str, config: GradCAMConfig = None):
        self.model_path = model_path
        self.config = config or GradCAMConfig()
        self.model = None
        self.grad_cam = None

    def load_model(self):
        """Load CNN model"""
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)

        # Import model class
        from ..models.deep_learning.cnn_classifier import EEGCNNClassifier, CNNConfig

        # Reconstruct model
        cnn_config = CNNConfig()
        self.model = EEGCNNClassifier(cnn_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Initialize Grad-CAM
        self.grad_cam = GradCAM(self.model, self.config.target_layer, self.config)

    def explain_prediction(self, eeg_data: np.ndarray, subject_id: str) -> Dict[str, Any]:
        """Generate comprehensive explanation for EEG prediction"""
        if self.model is None:
            self.load_model()

        # Convert EEG to spectrogram
        from ..models.deep_learning.cnn_classifier import EEGCNNClassifier
        spectrogram = EEGCNNClassifier.eeg_to_spectrogram(eeg_data)

        # Convert to tensor
        input_tensor = torch.from_numpy(spectrogram).unsqueeze(0).float()

        # Forward pass for prediction
        with torch.no_grad():
            prediction = self.model(input_tensor)
            probabilities = F.softmax(prediction, dim=1)
            predicted_class = prediction.argmax(dim=1).item()
            confidence = probabilities.max().item()

        # Generate CAM
        cam = self.grad_cam.generate_smooth_cam(input_tensor, predicted_class)

        # Create overlay
        overlay = self.grad_cam.overlay_cam_on_spectrogram(spectrogram[0, 0], cam)

        # Calculate attribution statistics
        attribution_stats = self._calculate_attribution_stats(cam, spectrogram[0, 0])

        # Generate data hash for audit
        data_hash = hashlib.sha256(eeg_data.tobytes()).hexdigest()[:16]

        explanation = {
            'subject_id': subject_id,
            'prediction': {
                'class': predicted_class,
                'confidence': float(confidence),
                'probabilities': probabilities.cpu().numpy().tolist()
            },
            'attribution': {
                'cam': cam,
                'overlay': overlay,
                'statistics': attribution_stats
            },
            'metadata': {
                'model_path': self.model_path,
                'target_layer': self.config.target_layer,
                'data_hash': data_hash,
                'smooth_grad': self.config.smooth_grad,
                'samples': self.config.smooth_grad_samples if self.config.smooth_grad else 1
            }
        }

        return explanation

    def _calculate_attribution_stats(self, cam: np.ndarray, spectrogram: np.ndarray) -> Dict[str, float]:
        """Calculate attribution statistics"""
        # Find peak attribution regions
        cam_flat = cam.flatten()
        cam_sorted = np.sort(cam_flat)[::-1]
        top_10_percent_threshold = cam_sorted[int(0.1 * len(cam_sorted))]

        # Frequency and time analysis
        freq_attribution = np.mean(cam, axis=1)  # Average across time
        time_attribution = np.mean(cam, axis=0)  # Average across frequency

        # Peak frequency and time
        peak_freq_idx = np.argmax(freq_attribution)
        peak_time_idx = np.argmax(time_attribution)

        stats = {
            'peak_attribution': float(cam.max()),
            'mean_attribution': float(cam.mean()),
            'attribution_std': float(cam.std()),
            'top_10_percent_threshold': float(top_10_percent_threshold),
            'peak_frequency_bin': int(peak_freq_idx),
            'peak_time_bin': int(peak_time_idx),
            'frequency_focus': float(freq_attribution.max() / freq_attribution.mean()),
            'temporal_focus': float(time_attribution.max() / time_attribution.mean())
        }

        return stats

    def batch_explain(self, eeg_batch: List[Tuple[np.ndarray, str]]) -> List[Dict[str, Any]]:
        """Generate explanations for batch of EEG data"""
        explanations = []

        for eeg_data, subject_id in eeg_batch:
            try:
                explanation = self.explain_prediction(eeg_data, subject_id)
                explanations.append(explanation)
            except Exception as e:
                logger.error(f"Failed to explain prediction for {subject_id}: {e}")
                explanations.append({
                    'subject_id': subject_id,
                    'error': str(e)
                })

        return explanations

    def cleanup(self):
        """Cleanup resources"""
        if self.grad_cam:
            self.grad_cam.cleanup()