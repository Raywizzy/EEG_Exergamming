"""
Transformer Attention Analysis with Rollout and Layer-wise Relevance Propagation (LRP)
Provides token/patch attribution for transformer-based EEG classification
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from dataclasses import dataclass
import hashlib
from scipy.stats import entropy

logger = logging.getLogger(__name__)

@dataclass
class AttentionAnalysisConfig:
    """Configuration for attention analysis"""
    use_rollout: bool = True
    use_lrp: bool = False  # Optional advanced attribution
    attention_threshold: float = 0.1
    head_fusion: str = "mean"  # "mean", "max", "min"
    layer_fusion: str = "last"  # "last", "mean", "rollout"
    patch_size: int = 16  # For vision transformer patches
    save_attention_maps: bool = True
    normalize_attention: bool = True

class AttentionRollout:
    """Attention Rollout for transformer interpretation"""

    def __init__(self, model: torch.nn.Module, config: AttentionAnalysisConfig):
        self.model = model
        self.config = config
        self.attention_maps = []

    def register_hooks(self):
        """Register hooks to capture attention maps"""
        def attention_hook(module, input, output):
            # Capture attention weights
            if hasattr(module, 'attention') and hasattr(module.attention, 'attention_weights'):
                self.attention_maps.append(module.attention.attention_weights.detach())

        hooks = []
        for module in self.model.modules():
            if hasattr(module, 'attention'):
                hooks.append(module.register_forward_hook(attention_hook))

        return hooks

    def compute_rollout(self, attention_maps: List[torch.Tensor]) -> torch.Tensor:
        """Compute attention rollout across layers"""
        if not attention_maps:
            raise ValueError("No attention maps found")

        # Start with identity matrix
        batch_size, num_heads, seq_len, _ = attention_maps[0].shape
        rollout = torch.eye(seq_len).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_heads, 1, 1)

        # Rollout through layers
        for attention in attention_maps:
            # Add residual connection (identity matrix with small weight)
            attention_with_residual = attention + 0.1 * torch.eye(seq_len).to(attention.device)

            # Normalize
            attention_with_residual = attention_with_residual / attention_with_residual.sum(dim=-1, keepdim=True)

            # Matrix multiplication for rollout
            rollout = torch.matmul(attention_with_residual, rollout)

        return rollout

    def fuse_heads(self, attention: torch.Tensor) -> torch.Tensor:
        """Fuse multi-head attention"""
        if self.config.head_fusion == "mean":
            return attention.mean(dim=1)
        elif self.config.head_fusion == "max":
            return attention.max(dim=1)[0]
        elif self.config.head_fusion == "min":
            return attention.min(dim=1)[0]
        else:
            return attention.mean(dim=1)

    def analyze_attention(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze attention patterns"""
        self.attention_maps = []
        hooks = self.register_hooks()

        try:
            # Forward pass
            with torch.no_grad():
                output = self.model(input_tensor)

            if not self.attention_maps:
                raise ValueError("No attention maps captured. Check model architecture.")

            # Compute rollout
            rollout = self.compute_rollout(self.attention_maps)

            # Fuse heads
            rollout_fused = self.fuse_heads(rollout)

            # Extract CLS token attention (first token)
            cls_attention = rollout_fused[:, 0, 1:]  # Exclude CLS to CLS

            results = {
                'rollout_attention': rollout_fused,
                'cls_attention': cls_attention,
                'raw_attention_maps': self.attention_maps,
                'prediction': output
            }

            return results

        finally:
            # Cleanup hooks
            for hook in hooks:
                hook.remove()

class LayerwiseRelevancePropagation:
    """Layer-wise Relevance Propagation for transformers (optional advanced method)"""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.relevance_scores = {}

    def compute_lrp(self, input_tensor: torch.Tensor, target_class: int = None) -> torch.Tensor:
        """Compute LRP relevance scores"""
        # This is a simplified LRP implementation
        # Full LRP for transformers requires more complex propagation rules

        # Enable gradients
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1)

        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()

        # Use input gradients as proxy for relevance
        relevance = input_tensor.grad.abs()

        return relevance

class TransformerAttentionAnalyzer:
    """High-level transformer attention analysis interface"""

    def __init__(self, model_path: str, config: AttentionAnalysisConfig = None):
        self.model_path = model_path
        self.config = config or AttentionAnalysisConfig()
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load transformer model"""
        # Note: This would need to be adapted for your specific transformer implementation
        # For now, creating a placeholder structure
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)

        # This would load your actual transformer model
        # self.model = YourTransformerModel(config)
        # self.model.load_state_dict(checkpoint['model_state_dict'])

        # Placeholder for demo
        logger.warning("Transformer model loading not implemented - using placeholder")
        self.model = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3)  # 3 classes
        )

    def explain_prediction(self, eeg_data: np.ndarray, subject_id: str) -> Dict[str, Any]:
        """Generate comprehensive attention explanation"""
        if self.model is None:
            self.load_model()

        # Convert EEG to transformer input format
        input_tensor = self._prepare_transformer_input(eeg_data)

        # Analyze attention
        if hasattr(self.model, 'attention') and self.config.use_rollout:
            rollout_analyzer = AttentionRollout(self.model, self.config)
            attention_results = rollout_analyzer.analyze_attention(input_tensor)
        else:
            # Fallback for models without explicit attention
            attention_results = self._compute_gradient_attention(input_tensor)

        # Compute prediction
        with torch.no_grad():
            prediction = self.model(input_tensor)
            probabilities = F.softmax(prediction, dim=1)
            predicted_class = prediction.argmax(dim=1).item()
            confidence = probabilities.max().item()

        # Optional LRP
        lrp_scores = None
        if self.config.use_lrp:
            lrp_analyzer = LayerwiseRelevancePropagation(self.model)
            lrp_scores = lrp_analyzer.compute_lrp(input_tensor, predicted_class)

        # Analyze attention patterns
        attention_stats = self._calculate_attention_statistics(attention_results)

        # Generate data hash
        data_hash = hashlib.sha256(eeg_data.tobytes()).hexdigest()[:16]

        explanation = {
            'subject_id': subject_id,
            'prediction': {
                'class': predicted_class,
                'confidence': float(confidence),
                'probabilities': probabilities.cpu().numpy().tolist()
            },
            'attention': {
                'rollout_attention': attention_results.get('rollout_attention'),
                'cls_attention': attention_results.get('cls_attention'),
                'statistics': attention_stats,
                'lrp_scores': lrp_scores
            },
            'metadata': {
                'model_path': self.model_path,
                'use_rollout': self.config.use_rollout,
                'use_lrp': self.config.use_lrp,
                'data_hash': data_hash,
                'head_fusion': self.config.head_fusion,
                'layer_fusion': self.config.layer_fusion
            }
        }

        return explanation

    def _prepare_transformer_input(self, eeg_data: np.ndarray) -> torch.Tensor:
        """Convert EEG data to transformer input format"""
        # This would implement your specific EEG tokenization strategy
        # Options: patch-based, channel-wise, temporal windows, etc.

        # Placeholder implementation - flatten and project
        flattened = eeg_data.flatten()

        # Truncate or pad to fixed length
        target_length = 1024
        if len(flattened) > target_length:
            flattened = flattened[:target_length]
        else:
            flattened = np.pad(flattened, (0, target_length - len(flattened)))

        # Convert to tensor and add batch dimension
        input_tensor = torch.from_numpy(flattened).float().unsqueeze(0)

        return input_tensor

    def _compute_gradient_attention(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Fallback attention computation using gradients"""
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        predicted_class = output.argmax(dim=1)

        # Backward pass
        self.model.zero_grad()
        output[0, predicted_class].backward()

        # Use gradient magnitude as proxy for attention
        attention_proxy = input_tensor.grad.abs().squeeze(0)

        return {
            'rollout_attention': attention_proxy.unsqueeze(0),
            'cls_attention': attention_proxy,
            'prediction': output
        }

    def _calculate_attention_statistics(self, attention_results: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate attention pattern statistics"""
        if 'cls_attention' not in attention_results:
            return {}

        cls_attention = attention_results['cls_attention'].cpu().numpy()

        # Attention entropy (measure of focus)
        attention_entropy = entropy(cls_attention + 1e-10)  # Add small constant for numerical stability

        # Attention concentration (top-k analysis)
        sorted_attention = np.sort(cls_attention)[::-1]
        top_10_percent = int(0.1 * len(sorted_attention))
        concentration_top10 = sorted_attention[:top_10_percent].sum()

        # Attention sparsity
        sparsity = (cls_attention < self.config.attention_threshold).sum() / len(cls_attention)

        stats = {
            'attention_entropy': float(attention_entropy),
            'max_attention': float(cls_attention.max()),
            'mean_attention': float(cls_attention.mean()),
            'attention_std': float(cls_attention.std()),
            'concentration_top10': float(concentration_top10),
            'sparsity': float(sparsity),
            'attention_range': float(cls_attention.max() - cls_attention.min())
        }

        return stats

    def visualize_attention(self, explanation: Dict[str, Any], save_path: Optional[Path] = None) -> plt.Figure:
        """Visualize attention patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        cls_attention = explanation['attention']['cls_attention']
        if cls_attention is not None:
            cls_attention_np = cls_attention.cpu().numpy() if torch.is_tensor(cls_attention) else cls_attention

            # Attention heatmap
            axes[0, 0].plot(cls_attention_np)
            axes[0, 0].set_title('CLS Token Attention')
            axes[0, 0].set_xlabel('Token Position')
            axes[0, 0].set_ylabel('Attention Weight')

            # Attention distribution
            axes[0, 1].hist(cls_attention_np, bins=50, alpha=0.7)
            axes[0, 1].set_title('Attention Distribution')
            axes[0, 1].set_xlabel('Attention Weight')
            axes[0, 1].set_ylabel('Frequency')

            # Top attention positions
            top_indices = np.argsort(cls_attention_np)[-10:]
            axes[1, 0].bar(range(len(top_indices)), cls_attention_np[top_indices])
            axes[1, 0].set_title('Top 10 Attention Positions')
            axes[1, 0].set_xlabel('Rank')
            axes[1, 0].set_ylabel('Attention Weight')

            # Cumulative attention
            sorted_attention = np.sort(cls_attention_np)[::-1]
            cumulative = np.cumsum(sorted_attention)
            axes[1, 1].plot(cumulative / cumulative[-1])
            axes[1, 1].set_title('Cumulative Attention')
            axes[1, 1].set_xlabel('Token Rank')
            axes[1, 1].set_ylabel('Cumulative Proportion')
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def batch_explain(self, eeg_batch: List[Tuple[np.ndarray, str]]) -> List[Dict[str, Any]]:
        """Generate explanations for batch of EEG data"""
        explanations = []

        for eeg_data, subject_id in eeg_batch:
            try:
                explanation = self.explain_prediction(eeg_data, subject_id)
                explanations.append(explanation)
            except Exception as e:
                logger.error(f"Failed to explain transformer prediction for {subject_id}: {e}")
                explanations.append({
                    'subject_id': subject_id,
                    'error': str(e)
                })

        return explanations