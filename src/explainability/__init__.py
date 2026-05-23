"""
Explainability Layer for EEG Classification Models
Phase VII-B: Model interpretability and clinical decision support
"""

from .grad_cam import GradCAM, CNNExplainer
from .attention_analysis import TransformerAttentionAnalyzer
from .tabular_attribution import Core15Explainer
from .artifact_manager import ExplainabilityArtifactManager
from .visualization import ExplanationVisualizer

__all__ = [
    'GradCAM',
    'CNNExplainer',
    'TransformerAttentionAnalyzer',
    'Core15Explainer',
    'ExplainabilityArtifactManager',
    'ExplanationVisualizer'
]