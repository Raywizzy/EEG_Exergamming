#!/usr/bin/env python3
"""
CNN Spectrogram Classifier for EEG-based Parkinson's Detection
Phase VII: Deep learning pipeline with GPU acceleration and clinical integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mne
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import logging
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class CNNConfig:
    """Configuration for CNN training and inference."""
    # Model architecture
    input_channels: int = 30  # Number of EEG channels
    num_classes: int = 2     # PD vs Control
    conv_filters: List[int] = None
    fc_units: List[int] = None
    dropout_rate: float = 0.3

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    early_stopping_patience: int = 10

    # Spectrogram parameters
    nperseg: int = 512
    noverlap: int = 256
    nfft: int = 1024
    freq_range: Tuple[float, float] = (1.0, 40.0)

    # Data parameters
    sampling_rate: int = 512
    segment_length: int = 2  # seconds

    def __post_init__(self):
        if self.conv_filters is None:
            self.conv_filters = [32, 64, 128, 256]
        if self.fc_units is None:
            self.fc_units = [512, 256, 128]

class EEGSpectrogramDataset(Dataset):
    """Dataset for EEG spectrograms."""

    def __init__(self, spectrograms: np.ndarray, labels: np.ndarray,
                 transform=None):
        """
        Args:
            spectrograms: Array of shape (n_samples, n_channels, freq_bins, time_bins)
            labels: Array of shape (n_samples,) with class labels
            transform: Optional transform to be applied on a sample
        """
        self.spectrograms = torch.FloatTensor(spectrograms)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        spectrogram = self.spectrograms[idx]
        label = self.labels[idx]

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, label

class EEGCNNClassifier(nn.Module):
    """2D CNN for EEG spectrogram classification."""

    def __init__(self, config: CNNConfig):
        super(EEGCNNClassifier, self).__init__()
        self.config = config

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = config.input_channels

        for i, out_channels in enumerate(config.conv_filters):
            # Use pooling only for first few layers to preserve spatial dimensions
            use_pooling = i < 2  # Only pool in first 2 layers

            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]

            if use_pooling:
                layers.extend([
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout2d(config.dropout_rate)
                ])
            else:
                layers.append(nn.Dropout2d(config.dropout_rate))

            conv_block = nn.Sequential(*layers)
            self.conv_layers.append(conv_block)
            in_channels = out_channels

        # Calculate flattened feature size
        self.feature_size = self._get_conv_output_size()

        # Global Average Pooling instead of fixed-size FC
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        in_features = config.conv_filters[-1]  # Last conv layer output channels

        for out_features in config.fc_units:
            fc_block = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout_rate)
            )
            self.fc_layers.append(fc_block)
            in_features = out_features

        # Output layer
        self.classifier = nn.Linear(in_features, config.num_classes)

        # Initialize weights
        self._initialize_weights()

    def _get_conv_output_size(self):
        """Calculate the output size of convolutional layers."""
        # Dummy input to calculate conv output size
        # For 2s segments at 512Hz with nperseg=512, we get ~21 freq bins x few time bins
        # Use a safe size that works with real data: freq bins x time bins
        dummy_input = torch.zeros(1, self.config.input_channels, 40, 8)

        x = dummy_input
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        return x.view(1, -1).size(1)

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass."""
        # Convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        # Classifier
        x = self.classifier(x)

        return x

class EEGSpectrogramProcessor:
    """Processor for converting EEG to spectrograms."""

    def __init__(self, config: CNNConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def eeg_to_spectrogram(self, eeg_data: np.ndarray,
                          sampling_rate: int) -> np.ndarray:
        """
        Convert EEG data to spectrogram.

        Args:
            eeg_data: Array of shape (n_channels, n_samples)
            sampling_rate: Sampling rate in Hz

        Returns:
            Spectrogram array of shape (n_channels, freq_bins, time_bins)
        """
        n_channels, n_samples = eeg_data.shape
        spectrograms = []

        for ch in range(n_channels):
            # Compute spectrogram
            frequencies, times, Sxx = signal.spectrogram(
                eeg_data[ch],
                fs=sampling_rate,
                nperseg=self.config.nperseg,
                noverlap=self.config.noverlap,
                nfft=self.config.nfft
            )

            # Filter frequency range
            freq_mask = (frequencies >= self.config.freq_range[0]) & \
                       (frequencies <= self.config.freq_range[1])
            Sxx_filtered = Sxx[freq_mask, :]

            # Convert to dB scale
            Sxx_db = 10 * np.log10(Sxx_filtered + 1e-10)

            spectrograms.append(Sxx_db)

        return np.array(spectrograms)

    def process_eeg_file(self, file_path: str) -> np.ndarray:
        """
        Process EEG file to spectrograms.

        Args:
            file_path: Path to EEG file (.fif format)

        Returns:
            Spectrogram array
        """
        try:
            # Load EEG data
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)

            # Apply preprocessing
            raw.filter(self.config.freq_range[0], self.config.freq_range[1],
                      verbose=False)
            raw.notch_filter(50, verbose=False)  # Remove line noise

            # Create epochs - use longer segments for better spectrograms
            segment_length = max(self.config.segment_length, 4)  # At least 4 seconds
            events = mne.make_fixed_length_events(
                raw, duration=segment_length
            )

            epochs = mne.Epochs(
                raw, events, tmin=0, tmax=segment_length-1/raw.info['sfreq'],
                baseline=None, preload=True, verbose=False
            )

            # Convert to spectrograms
            spectrograms = []
            for epoch_data in epochs.get_data():
                spectrogram = self.eeg_to_spectrogram(
                    epoch_data, int(raw.info['sfreq'])
                )
                spectrograms.append(spectrogram)

            return np.array(spectrograms)

        except Exception as e:
            self.logger.error(f"Failed to process EEG file {file_path}: {e}")
            raise

class CNNTrainer:
    """Trainer for CNN spectrogram classifier."""

    def __init__(self, config: CNNConfig, device: str = None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)

        # Initialize model
        self.model = EEGCNNClassifier(config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        self.logger.info(f"CNN Trainer initialized on {self.device}")
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Train the model with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()

        self.logger.info(f"Starting training for {self.config.epochs} epochs...")

        for epoch in range(self.config.epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            self.logger.info(
                f"Epoch {epoch+1}/{self.config.epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'config': self.config
                }, 'best_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        training_time = time.time() - start_time

        # Load best model
        checkpoint = torch.load('best_model.pth', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        return {
            'training_time': training_time,
            'best_val_loss': best_val_loss,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'final_train_acc': self.train_accuracies[-1],
            'final_val_acc': self.val_accuracies[-1],
            'epochs_trained': len(self.train_losses)
        }

    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate model on test set."""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                # Get predictions and probabilities
                probs = F.softmax(output, dim=1)
                preds = output.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted'
        )

        # Calculate balanced accuracy
        tn = sum((np.array(all_targets) == 0) & (np.array(all_preds) == 0))
        tp = sum((np.array(all_targets) == 1) & (np.array(all_preds) == 1))
        fn = sum((np.array(all_targets) == 1) & (np.array(all_preds) == 0))
        fp = sum((np.array(all_targets) == 0) & (np.array(all_preds) == 1))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (sensitivity + specificity) / 2

        # AUC
        all_probs_class1 = [prob[1] for prob in all_probs]
        auc = roc_auc_score(all_targets, all_probs_class1)

        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'predictions': all_preds,
            'probabilities': all_probs,
            'targets': all_targets
        }

    def predict(self, data: torch.Tensor) -> Tuple[int, float]:
        """
        Make prediction on single sample.

        Args:
            data: Input tensor of shape (1, channels, freq_bins, time_bins)

        Returns:
            Tuple of (predicted_class, confidence_score)
        """
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            output = self.model(data)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()

        return pred_class, confidence

class CNNInferenceWrapper:
    """Wrapper for CNN inference compatible with job manager."""

    def __init__(self, model_path: str, config: CNNConfig, device: str = None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = EEGSpectrogramProcessor(config)

        # Load model
        self.model = EEGCNNClassifier(config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        self.logger = logging.getLogger(__name__)

    def predict_from_file(self, eeg_file_path: str) -> Dict:
        """
        Predict from EEG file.

        Args:
            eeg_file_path: Path to EEG file

        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()

        try:
            # Process EEG to spectrograms
            spectrograms = self.processor.process_eeg_file(eeg_file_path)

            # Average predictions across all segments
            all_predictions = []
            all_confidences = []

            for spectrogram in spectrograms:
                # Convert to tensor and add batch dimension
                spec_tensor = torch.FloatTensor(spectrogram).unsqueeze(0)

                # Predict
                pred_class, confidence = self._predict_single(spec_tensor)
                all_predictions.append(pred_class)
                all_confidences.append(confidence)

            # Aggregate predictions (majority vote)
            final_prediction = int(np.round(np.mean(all_predictions)))
            final_confidence = np.mean(all_confidences)

            # Convert to clinical labels
            class_labels = {0: 'CONTROL', 1: 'PD'}
            prediction_label = class_labels[final_prediction]

            processing_time = (time.time() - start_time) * 1000  # ms

            return {
                'prediction_class': prediction_label,
                'confidence_score': float(final_confidence),
                'class_probabilities': {
                    'CONTROL': float(1 - final_confidence) if final_prediction == 1 else float(final_confidence),
                    'PD': float(final_confidence) if final_prediction == 1 else float(1 - final_confidence)
                },
                'processing_time_ms': int(processing_time),
                'segments_processed': len(spectrograms),
                'individual_predictions': all_predictions,
                'individual_confidences': all_confidences,
                'model_type': 'cnn_spectrogram',
                'input_shape': list(spectrograms[0].shape) if len(spectrograms) > 0 else None
            }

        except Exception as e:
            self.logger.error(f"CNN prediction failed: {e}")
            return {
                'prediction_class': 'ERROR',
                'confidence_score': 0.0,
                'error_message': str(e),
                'processing_time_ms': int((time.time() - start_time) * 1000)
            }

    def _predict_single(self, data: torch.Tensor) -> Tuple[int, float]:
        """Make prediction on single spectrogram."""
        with torch.no_grad():
            data = data.to(self.device)
            output = self.model(data)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()

        return pred_class, confidence

def create_demo_cnn_model() -> str:
    """Create and save a demo CNN model for testing."""
    print("🧠 Creating demo CNN model...")

    # Configuration
    config = CNNConfig(
        input_channels=30,
        conv_filters=[32, 64, 128],
        fc_units=[256, 128],
        epochs=5,  # Short training for demo
        batch_size=16
    )

    # Create model
    model = EEGCNNClassifier(config)

    # Save model
    model_dir = Path("models/demo")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "cnn_demo_model.pth"

    # Save model with metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'architecture': 'CNN_2D_Spectrogram',
        'version': '1.0.0-demo'
    }, model_path)

    print(f"✅ Demo CNN model saved: {model_path}")
    return str(model_path)

def main():
    """Demo CNN classifier functionality."""
    print("="*60)
    print("CNN SPECTROGRAM CLASSIFIER - PHASE VII")
    print("Deep Learning for EEG-based Parkinson's Detection")
    print("="*60)

    try:
        # Create demo model
        model_path = create_demo_cnn_model()

        # Load the same config used for saving
        checkpoint = torch.load(model_path, weights_only=False)
        config = checkpoint['config']

        # Check if demo EEG file exists
        demo_file = Path("data/demo_eeg.fif")
        if demo_file.exists():
            print("🔍 Testing CNN inference...")

            wrapper = CNNInferenceWrapper(model_path, config)
            result = wrapper.predict_from_file(str(demo_file))

            print(f"✅ CNN Prediction: {result['prediction_class']}")
            print(f"✅ Confidence: {result['confidence_score']:.3f}")
            print(f"✅ Processing time: {result['processing_time_ms']}ms")
            print(f"✅ Segments processed: {result.get('segments_processed', 0)}")
        else:
            print("📄 Demo EEG file not found, skipping inference test")

        print(f"\n🎯 CNN pipeline ready for Phase VII integration!")
        print(f"   Model: {model_path}")
        print(f"   GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU device: {torch.cuda.get_device_name(0)}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())