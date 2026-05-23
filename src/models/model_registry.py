#!/usr/bin/env python3
"""
Model Registry for Deep Learning Pipeline
Phase VII: Versioned model management with audit trails and deployment tracking
"""

import sqlite3
import json
import uuid
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import torch
import pickle

class ModelType(Enum):
    CORE15_PLUS = "core15+"
    CNN_SPECTROGRAM = "cnn_spectrogram"
    RNN_TEMPORAL = "rnn_temporal"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"

class ModelStatus(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"

class Framework(Enum):
    SKLEARN = "sklearn"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    CUSTOM = "custom"

@dataclass
class ModelMetadata:
    """Comprehensive model metadata for registry."""
    model_id: str
    model_name: str
    model_type: ModelType
    architecture: str
    version: str
    framework: Framework
    hyperparameters: Dict
    training_data_sources: List[str]
    performance_metrics: Dict
    model_path: str
    checkpoint_path: Optional[str] = None
    is_deployed: bool = False
    deployment_sites: List[str] = None
    created_by: str = "system"
    status: ModelStatus = ModelStatus.TRAINING

@dataclass
class TrainingRun:
    """Training run metadata and metrics."""
    run_id: str
    model_id: str
    experiment_name: str
    run_name: str
    training_config: Dict
    dataset_split: Dict
    training_metrics: Dict
    validation_metrics: Dict
    final_performance: Dict
    training_duration_seconds: Optional[int] = None
    gpu_used: bool = False
    gpu_model: Optional[str] = None
    status: str = "running"
    error_message: Optional[str] = None
    artifacts_path: Optional[str] = None

class ModelRegistry:
    """Clinical-grade model registry with versioning and audit trails."""

    def __init__(self, db_path: str = "data/eeg_platform.db",
                 models_dir: str = "models"):
        """Initialize model registry."""
        self.db_path = Path(db_path)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.models_dir / "checkpoints").mkdir(exist_ok=True)
        (self.models_dir / "artifacts").mkdir(exist_ok=True)
        (self.models_dir / "exports").mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)

    def get_db_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def audit_log(self, event_type: str, entity_type: str, entity_id: str,
                  action: str, details: Dict = None, user_id: str = "system"):
        """Log audit event for model registry operations."""
        conn = self.get_db_connection()
        try:
            conn.execute("""
                INSERT INTO audit_events
                (event_type, entity_type, entity_id, user_id, action, details, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event_type, entity_type, entity_id, user_id, action,
                json.dumps(details, default=str) if details else None, True
            ))
            conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")
        finally:
            conn.close()

    # ================================================================================
    # MODEL REGISTRATION AND VERSIONING
    # ================================================================================

    def register_model(self, metadata: ModelMetadata, model_object: Any = None,
                      user_id: str = "system") -> str:
        """Register a new model with versioning and audit trail."""

        # Generate unique model ID if not provided
        if not metadata.model_id:
            metadata.model_id = f"MODEL_{uuid.uuid4().hex[:12].upper()}"

        # Create model directory
        model_dir = self.models_dir / metadata.model_id
        model_dir.mkdir(exist_ok=True)

        # Save model object if provided
        if model_object is not None:
            model_path = model_dir / f"{metadata.model_id}_v{metadata.version}.pkl"

            if metadata.framework == Framework.PYTORCH:
                # Save PyTorch model
                if hasattr(model_object, 'state_dict'):
                    torch.save({
                        'model_state_dict': model_object.state_dict(),
                        'model_architecture': metadata.architecture,
                        'hyperparameters': metadata.hyperparameters,
                        'version': metadata.version
                    }, model_path)
                else:
                    torch.save(model_object, model_path)
            else:
                # Save other models with pickle
                with open(model_path, 'wb') as f:
                    pickle.dump(model_object, f)

            metadata.model_path = str(model_path)

        # Calculate model hash for integrity
        model_hash = self._calculate_model_hash(metadata.model_path) if Path(metadata.model_path).exists() else None

        # Register in database
        conn = self.get_db_connection()
        try:
            conn.execute("""
                INSERT INTO models
                (model_id, model_name, model_type, architecture, version, framework,
                 hyperparameters, training_data_sources, performance_metrics,
                 model_path, checkpoint_path, is_deployed, deployment_sites,
                 created_by, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.model_id,
                metadata.model_name,
                metadata.model_type.value,
                metadata.architecture,
                metadata.version,
                metadata.framework.value,
                json.dumps(metadata.hyperparameters),
                json.dumps(metadata.training_data_sources),
                json.dumps(metadata.performance_metrics, default=str),
                metadata.model_path,
                metadata.checkpoint_path,
                metadata.is_deployed,
                json.dumps(metadata.deployment_sites or []),
                metadata.created_by,
                metadata.status.value
            ))
            conn.commit()

            # Log audit event
            self.audit_log('model_register', 'model', metadata.model_id,
                          'register_model',
                          {
                              'model_name': metadata.model_name,
                              'model_type': metadata.model_type.value,
                              'version': metadata.version,
                              'framework': metadata.framework.value,
                              'model_hash': model_hash
                          }, user_id)

            self.logger.info(f"Registered model {metadata.model_id} v{metadata.version}")
            return metadata.model_id

        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")
            raise
        finally:
            conn.close()

    def update_model_status(self, model_id: str, status: ModelStatus,
                           user_id: str = "system") -> bool:
        """Update model status with audit trail."""
        conn = self.get_db_connection()
        try:
            # Get current status
            cursor = conn.execute("SELECT status FROM models WHERE model_id = ?", (model_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Model not found: {model_id}")

            old_status = row[0]

            # Update status
            conn.execute("""
                UPDATE models
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE model_id = ?
            """, (status.value, model_id))
            conn.commit()

            # Log audit event
            self.audit_log('model_status_change', 'model', model_id,
                          f'status_change_{old_status}_to_{status.value}',
                          {'old_status': old_status, 'new_status': status.value}, user_id)

            self.logger.info(f"Updated model {model_id} status: {old_status} → {status.value}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update model status: {e}")
            return False
        finally:
            conn.close()

    def deploy_model(self, model_id: str, site_ids: List[str],
                    user_id: str = "system") -> bool:
        """Deploy model to specified sites."""
        conn = self.get_db_connection()
        try:
            # Update deployment status
            conn.execute("""
                UPDATE models
                SET is_deployed = TRUE, deployment_sites = ?, status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE model_id = ?
            """, (json.dumps(site_ids), ModelStatus.DEPLOYED.value, model_id))
            conn.commit()

            # Log audit event
            self.audit_log('model_deploy', 'model', model_id, 'deploy_model',
                          {'deployment_sites': site_ids}, user_id)

            self.logger.info(f"Deployed model {model_id} to sites: {site_ids}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to deploy model: {e}")
            return False
        finally:
            conn.close()

    # ================================================================================
    # MODEL LOADING AND INFERENCE
    # ================================================================================

    def load_model(self, model_id: str, version: str = "latest") -> Tuple[Any, Dict]:
        """Load model object and metadata for inference."""
        conn = self.get_db_connection()
        try:
            # Get model metadata
            if version == "latest":
                cursor = conn.execute("""
                    SELECT * FROM models
                    WHERE model_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (model_id,))
            else:
                cursor = conn.execute("""
                    SELECT * FROM models
                    WHERE model_id = ? AND version = ?
                """, (model_id, version))

            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Model not found: {model_id} v{version}")

            metadata = dict(row)
            model_path = Path(metadata['model_path'])

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load model based on framework
            framework = Framework(metadata['framework'])

            if framework == Framework.PYTORCH:
                if metadata['model_type'] in ['cnn_spectrogram', 'rnn_temporal', 'transformer']:
                    # Load PyTorch state dict
                    checkpoint = torch.load(model_path, map_location='cpu')
                    model_object = checkpoint  # Will need to reconstruct architecture
                else:
                    model_object = torch.load(model_path, map_location='cpu')
            else:
                # Load with pickle
                with open(model_path, 'rb') as f:
                    model_object = pickle.load(f)

            # Log access for audit
            self.audit_log('model_load', 'model', model_id, 'load_model',
                          {'version': metadata['version']})

            return model_object, metadata

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
        finally:
            conn.close()

    def get_deployed_models(self, site_id: str = None) -> List[Dict]:
        """Get all deployed models, optionally filtered by site."""
        conn = self.get_db_connection()
        try:
            if site_id:
                cursor = conn.execute("""
                    SELECT * FROM models
                    WHERE is_deployed = TRUE
                    AND deployment_sites LIKE ?
                    ORDER BY created_at DESC
                """, (f'%{site_id}%',))
            else:
                cursor = conn.execute("""
                    SELECT * FROM models
                    WHERE is_deployed = TRUE
                    ORDER BY created_at DESC
                """)

            return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            self.logger.error(f"Failed to get deployed models: {e}")
            return []
        finally:
            conn.close()

    # ================================================================================
    # TRAINING RUN MANAGEMENT
    # ================================================================================

    def start_training_run(self, training_run: TrainingRun, user_id: str = "system") -> str:
        """Start a new training run with full tracking."""
        conn = self.get_db_connection()
        try:
            # Detect GPU info
            if training_run.gpu_used and torch.cuda.is_available():
                training_run.gpu_model = torch.cuda.get_device_name(0)

            conn.execute("""
                INSERT INTO training_runs
                (run_id, model_id, experiment_name, run_name, training_config,
                 dataset_split, training_metrics, validation_metrics, final_performance,
                 gpu_used, gpu_model, status, artifacts_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                training_run.run_id,
                training_run.model_id,
                training_run.experiment_name,
                training_run.run_name,
                json.dumps(training_run.training_config),
                json.dumps(training_run.dataset_split),
                json.dumps(training_run.training_metrics),
                json.dumps(training_run.validation_metrics),
                json.dumps(training_run.final_performance),
                training_run.gpu_used,
                training_run.gpu_model,
                training_run.status,
                training_run.artifacts_path
            ))
            conn.commit()

            # Log audit event
            self.audit_log('training_start', 'training_run', training_run.run_id,
                          'start_training_run',
                          {
                              'model_id': training_run.model_id,
                              'experiment_name': training_run.experiment_name,
                              'gpu_used': training_run.gpu_used
                          }, user_id)

            self.logger.info(f"Started training run {training_run.run_id} for model {training_run.model_id}")
            return training_run.run_id

        except Exception as e:
            self.logger.error(f"Failed to start training run: {e}")
            raise
        finally:
            conn.close()

    def update_training_metrics(self, run_id: str, epoch: int, metrics: Dict):
        """Update training metrics for a specific epoch."""
        conn = self.get_db_connection()
        try:
            # Get current metrics
            cursor = conn.execute("SELECT training_metrics FROM training_runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Training run not found: {run_id}")

            current_metrics = json.loads(row[0]) if row[0] else {}

            # Add new epoch metrics
            if 'epochs' not in current_metrics:
                current_metrics['epochs'] = {}

            current_metrics['epochs'][str(epoch)] = metrics
            current_metrics['latest_epoch'] = epoch

            # Update database
            conn.execute("""
                UPDATE training_runs
                SET training_metrics = ?
                WHERE run_id = ?
            """, (json.dumps(current_metrics), run_id))
            conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to update training metrics: {e}")
            raise
        finally:
            conn.close()

    def complete_training_run(self, run_id: str, final_performance: Dict,
                             duration_seconds: int, status: str = "completed",
                             error_message: str = None):
        """Complete a training run with final metrics."""
        conn = self.get_db_connection()
        try:
            conn.execute("""
                UPDATE training_runs
                SET final_performance = ?, training_duration_seconds = ?,
                    completed_at = CURRENT_TIMESTAMP, status = ?, error_message = ?
                WHERE run_id = ?
            """, (
                json.dumps(final_performance),
                duration_seconds,
                status,
                error_message,
                run_id
            ))
            conn.commit()

            # Log audit event
            self.audit_log('training_complete', 'training_run', run_id,
                          f'complete_training_{status}',
                          {
                              'final_performance': final_performance,
                              'duration_seconds': duration_seconds,
                              'status': status
                          })

            self.logger.info(f"Completed training run {run_id} with status: {status}")

        except Exception as e:
            self.logger.error(f"Failed to complete training run: {e}")
            raise
        finally:
            conn.close()

    # ================================================================================
    # MODEL COMPARISON AND ANALYTICS
    # ================================================================================

    def compare_models(self, model_ids: List[str], experiment_name: str,
                      dataset_name: str, validation_type: str = "loso",
                      user_id: str = "system") -> str:
        """Compare multiple models and store results."""
        comparison_id = f"COMP_{uuid.uuid4().hex[:8].upper()}"

        conn = self.get_db_connection()
        try:
            # Get model performance metrics
            comparison_metrics = {}
            statistical_tests = {}
            best_model_id = None
            best_score = 0

            for model_id in model_ids:
                cursor = conn.execute("""
                    SELECT model_name, performance_metrics, model_type
                    FROM models WHERE model_id = ?
                """, (model_id,))
                row = cursor.fetchone()

                if row:
                    metrics = json.loads(row[1]) if row[1] else {}
                    comparison_metrics[model_id] = {
                        'model_name': row[0],
                        'model_type': row[2],
                        'metrics': metrics
                    }

                    # Determine best model by balanced accuracy
                    ba = metrics.get('balanced_accuracy', 0)
                    if ba > best_score:
                        best_score = ba
                        best_model_id = model_id

            # Store comparison results
            conn.execute("""
                INSERT INTO model_comparisons
                (comparison_id, experiment_name, model_ids, dataset_name,
                 validation_type, comparison_metrics, statistical_tests,
                 best_model_id, comparison_summary, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                comparison_id,
                experiment_name,
                json.dumps(model_ids),
                dataset_name,
                validation_type,
                json.dumps(comparison_metrics),
                json.dumps(statistical_tests),
                best_model_id,
                f"Compared {len(model_ids)} models on {dataset_name}",
                user_id
            ))
            conn.commit()

            # Log audit event
            self.audit_log('model_comparison', 'comparison', comparison_id,
                          'compare_models',
                          {
                              'model_ids': model_ids,
                              'experiment_name': experiment_name,
                              'best_model_id': best_model_id,
                              'best_score': best_score
                          }, user_id)

            self.logger.info(f"Created model comparison {comparison_id} with {len(model_ids)} models")
            return comparison_id

        except Exception as e:
            self.logger.error(f"Failed to create model comparison: {e}")
            raise
        finally:
            conn.close()

    def get_model_history(self, model_name: str) -> List[Dict]:
        """Get version history for a model."""
        conn = self.get_db_connection()
        try:
            cursor = conn.execute("""
                SELECT * FROM models
                WHERE model_name = ?
                ORDER BY created_at ASC
            """, (model_name,))

            return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            self.logger.error(f"Failed to get model history: {e}")
            return []
        finally:
            conn.close()

    # ================================================================================
    # UTILITY METHODS
    # ================================================================================

    def _calculate_model_hash(self, model_path: str) -> str:
        """Calculate SHA256 hash of model file for integrity checking."""
        try:
            with open(model_path, 'rb') as f:
                model_data = f.read()
            return hashlib.sha256(model_data).hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to calculate model hash: {e}")
            return None

    def get_registry_stats(self) -> Dict:
        """Get model registry statistics."""
        conn = self.get_db_connection()
        try:
            # Model counts by type and status
            cursor = conn.execute("""
                SELECT model_type, status, COUNT(*) as count
                FROM models
                GROUP BY model_type, status
            """)
            model_stats = [dict(row) for row in cursor.fetchall()]

            # Training run stats
            cursor = conn.execute("""
                SELECT status, COUNT(*) as count, AVG(training_duration_seconds) as avg_duration
                FROM training_runs
                GROUP BY status
            """)
            training_stats = [dict(row) for row in cursor.fetchall()]

            # Deployment stats
            cursor = conn.execute("""
                SELECT COUNT(*) as deployed_models,
                       COUNT(DISTINCT deployment_sites) as unique_sites
                FROM models
                WHERE is_deployed = TRUE
            """)
            deployment_stats = dict(cursor.fetchone())

            return {
                'model_statistics': model_stats,
                'training_statistics': training_stats,
                'deployment_statistics': deployment_stats,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to get registry stats: {e}")
            return {}
        finally:
            conn.close()

def main():
    """Demo model registry functionality."""
    print("="*60)
    print("MODEL REGISTRY - PHASE VII DEEP LEARNING")
    print("Versioned Model Management with Audit Trails")
    print("="*60)

    # Initialize registry
    registry = ModelRegistry()

    try:
        # Create demo model metadata
        metadata = ModelMetadata(
            model_id="MODEL_CNN_DEMO_001",
            model_name="EEG_CNN_Spectrogram_v1",
            model_type=ModelType.CNN_SPECTROGRAM,
            architecture="CNN_2D_Spectrogram",
            version="1.0.0",
            framework=Framework.PYTORCH,
            hyperparameters={
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 50,
                "dropout_rate": 0.3,
                "conv_filters": [32, 64, 128],
                "fc_units": [256, 128]
            },
            training_data_sources=["ds002778", "ds003490", "ds004584"],
            performance_metrics={
                "balanced_accuracy": 0.924,
                "sensitivity": 0.911,
                "specificity": 0.937,
                "auc": 0.962,
                "validation_loss": 0.234
            },
            model_path="models/MODEL_CNN_DEMO_001/MODEL_CNN_DEMO_001_v1.0.0.pkl"
        )

        # Register model
        model_id = registry.register_model(metadata)
        print(f"✅ Registered model: {model_id}")

        # Create demo training run
        training_run = TrainingRun(
            run_id=f"RUN_{uuid.uuid4().hex[:8].upper()}",
            model_id=model_id,
            experiment_name="CNN_Spectrogram_Baseline",
            run_name="initial_training",
            training_config={
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 50
            },
            dataset_split={
                "train_subjects": 60,
                "validation_subjects": 15,
                "test_subjects": 6
            },
            training_metrics={"final_train_loss": 0.156, "final_train_acc": 0.943},
            validation_metrics={"final_val_loss": 0.234, "final_val_acc": 0.924},
            final_performance={"test_balanced_accuracy": 0.924},
            gpu_used=torch.cuda.is_available()
        )

        # Start and complete training run
        run_id = registry.start_training_run(training_run)
        registry.complete_training_run(run_id, training_run.final_performance, 1800)
        print(f"✅ Completed training run: {run_id}")

        # Deploy model
        registry.deploy_model(model_id, ["SITE_AMC_001", "SITE_NEURO_002"])
        print(f"✅ Deployed model to sites")

        # Get registry statistics
        stats = registry.get_registry_stats()
        print(f"✅ Registry stats: {len(stats['model_statistics'])} model types")

        print("\n🎯 Model registry ready for Phase VII deep learning!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())