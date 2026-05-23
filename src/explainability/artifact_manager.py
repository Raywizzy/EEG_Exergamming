"""
Unified Explainability Artifact Management System
Handles persistence, retrieval, and export of model explanations across all model types
"""

import json
import pickle
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import logging
import hashlib
import zipfile
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import base64
import io

logger = logging.getLogger(__name__)

@dataclass
class ExplanationMetadata:
    """Metadata for explainability artifacts"""
    explanation_id: str
    subject_id: str
    model_id: str
    model_type: str  # "core15+", "cnn_spectrogram", "transformer", "ensemble"
    explanation_type: str  # "grad_cam", "attention", "feature_attribution", "ensemble"
    created_at: datetime
    created_by: str
    data_hash: str
    model_version: str
    confidence_score: float
    predicted_class: int
    explanation_config: Dict[str, Any]
    quality_metrics: Dict[str, float]
    file_paths: Dict[str, str]

@dataclass
class ExplanationArtifact:
    """Complete explainability artifact"""
    metadata: ExplanationMetadata
    explanation_data: Dict[str, Any]
    visualizations: Dict[str, bytes]  # Stored as bytes for DB
    summary_stats: Dict[str, float]
    audit_trail: List[Dict[str, Any]]

class ExplainabilityArtifactManager:
    """Manages storage, retrieval, and export of explainability artifacts"""

    def __init__(self, db_path: str, artifacts_dir: str):
        self.db_path = Path(db_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.artifacts_dir / "visualizations").mkdir(exist_ok=True)
        (self.artifacts_dir / "data").mkdir(exist_ok=True)
        (self.artifacts_dir / "reports").mkdir(exist_ok=True)
        (self.artifacts_dir / "exports").mkdir(exist_ok=True)

        self._init_database()

    def _init_database(self):
        """Initialize explainability database tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Explainability artifacts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS explainability_artifacts (
                    explanation_id TEXT PRIMARY KEY,
                    subject_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_type TEXT CHECK(model_type IN ('core15+', 'cnn_spectrogram', 'rnn_temporal', 'transformer', 'ensemble')),
                    explanation_type TEXT CHECK(explanation_type IN ('grad_cam', 'attention', 'feature_attribution', 'ensemble', 'integrated_gradients', 'shap')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT,
                    data_hash TEXT NOT NULL,
                    model_version TEXT,
                    confidence_score REAL,
                    predicted_class INTEGER,
                    explanation_config TEXT, -- JSON
                    quality_metrics TEXT, -- JSON
                    file_paths TEXT, -- JSON
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            """)

            # Explanation data storage (for complex nested data)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS explanation_data (
                    explanation_id TEXT,
                    data_key TEXT,
                    data_value BLOB,
                    data_type TEXT, -- "json", "numpy", "pickle"
                    compression TEXT, -- "none", "gzip"
                    PRIMARY KEY (explanation_id, data_key),
                    FOREIGN KEY (explanation_id) REFERENCES explainability_artifacts (explanation_id)
                )
            """)

            # Visualization storage
            conn.execute("""
                CREATE TABLE IF NOT EXISTS explanation_visualizations (
                    explanation_id TEXT,
                    viz_name TEXT,
                    viz_type TEXT, -- "plot", "heatmap", "attention_map", "feature_plot"
                    viz_data BLOB, -- PNG/SVG bytes
                    viz_metadata TEXT, -- JSON
                    PRIMARY KEY (explanation_id, viz_name),
                    FOREIGN KEY (explanation_id) REFERENCES explainability_artifacts (explanation_id)
                )
            """)

            # Audit trail for explanations
            conn.execute("""
                CREATE TABLE IF NOT EXISTS explanation_audit (
                    audit_id TEXT PRIMARY KEY,
                    explanation_id TEXT,
                    action TEXT, -- "created", "accessed", "exported", "modified"
                    user_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    details TEXT, -- JSON
                    ip_address TEXT,
                    FOREIGN KEY (explanation_id) REFERENCES explainability_artifacts (explanation_id)
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_subject ON explainability_artifacts (subject_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_model ON explainability_artifacts (model_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_type ON explainability_artifacts (explanation_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_created ON explainability_artifacts (created_at)")

    def store_explanation(self, artifact: ExplanationArtifact, user_id: str = "system") -> str:
        """Store complete explainability artifact"""
        explanation_id = artifact.metadata.explanation_id

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Store metadata
                conn.execute("""
                    INSERT INTO explainability_artifacts (
                        explanation_id, subject_id, model_id, model_type, explanation_type,
                        created_at, created_by, data_hash, model_version, confidence_score,
                        predicted_class, explanation_config, quality_metrics, file_paths
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    explanation_id,
                    artifact.metadata.subject_id,
                    artifact.metadata.model_id,
                    artifact.metadata.model_type,
                    artifact.metadata.explanation_type,
                    artifact.metadata.created_at,
                    artifact.metadata.created_by,
                    artifact.metadata.data_hash,
                    artifact.metadata.model_version,
                    artifact.metadata.confidence_score,
                    artifact.metadata.predicted_class,
                    json.dumps(artifact.metadata.explanation_config),
                    json.dumps(artifact.metadata.quality_metrics),
                    json.dumps(artifact.metadata.file_paths)
                ))

                # Store explanation data
                for key, value in artifact.explanation_data.items():
                    serialized_data, data_type = self._serialize_data(value)
                    conn.execute("""
                        INSERT INTO explanation_data (explanation_id, data_key, data_value, data_type, compression)
                        VALUES (?, ?, ?, ?, ?)
                    """, (explanation_id, key, serialized_data, data_type, "none"))

                # Store visualizations
                for viz_name, viz_data in artifact.visualizations.items():
                    conn.execute("""
                        INSERT INTO explanation_visualizations (explanation_id, viz_name, viz_type, viz_data, viz_metadata)
                        VALUES (?, ?, ?, ?, ?)
                    """, (explanation_id, viz_name, "plot", viz_data, "{}"))

                # Create audit entry
                self._create_audit_entry(explanation_id, "created", user_id, {"artifact_size": len(str(artifact))})

            logger.info(f"Stored explainability artifact {explanation_id}")
            return explanation_id

        except Exception as e:
            logger.error(f"Failed to store explainability artifact: {e}")
            raise

    def retrieve_explanation(self, explanation_id: str, user_id: str = "system") -> Optional[ExplanationArtifact]:
        """Retrieve complete explainability artifact"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get metadata
                cursor = conn.execute("""
                    SELECT * FROM explainability_artifacts WHERE explanation_id = ?
                """, (explanation_id,))

                row = cursor.fetchone()
                if not row:
                    return None

                # Reconstruct metadata
                metadata = ExplanationMetadata(
                    explanation_id=row[0],
                    subject_id=row[1],
                    model_id=row[2],
                    model_type=row[3],
                    explanation_type=row[4],
                    created_at=datetime.fromisoformat(row[5]),
                    created_by=row[6],
                    data_hash=row[7],
                    model_version=row[8],
                    confidence_score=row[9],
                    predicted_class=row[10],
                    explanation_config=json.loads(row[11]) if row[11] else {},
                    quality_metrics=json.loads(row[12]) if row[12] else {},
                    file_paths=json.loads(row[13]) if row[13] else {}
                )

                # Get explanation data
                cursor = conn.execute("""
                    SELECT data_key, data_value, data_type FROM explanation_data
                    WHERE explanation_id = ?
                """, (explanation_id,))

                explanation_data = {}
                for data_row in cursor.fetchall():
                    key, serialized_data, data_type = data_row
                    explanation_data[key] = self._deserialize_data(serialized_data, data_type)

                # Get visualizations
                cursor = conn.execute("""
                    SELECT viz_name, viz_data FROM explanation_visualizations
                    WHERE explanation_id = ?
                """, (explanation_id,))

                visualizations = {}
                for viz_row in cursor.fetchall():
                    viz_name, viz_data = viz_row
                    visualizations[viz_name] = viz_data

                # Get audit trail
                cursor = conn.execute("""
                    SELECT action, user_id, timestamp, details FROM explanation_audit
                    WHERE explanation_id = ? ORDER BY timestamp
                """, (explanation_id,))

                audit_trail = []
                for audit_row in cursor.fetchall():
                    audit_trail.append({
                        'action': audit_row[0],
                        'user_id': audit_row[1],
                        'timestamp': audit_row[2],
                        'details': json.loads(audit_row[3]) if audit_row[3] else {}
                    })

                # Create audit entry for access
                self._create_audit_entry(explanation_id, "accessed", user_id, {})

                artifact = ExplanationArtifact(
                    metadata=metadata,
                    explanation_data=explanation_data,
                    visualizations=visualizations,
                    summary_stats={},  # Can be computed from explanation_data
                    audit_trail=audit_trail
                )

                return artifact

        except Exception as e:
            logger.error(f"Failed to retrieve explainability artifact {explanation_id}: {e}")
            return None

    def search_explanations(self, filters: Dict[str, Any], limit: int = 100) -> List[ExplanationMetadata]:
        """Search explainability artifacts with filters"""
        query = "SELECT * FROM explainability_artifacts WHERE 1=1"
        params = []

        # Build dynamic query
        if 'subject_id' in filters:
            query += " AND subject_id = ?"
            params.append(filters['subject_id'])

        if 'model_id' in filters:
            query += " AND model_id = ?"
            params.append(filters['model_id'])

        if 'model_type' in filters:
            query += " AND model_type = ?"
            params.append(filters['model_type'])

        if 'explanation_type' in filters:
            query += " AND explanation_type = ?"
            params.append(filters['explanation_type'])

        if 'confidence_min' in filters:
            query += " AND confidence_score >= ?"
            params.append(filters['confidence_min'])

        if 'date_from' in filters:
            query += " AND created_at >= ?"
            params.append(filters['date_from'])

        if 'date_to' in filters:
            query += " AND created_at <= ?"
            params.append(filters['date_to'])

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                results = []

                for row in cursor.fetchall():
                    metadata = ExplanationMetadata(
                        explanation_id=row[0],
                        subject_id=row[1],
                        model_id=row[2],
                        model_type=row[3],
                        explanation_type=row[4],
                        created_at=datetime.fromisoformat(row[5]),
                        created_by=row[6],
                        data_hash=row[7],
                        model_version=row[8],
                        confidence_score=row[9],
                        predicted_class=row[10],
                        explanation_config=json.loads(row[11]) if row[11] else {},
                        quality_metrics=json.loads(row[12]) if row[12] else {},
                        file_paths=json.loads(row[13]) if row[13] else {}
                    )
                    results.append(metadata)

                return results

        except Exception as e:
            logger.error(f"Failed to search explanations: {e}")
            return []

    def export_explanation_pdf(self, explanation_id: str, output_path: Path,
                              user_id: str = "system") -> bool:
        """Export explanation as PDF report"""
        artifact = self.retrieve_explanation(explanation_id, user_id)
        if not artifact:
            return False

        try:
            # Create PDF document
            doc = SimpleDocTemplate(str(output_path), pagesize=letter)
            story = []
            styles = getSampleStyleSheet()

            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                textColor=colors.darkblue,
                spaceAfter=30
            )
            story.append(Paragraph(f"Model Explanation Report", title_style))
            story.append(Spacer(1, 12))

            # Metadata section
            story.append(Paragraph("Explanation Metadata", styles['Heading2']))
            metadata_data = [
                ['Subject ID', artifact.metadata.subject_id],
                ['Model ID', artifact.metadata.model_id],
                ['Model Type', artifact.metadata.model_type],
                ['Explanation Type', artifact.metadata.explanation_type],
                ['Created At', artifact.metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')],
                ['Confidence Score', f"{artifact.metadata.confidence_score:.3f}"],
                ['Predicted Class', str(artifact.metadata.predicted_class)],
                ['Data Hash', artifact.metadata.data_hash]
            ]

            metadata_table = Table(metadata_data, colWidths=[2*inch, 3*inch])
            metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(metadata_table)
            story.append(Spacer(1, 20))

            # Prediction section
            story.append(Paragraph("Prediction Results", styles['Heading2']))
            if 'prediction' in artifact.explanation_data:
                pred_data = artifact.explanation_data['prediction']
                story.append(Paragraph(f"Predicted Class: {pred_data.get('class', 'Unknown')}", styles['Normal']))
                story.append(Paragraph(f"Confidence: {pred_data.get('confidence', 0):.3f}", styles['Normal']))

                if 'probabilities' in pred_data:
                    prob_text = "Class Probabilities: " + ", ".join([f"Class {i}: {p:.3f}" for i, p in enumerate(pred_data['probabilities'])])
                    story.append(Paragraph(prob_text, styles['Normal']))
            story.append(Spacer(1, 20))

            # Add visualizations
            if artifact.visualizations:
                story.append(Paragraph("Visualizations", styles['Heading2']))
                for viz_name, viz_data in artifact.visualizations.items():
                    try:
                        # Save visualization temporarily
                        temp_viz_path = self.artifacts_dir / "temp_viz.png"
                        with open(temp_viz_path, 'wb') as f:
                            f.write(viz_data)

                        # Add to PDF
                        story.append(Paragraph(f"{viz_name.replace('_', ' ').title()}", styles['Heading3']))
                        img = Image(str(temp_viz_path), width=6*inch, height=4*inch)
                        story.append(img)
                        story.append(Spacer(1, 12))

                        # Clean up
                        temp_viz_path.unlink()

                    except Exception as e:
                        logger.warning(f"Failed to add visualization {viz_name}: {e}")

            # Attribution summary (if available)
            if 'attributions' in artifact.explanation_data:
                story.append(Paragraph("Attribution Summary", styles['Heading2']))
                attr_data = artifact.explanation_data['attributions']

                if 'feature_importance' in attr_data:
                    for method, importance in attr_data['feature_importance'].items():
                        story.append(Paragraph(f"{method.title()} Method", styles['Heading3']))

                        if 'top_features' in importance:
                            feature_data = [['Feature', 'Attribution', 'Value']]
                            for feat in importance['top_features'][:10]:
                                feature_data.append([
                                    feat['feature_name'],
                                    f"{feat['attribution_value']:.3f}",
                                    f"{feat['feature_value']:.3f}"
                                ])

                            feature_table = Table(feature_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
                            feature_table.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, 0), 10),
                                ('GRID', (0, 0), (-1, -1), 1, colors.black)
                            ]))
                            story.append(feature_table)
                            story.append(Spacer(1, 12))

            # Build PDF
            doc.build(story)

            # Create audit entry
            self._create_audit_entry(explanation_id, "exported", user_id, {
                "export_type": "pdf",
                "export_path": str(output_path)
            })

            return True

        except Exception as e:
            logger.error(f"Failed to export PDF: {e}")
            return False

    def export_explanation_bundle(self, explanation_id: str, output_path: Path,
                                user_id: str = "system") -> bool:
        """Export complete explanation as ZIP bundle"""
        artifact = self.retrieve_explanation(explanation_id, user_id)
        if not artifact:
            return False

        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add metadata as JSON
                metadata_json = json.dumps(asdict(artifact.metadata), default=str, indent=2)
                zipf.writestr("metadata.json", metadata_json)

                # Add explanation data
                for key, value in artifact.explanation_data.items():
                    if isinstance(value, np.ndarray):
                        # Save numpy arrays as .npy files
                        buffer = io.BytesIO()
                        np.save(buffer, value)
                        zipf.writestr(f"data/{key}.npy", buffer.getvalue())
                    else:
                        # Save as JSON
                        data_json = json.dumps(value, default=str, indent=2)
                        zipf.writestr(f"data/{key}.json", data_json)

                # Add visualizations
                for viz_name, viz_data in artifact.visualizations.items():
                    zipf.writestr(f"visualizations/{viz_name}.png", viz_data)

                # Add audit trail
                audit_json = json.dumps(artifact.audit_trail, default=str, indent=2)
                zipf.writestr("audit_trail.json", audit_json)

                # Add summary report
                summary = {
                    "explanation_summary": {
                        "subject_id": artifact.metadata.subject_id,
                        "model_type": artifact.metadata.model_type,
                        "explanation_type": artifact.metadata.explanation_type,
                        "confidence": artifact.metadata.confidence_score,
                        "created_at": artifact.metadata.created_at.isoformat()
                    }
                }
                zipf.writestr("summary.json", json.dumps(summary, indent=2))

            # Create audit entry
            self._create_audit_entry(explanation_id, "exported", user_id, {
                "export_type": "bundle",
                "export_path": str(output_path)
            })

            return True

        except Exception as e:
            logger.error(f"Failed to export bundle: {e}")
            return False

    def _serialize_data(self, data: Any) -> Tuple[bytes, str]:
        """Serialize data for database storage"""
        if isinstance(data, np.ndarray):
            buffer = io.BytesIO()
            np.save(buffer, data)
            return buffer.getvalue(), "numpy"
        elif isinstance(data, (dict, list)):
            return json.dumps(data, default=str).encode(), "json"
        else:
            return pickle.dumps(data), "pickle"

    def _deserialize_data(self, serialized_data: bytes, data_type: str) -> Any:
        """Deserialize data from database"""
        if data_type == "numpy":
            buffer = io.BytesIO(serialized_data)
            return np.load(buffer)
        elif data_type == "json":
            return json.loads(serialized_data.decode())
        elif data_type == "pickle":
            return pickle.loads(serialized_data)
        else:
            return serialized_data

    def _create_audit_entry(self, explanation_id: str, action: str, user_id: str,
                          details: Dict[str, Any], ip_address: str = None):
        """Create audit trail entry"""
        audit_id = str(uuid.uuid4())

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO explanation_audit (audit_id, explanation_id, action, user_id, details, ip_address)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (audit_id, explanation_id, action, user_id, json.dumps(details), ip_address))

    def get_explanation_statistics(self) -> Dict[str, Any]:
        """Get overall statistics about stored explanations"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_explanations,
                    COUNT(DISTINCT subject_id) as unique_subjects,
                    COUNT(DISTINCT model_id) as unique_models,
                    model_type,
                    explanation_type,
                    AVG(confidence_score) as avg_confidence
                FROM explainability_artifacts
                GROUP BY model_type, explanation_type
            """)

            stats = {
                'total_explanations': 0,
                'unique_subjects': 0,
                'unique_models': 0,
                'by_type': {}
            }

            for row in cursor.fetchall():
                total, subjects, models, model_type, exp_type, avg_conf = row
                stats['total_explanations'] += total
                stats['unique_subjects'] = max(stats['unique_subjects'], subjects)
                stats['unique_models'] = max(stats['unique_models'], models)

                key = f"{model_type}_{exp_type}"
                stats['by_type'][key] = {
                    'count': total,
                    'avg_confidence': avg_conf
                }

            return stats