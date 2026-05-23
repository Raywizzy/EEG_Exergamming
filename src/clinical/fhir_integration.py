"""
FHIR Integration for Hospital EHR Connectivity
Provides HL7 FHIR R4 compliant interfaces for EEG data and AI predictions
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
import requests
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import base64

logger = logging.getLogger(__name__)

@dataclass
class FHIRConfig:
    """FHIR server configuration"""
    base_url: str
    client_id: str
    client_secret: Optional[str] = None
    auth_url: Optional[str] = None
    scope: str = "system/*.read system/*.write"
    timeout: int = 30
    verify_ssl: bool = True

@dataclass
class PatientReference:
    """FHIR Patient reference"""
    patient_id: str
    display_name: Optional[str] = None
    mrn: Optional[str] = None  # Medical Record Number

class FHIRClient:
    """FHIR R4 compliant client for EHR integration"""

    def __init__(self, config: FHIRConfig):
        self.config = config
        self.access_token = None
        self.token_expires_at = None

    def authenticate(self) -> bool:
        """Authenticate with FHIR server using OAuth2/SMART"""
        if not self.config.auth_url:
            logger.warning("No auth URL configured, skipping authentication")
            return True

        try:
            auth_data = {
                'grant_type': 'client_credentials',
                'client_id': self.config.client_id,
                'scope': self.config.scope
            }

            if self.config.client_secret:
                auth_data['client_secret'] = self.config.client_secret

            response = requests.post(
                self.config.auth_url,
                data=auth_data,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl
            )

            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get('access_token')
                expires_in = token_data.get('expires_in', 3600)
                self.token_expires_at = datetime.now() + datetime.timedelta(seconds=expires_in)

                logger.info("FHIR authentication successful")
                return True
            else:
                logger.error(f"FHIR authentication failed: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"FHIR authentication error: {e}")
            return False

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication"""
        headers = {
            'Content-Type': 'application/fhir+json',
            'Accept': 'application/fhir+json'
        }

        if self.access_token:
            headers['Authorization'] = f'Bearer {self.access_token}'

        return headers

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Optional[Dict]:
        """Make authenticated FHIR request"""
        # Check token expiration
        if self.token_expires_at and datetime.now() >= self.token_expires_at:
            self.authenticate()

        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self._get_headers(),
                json=data,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl
            )

            if response.status_code in [200, 201]:
                return response.json()
            else:
                logger.error(f"FHIR request failed: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"FHIR request error: {e}")
            return None

    def create_resource(self, resource: Dict[str, Any]) -> Optional[str]:
        """Create FHIR resource"""
        resource_type = resource.get('resourceType')
        if not resource_type:
            logger.error("Resource must have resourceType")
            return None

        result = self._make_request('POST', resource_type, resource)
        if result:
            return result.get('id')
        return None

    def get_resource(self, resource_type: str, resource_id: str) -> Optional[Dict]:
        """Get FHIR resource by ID"""
        return self._make_request('GET', f"{resource_type}/{resource_id}")

    def update_resource(self, resource: Dict[str, Any]) -> bool:
        """Update FHIR resource"""
        resource_type = resource.get('resourceType')
        resource_id = resource.get('id')

        if not resource_type or not resource_id:
            logger.error("Resource must have resourceType and id for update")
            return False

        result = self._make_request('PUT', f"{resource_type}/{resource_id}", resource)
        return result is not None

    def search_resources(self, resource_type: str, parameters: Dict[str, str]) -> Optional[List[Dict]]:
        """Search FHIR resources with parameters"""
        query_string = '&'.join([f"{k}={v}" for k, v in parameters.items()])
        endpoint = f"{resource_type}?{query_string}"

        result = self._make_request('GET', endpoint)
        if result and result.get('resourceType') == 'Bundle':
            return [entry.get('resource') for entry in result.get('entry', [])]

        return None

class EEGObservationBuilder:
    """Builder for FHIR Observation resources representing EEG data and AI predictions"""

    def __init__(self):
        self.observation = {
            'resourceType': 'Observation',
            'id': str(uuid.uuid4()),
            'status': 'final',
            'category': [{
                'coding': [{
                    'system': 'http://terminology.hl7.org/CodeSystem/observation-category',
                    'code': 'procedure',
                    'display': 'Procedure'
                }]
            }],
            'meta': {
                'profile': ['http://hl7.org/fhir/StructureDefinition/Observation']
            }
        }

    def set_patient(self, patient_ref: PatientReference) -> 'EEGObservationBuilder':
        """Set patient reference"""
        self.observation['subject'] = {
            'reference': f"Patient/{patient_ref.patient_id}",
            'display': patient_ref.display_name
        }
        return self

    def set_encounter(self, encounter_id: str) -> 'EEGObservationBuilder':
        """Set encounter reference"""
        self.observation['encounter'] = {
            'reference': f"Encounter/{encounter_id}"
        }
        return self

    def set_eeg_recording(self, recording_metadata: Dict[str, Any]) -> 'EEGObservationBuilder':
        """Set EEG recording as the primary observation"""
        self.observation.update({
            'code': {
                'coding': [{
                    'system': 'http://loinc.org',
                    'code': '11524-6',
                    'display': 'EEG study'
                }]
            },
            'effectiveDateTime': datetime.now(timezone.utc).isoformat(),
            'component': [{
                'code': {
                    'coding': [{
                        'system': 'http://snomed.info/sct',
                        'code': '54550000',
                        'display': 'Electroencephalogram'
                    }]
                },
                'valueString': json.dumps(recording_metadata)
            }]
        })
        return self

    def add_ai_prediction(self, prediction_result: Dict[str, Any],
                         model_info: Dict[str, str]) -> 'EEGObservationBuilder':
        """Add AI prediction as observation component"""
        if 'component' not in self.observation:
            self.observation['component'] = []

        # AI prediction component
        prediction_component = {
            'code': {
                'coding': [{
                    'system': 'http://snomed.info/sct',
                    'code': '386053000',
                    'display': 'Evaluation procedure'
                }],
                'text': 'AI/ML Prediction'
            },
            'valueCodeableConcept': {
                'coding': [{
                    'system': 'http://snomed.info/sct',
                    'code': prediction_result.get('prediction_code', '182840001'),
                    'display': prediction_result.get('prediction_display', 'Parkinson disease')
                }],
                'text': f"Prediction: {prediction_result.get('predicted_class', 'Unknown')}"
            }
        }

        self.observation['component'].append(prediction_component)

        # Confidence component
        confidence_component = {
            'code': {
                'coding': [{
                    'system': 'http://loinc.org',
                    'code': 'LA11114-1',
                    'display': 'Confidence level'
                }]
            },
            'valueQuantity': {
                'value': prediction_result.get('confidence', 0.0),
                'unit': 'percent',
                'system': 'http://unitsofmeasure.org',
                'code': '%'
            }
        }

        self.observation['component'].append(confidence_component)

        # Model information
        self.observation['device'] = {
            'display': f"AI Model: {model_info.get('model_name', 'Unknown')} v{model_info.get('version', '1.0')}"
        }

        # Add model metadata as extension
        self.observation['extension'] = [{
            'url': 'http://example.org/fhir/StructureDefinition/ai-model-info',
            'valueString': json.dumps(model_info)
        }]

        return self

    def add_robustness_metrics(self, robustness_data: Dict[str, Any]) -> 'EEGObservationBuilder':
        """Add robustness validation metrics"""
        if 'component' not in self.observation:
            self.observation['component'] = []

        robustness_component = {
            'code': {
                'coding': [{
                    'system': 'http://loinc.org',
                    'code': '72134-0',
                    'display': 'Quality assessment'
                }],
                'text': 'Robustness Validation'
            },
            'valueString': json.dumps(robustness_data)
        }

        self.observation['component'].append(robustness_component)
        return self

    def add_explainability_data(self, explanation_id: str, explanation_summary: Dict[str, Any]) -> 'EEGObservationBuilder':
        """Add explainability information"""
        if 'extension' not in self.observation:
            self.observation['extension'] = []

        explainability_extension = {
            'url': 'http://example.org/fhir/StructureDefinition/ai-explainability',
            'extension': [
                {
                    'url': 'explanation-id',
                    'valueString': explanation_id
                },
                {
                    'url': 'explanation-summary',
                    'valueString': json.dumps(explanation_summary)
                }
            ]
        }

        self.observation['extension'].append(explainability_extension)
        return self

    def build(self) -> Dict[str, Any]:
        """Build the FHIR Observation resource"""
        # Add issued timestamp
        self.observation['issued'] = datetime.now(timezone.utc).isoformat()

        # Add performer (the AI system)
        self.observation['performer'] = [{
            'display': 'EEG AI Analysis System',
            'type': 'Device'
        }]

        return self.observation

class DiagnosticReportBuilder:
    """Builder for FHIR DiagnosticReport resources for comprehensive EEG analysis reports"""

    def __init__(self):
        self.report = {
            'resourceType': 'DiagnosticReport',
            'id': str(uuid.uuid4()),
            'status': 'final',
            'category': [{
                'coding': [{
                    'system': 'http://terminology.hl7.org/CodeSystem/v2-0074',
                    'code': 'NE',
                    'display': 'Neurophysiology'
                }]
            }],
            'code': {
                'coding': [{
                    'system': 'http://loinc.org',
                    'code': '11524-6',
                    'display': 'EEG study'
                }]
            },
            'meta': {
                'profile': ['http://hl7.org/fhir/StructureDefinition/DiagnosticReport']
            }
        }

    def set_patient(self, patient_ref: PatientReference) -> 'DiagnosticReportBuilder':
        """Set patient reference"""
        self.report['subject'] = {
            'reference': f"Patient/{patient_ref.patient_id}",
            'display': patient_ref.display_name
        }
        return self

    def set_encounter(self, encounter_id: str) -> 'DiagnosticReportBuilder':
        """Set encounter reference"""
        self.report['encounter'] = {
            'reference': f"Encounter/{encounter_id}"
        }
        return self

    def add_observation(self, observation_id: str) -> 'DiagnosticReportBuilder':
        """Add observation reference"""
        if 'result' not in self.report:
            self.report['result'] = []

        self.report['result'].append({
            'reference': f"Observation/{observation_id}"
        })
        return self

    def set_conclusion(self, conclusion: str, conclusion_codes: List[Dict[str, str]] = None) -> 'DiagnosticReportBuilder':
        """Set diagnostic conclusion"""
        self.report['conclusion'] = conclusion

        if conclusion_codes:
            self.report['conclusionCode'] = []
            for code_info in conclusion_codes:
                self.report['conclusionCode'].append({
                    'coding': [{
                        'system': code_info.get('system', 'http://snomed.info/sct'),
                        'code': code_info.get('code'),
                        'display': code_info.get('display')
                    }]
                })

        return self

    def add_ai_analysis_summary(self, analysis_summary: Dict[str, Any]) -> 'DiagnosticReportBuilder':
        """Add AI analysis summary"""
        if 'extension' not in self.report:
            self.report['extension'] = []

        ai_summary_extension = {
            'url': 'http://example.org/fhir/StructureDefinition/ai-analysis-summary',
            'valueString': json.dumps(analysis_summary)
        }

        self.report['extension'].append(ai_summary_extension)
        return self

    def add_pdf_attachment(self, pdf_data: bytes, filename: str) -> 'DiagnosticReportBuilder':
        """Add PDF report as attachment"""
        if 'presentedForm' not in self.report:
            self.report['presentedForm'] = []

        # Encode PDF as base64
        pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')

        attachment = {
            'contentType': 'application/pdf',
            'data': pdf_base64,
            'title': filename,
            'creation': datetime.now(timezone.utc).isoformat()
        }

        self.report['presentedForm'].append(attachment)
        return self

    def set_effective_period(self, start_time: datetime, end_time: datetime) -> 'DiagnosticReportBuilder':
        """Set effective period for the diagnostic procedure"""
        self.report['effectivePeriod'] = {
            'start': start_time.isoformat(),
            'end': end_time.isoformat()
        }
        return self

    def build(self) -> Dict[str, Any]:
        """Build the FHIR DiagnosticReport resource"""
        # Add issued timestamp
        self.report['issued'] = datetime.now(timezone.utc).isoformat()

        # Add performer (the AI system and supervising clinician)
        self.report['performer'] = [
            {
                'display': 'EEG AI Analysis System',
                'type': 'Device'
            },
            {
                'display': 'Supervising Neurologist',
                'type': 'Practitioner'
            }
        ]

        return self.report

class EHRIntegrationManager:
    """High-level manager for EHR integration workflows"""

    def __init__(self, fhir_client: FHIRClient):
        self.fhir_client = fhir_client

    def create_eeg_analysis_record(self,
                                 patient_ref: PatientReference,
                                 eeg_metadata: Dict[str, Any],
                                 prediction_result: Dict[str, Any],
                                 model_info: Dict[str, str],
                                 encounter_id: Optional[str] = None,
                                 explanation_id: Optional[str] = None,
                                 robustness_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Create complete EEG analysis record in EHR"""

        try:
            # Build EEG observation
            obs_builder = EEGObservationBuilder()
            obs_builder.set_patient(patient_ref)

            if encounter_id:
                obs_builder.set_encounter(encounter_id)

            obs_builder.set_eeg_recording(eeg_metadata)
            obs_builder.add_ai_prediction(prediction_result, model_info)

            if robustness_data:
                obs_builder.add_robustness_metrics(robustness_data)

            if explanation_id:
                explanation_summary = {
                    'explanation_id': explanation_id,
                    'explanation_available': True,
                    'confidence_level': prediction_result.get('confidence', 0.0)
                }
                obs_builder.add_explainability_data(explanation_id, explanation_summary)

            observation = obs_builder.build()

            # Create observation in FHIR server
            observation_id = self.fhir_client.create_resource(observation)

            if observation_id:
                logger.info(f"Created EEG observation: {observation_id}")
                return observation_id
            else:
                logger.error("Failed to create EEG observation")
                return None

        except Exception as e:
            logger.error(f"Failed to create EEG analysis record: {e}")
            return None

    def create_diagnostic_report(self,
                               patient_ref: PatientReference,
                               observation_ids: List[str],
                               conclusion: str,
                               analysis_summary: Dict[str, Any],
                               pdf_report: Optional[bytes] = None,
                               encounter_id: Optional[str] = None) -> Optional[str]:
        """Create diagnostic report summarizing EEG analysis"""

        try:
            # Build diagnostic report
            report_builder = DiagnosticReportBuilder()
            report_builder.set_patient(patient_ref)

            if encounter_id:
                report_builder.set_encounter(encounter_id)

            # Add all observations
            for obs_id in observation_ids:
                report_builder.add_observation(obs_id)

            # Set conclusion based on AI analysis
            conclusion_codes = []
            if 'predicted_class' in analysis_summary:
                if analysis_summary['predicted_class'] == 'PD':
                    conclusion_codes.append({
                        'system': 'http://snomed.info/sct',
                        'code': '49049000',
                        'display': 'Parkinson disease'
                    })
                else:
                    conclusion_codes.append({
                        'system': 'http://snomed.info/sct',
                        'code': '17621005',
                        'display': 'Normal'
                    })

            report_builder.set_conclusion(conclusion, conclusion_codes)
            report_builder.add_ai_analysis_summary(analysis_summary)

            # Add PDF if provided
            if pdf_report:
                report_builder.add_pdf_attachment(
                    pdf_report,
                    f"EEG_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                )

            report = report_builder.build()

            # Create report in FHIR server
            report_id = self.fhir_client.create_resource(report)

            if report_id:
                logger.info(f"Created diagnostic report: {report_id}")
                return report_id
            else:
                logger.error("Failed to create diagnostic report")
                return None

        except Exception as e:
            logger.error(f"Failed to create diagnostic report: {e}")
            return None

    def get_patient_eeg_history(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get patient's EEG analysis history"""
        try:
            observations = self.fhir_client.search_resources(
                'Observation',
                {
                    'subject': f"Patient/{patient_id}",
                    'code': 'http://loinc.org|11524-6',
                    '_sort': '-date'
                }
            )

            if observations:
                return observations
            else:
                return []

        except Exception as e:
            logger.error(f"Failed to get patient EEG history: {e}")
            return []

    def validate_fhir_connectivity(self) -> Dict[str, Any]:
        """Validate FHIR server connectivity and capabilities"""
        try:
            # Test authentication
            auth_success = self.fhir_client.authenticate()

            # Test capability statement
            capability = self.fhir_client._make_request('GET', 'metadata')

            # Test patient search (should work for any FHIR server)
            test_search = self.fhir_client.search_resources('Patient', {'_count': '1'})

            return {
                'authentication': auth_success,
                'capability_statement': capability is not None,
                'search_capability': test_search is not None,
                'server_url': self.fhir_client.config.base_url,
                'fhir_version': capability.get('fhirVersion') if capability else 'Unknown',
                'validation_time': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"FHIR connectivity validation failed: {e}")
            return {
                'authentication': False,
                'capability_statement': False,
                'search_capability': False,
                'error': str(e),
                'validation_time': datetime.now(timezone.utc).isoformat()
            }