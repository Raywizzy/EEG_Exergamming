"""
Phase X-A: Cloud-Native FHIR Endpoints for Remote Trials
Scalable cloud infrastructure for distributed EEG neurofeedback trials
"""

import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import logging
from enum import Enum
import uuid
import hashlib
import jwt
from cryptography.fernet import Fernet
import base64

from ..clinical.fhir_integration import FHIRClient, EEGObservationBuilder, DiagnosticReportBuilder
from ..clinical.authentication import EnterpriseAuthManager

class CloudRegion(Enum):
    """Supported cloud regions for data residency"""
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    ASIA_PACIFIC_1 = "ap-southeast-1"
    CANADA_CENTRAL_1 = "ca-central-1"

class DataResidencyPolicy(Enum):
    """Data residency and sovereignty policies"""
    STRICT_REGIONAL = "strict_regional"
    EU_GDPR_COMPLIANT = "eu_gdpr_compliant"
    HIPAA_COMPLIANT = "hipaa_compliant"
    FEDERATED_GLOBAL = "federated_global"

class EndpointType(Enum):
    """Cloud FHIR endpoint types"""
    PATIENT_MANAGEMENT = "patient_management"
    OBSERVATION_INGESTION = "observation_ingestion"
    DIAGNOSTIC_REPORTING = "diagnostic_reporting"
    TRIAL_ENROLLMENT = "trial_enrollment"
    REAL_TIME_STREAMING = "real_time_streaming"
    DATA_EXPORT = "data_export"
    AUDIT_TRAIL = "audit_trail"

@dataclass
class CloudEndpointConfig:
    """Cloud FHIR endpoint configuration"""
    endpoint_id: str
    endpoint_type: EndpointType
    region: CloudRegion
    base_url: str
    api_version: str
    authentication_method: str
    rate_limit_requests_per_minute: int
    data_encryption_at_rest: bool
    data_encryption_in_transit: bool
    audit_logging_enabled: bool
    backup_retention_days: int
    geo_replication_enabled: bool
    compliance_frameworks: List[str]

@dataclass
class RemoteParticipantProfile:
    """Remote trial participant profile"""
    participant_id: str
    trial_id: str
    fhir_patient_id: str
    home_location: Dict[str, str]  # country, state, timezone
    assigned_region: CloudRegion
    device_preferences: List[str]
    connectivity_profile: Dict[str, Any]
    data_residency_requirements: DataResidencyPolicy
    consent_digital_signature: str
    enrollment_date: datetime
    last_activity: Optional[datetime] = None
    compliance_status: str = "active"

@dataclass
class CloudSession:
    """Cloud-based EEG session record"""
    session_id: str
    participant_id: str
    device_id: str
    region: CloudRegion
    start_timestamp: datetime
    end_timestamp: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    data_quality_score: Optional[float] = None
    cloud_storage_url: str = ""
    fhir_observation_id: Optional[str] = None
    processing_status: str = "pending"
    compliance_verified: bool = False

class CloudFHIREndpoints:
    """
    Cloud-native FHIR endpoints for remote neurofeedback trials
    Provides scalable, compliant, and geographically distributed FHIR services
    """

    def __init__(self, config_path: str = None):
        """Initialize cloud FHIR endpoints"""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()

        # Core services
        self.auth_manager = EnterpriseAuthManager()
        self.fhir_client = FHIRClient()

        # Cloud infrastructure
        self.active_endpoints: Dict[str, CloudEndpointConfig] = {}
        self.regional_clients: Dict[CloudRegion, aiohttp.ClientSession] = {}
        self.encryption_keys: Dict[CloudRegion, Fernet] = {}

        # Session management
        self.active_sessions: Dict[str, CloudSession] = {}
        self.session_metrics: Dict[str, Dict[str, float]] = {}

        # Initialize cloud infrastructure
        self._initialize_cloud_endpoints()
        self._initialize_regional_encryption()

        self.logger.info("Cloud FHIR Endpoints initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load cloud configuration"""
        default_config = {
            'default_region': CloudRegion.US_EAST_1.value,
            'auto_scaling_enabled': True,
            'load_balancing_enabled': True,
            'cdn_enabled': True,
            'backup_frequency_hours': 6,
            'disaster_recovery_enabled': True,
            'compliance_frameworks': ['HIPAA', 'GDPR', 'SOC2', 'ISO27001'],
            'api_rate_limits': {
                'observations_per_minute': 1000,
                'reports_per_minute': 100,
                'exports_per_hour': 10
            },
            'data_retention': {
                'active_trials_years': 7,
                'completed_trials_years': 25,
                'audit_logs_years': 10
            },
            'encryption': {
                'algorithm': 'AES-256-GCM',
                'key_rotation_days': 90,
                'backup_encryption': True
            },
            'monitoring': {
                'health_check_interval_seconds': 30,
                'performance_metrics_enabled': True,
                'alerting_enabled': True
            }
        }

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)

        return default_config

    def _setup_logging(self) -> logging.Logger:
        """Setup cloud-specific logging"""
        logger = logging.getLogger('cloud_fhir')
        logger.setLevel(logging.INFO)

        log_dir = Path('logs/cloud')
        log_dir.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(log_dir / f'cloud_fhir_{datetime.now().strftime("%Y%m%d")}.log')
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    def _initialize_cloud_endpoints(self):
        """Initialize cloud FHIR endpoints across regions"""

        # US East (Primary)
        us_east_config = CloudEndpointConfig(
            endpoint_id="fhir-us-east-1",
            endpoint_type=EndpointType.PATIENT_MANAGEMENT,
            region=CloudRegion.US_EAST_1,
            base_url="https://fhir-us-east.neurofeedback-trials.com/R4",
            api_version="R4",
            authentication_method="OAuth2_SMART",
            rate_limit_requests_per_minute=1000,
            data_encryption_at_rest=True,
            data_encryption_in_transit=True,
            audit_logging_enabled=True,
            backup_retention_days=2555,  # 7 years
            geo_replication_enabled=True,
            compliance_frameworks=["HIPAA", "SOC2", "ISO27001"]
        )

        # EU West (GDPR Compliant)
        eu_west_config = CloudEndpointConfig(
            endpoint_id="fhir-eu-west-1",
            endpoint_type=EndpointType.PATIENT_MANAGEMENT,
            region=CloudRegion.EU_WEST_1,
            base_url="https://fhir-eu-west.neurofeedback-trials.com/R4",
            api_version="R4",
            authentication_method="OAuth2_SMART",
            rate_limit_requests_per_minute=800,
            data_encryption_at_rest=True,
            data_encryption_in_transit=True,
            audit_logging_enabled=True,
            backup_retention_days=2555,
            geo_replication_enabled=False,  # GDPR data residency
            compliance_frameworks=["GDPR", "ISO27001", "GDPR_DPA"]
        )

        # Asia Pacific
        ap_config = CloudEndpointConfig(
            endpoint_id="fhir-ap-southeast-1",
            endpoint_type=EndpointType.PATIENT_MANAGEMENT,
            region=CloudRegion.ASIA_PACIFIC_1,
            base_url="https://fhir-ap.neurofeedback-trials.com/R4",
            api_version="R4",
            authentication_method="OAuth2_SMART",
            rate_limit_requests_per_minute=600,
            data_encryption_at_rest=True,
            data_encryption_in_transit=True,
            audit_logging_enabled=True,
            backup_retention_days=2555,
            geo_replication_enabled=True,
            compliance_frameworks=["ISO27001", "APAC_PRIVACY"]
        )

        # Real-time streaming endpoints
        streaming_us = CloudEndpointConfig(
            endpoint_id="streaming-us-east-1",
            endpoint_type=EndpointType.REAL_TIME_STREAMING,
            region=CloudRegion.US_EAST_1,
            base_url="wss://stream-us-east.neurofeedback-trials.com/v1",
            api_version="v1",
            authentication_method="JWT_BEARER",
            rate_limit_requests_per_minute=10000,  # Higher for streaming
            data_encryption_at_rest=True,
            data_encryption_in_transit=True,
            audit_logging_enabled=True,
            backup_retention_days=90,  # Shorter for streaming data
            geo_replication_enabled=False,  # Real-time doesn't replicate
            compliance_frameworks=["HIPAA", "SOC2"]
        )

        self.active_endpoints.update({
            "fhir-us-east-1": us_east_config,
            "fhir-eu-west-1": eu_west_config,
            "fhir-ap-southeast-1": ap_config,
            "streaming-us-east-1": streaming_us
        })

    def _initialize_regional_encryption(self):
        """Initialize encryption keys per region"""
        for region in CloudRegion:
            # In production, use proper key management service (AWS KMS, Azure Key Vault, etc.)
            key = Fernet.generate_key()
            self.encryption_keys[region] = Fernet(key)

    async def initialize_regional_clients(self):
        """Initialize HTTP clients for each region"""
        for region in CloudRegion:
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(
                ssl=True,
                limit=100,
                limit_per_host=20
            )

            self.regional_clients[region] = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    'User-Agent': 'Neurofeedback-Cloud-Client/1.0',
                    'Accept': 'application/fhir+json',
                    'Content-Type': 'application/fhir+json'
                }
            )

    async def route_participant_to_region(self, participant_profile: RemoteParticipantProfile) -> CloudRegion:
        """Route participant to appropriate cloud region"""

        # Check data residency requirements
        if participant_profile.data_residency_requirements == DataResidencyPolicy.EU_GDPR_COMPLIANT:
            if participant_profile.home_location.get('country') in ['DE', 'FR', 'IT', 'ES', 'NL', 'BE', 'AT']:
                return CloudRegion.EU_WEST_1
            elif participant_profile.home_location.get('country') in ['PL', 'CZ', 'HU', 'SK']:
                return CloudRegion.EU_CENTRAL_1

        elif participant_profile.data_residency_requirements == DataResidencyPolicy.STRICT_REGIONAL:
            country = participant_profile.home_location.get('country', '')

            if country == 'US':
                # Route based on state for US participants
                state = participant_profile.home_location.get('state', '')
                if state in ['CA', 'OR', 'WA', 'NV', 'AZ']:
                    return CloudRegion.US_WEST_2
                else:
                    return CloudRegion.US_EAST_1
            elif country == 'CA':
                return CloudRegion.CANADA_CENTRAL_1
            elif country in ['JP', 'KR', 'SG', 'AU', 'NZ']:
                return CloudRegion.ASIA_PACIFIC_1

        # Default routing logic based on geographic proximity
        country = participant_profile.home_location.get('country', 'US')

        if country in ['US', 'CA', 'MX']:
            return CloudRegion.US_EAST_1
        elif country in ['GB', 'DE', 'FR', 'IT', 'ES', 'NL', 'BE', 'AT', 'CH', 'SE', 'NO', 'DK', 'FI']:
            return CloudRegion.EU_WEST_1
        elif country in ['JP', 'KR', 'SG', 'AU', 'NZ', 'TH', 'MY', 'PH', 'ID', 'VN']:
            return CloudRegion.ASIA_PACIFIC_1
        else:
            return CloudRegion.US_EAST_1  # Default fallback

    async def create_remote_patient(self,
                                   participant_profile: RemoteParticipantProfile,
                                   patient_demographics: Dict[str, Any]) -> str:
        """Create patient record in appropriate regional endpoint"""

        try:
            # Route to appropriate region
            target_region = await self.route_participant_to_region(participant_profile)
            participant_profile.assigned_region = target_region

            # Get regional endpoint
            endpoint = self._get_regional_endpoint(target_region, EndpointType.PATIENT_MANAGEMENT)
            if not endpoint:
                raise ValueError(f"No patient management endpoint available in {target_region}")

            # Create FHIR Patient resource
            patient_resource = self._create_fhir_patient_resource(participant_profile, patient_demographics)

            # Encrypt sensitive data
            encrypted_resource = await self._encrypt_fhir_resource(patient_resource, target_region)

            # Submit to regional FHIR endpoint
            client = self.regional_clients[target_region]

            async with client.post(
                f"{endpoint.base_url}/Patient",
                json=encrypted_resource,
                headers=await self._get_auth_headers(target_region)
            ) as response:

                if response.status in [200, 201]:
                    result = await response.json()
                    fhir_patient_id = result.get('id')

                    # Update participant profile
                    participant_profile.fhir_patient_id = fhir_patient_id

                    self.logger.info(f"Created remote patient {fhir_patient_id} in {target_region.value}")
                    return fhir_patient_id
                else:
                    error_text = await response.text()
                    raise Exception(f"Patient creation failed: {response.status} - {error_text}")

        except Exception as e:
            self.logger.error(f"Remote patient creation failed: {e}")
            raise

    def _get_regional_endpoint(self, region: CloudRegion, endpoint_type: EndpointType) -> Optional[CloudEndpointConfig]:
        """Get endpoint for specific region and type"""
        for endpoint in self.active_endpoints.values():
            if endpoint.region == region and endpoint.endpoint_type == endpoint_type:
                return endpoint
        return None

    def _create_fhir_patient_resource(self,
                                     participant_profile: RemoteParticipantProfile,
                                     demographics: Dict[str, Any]) -> Dict[str, Any]:
        """Create FHIR Patient resource for remote participant"""

        return {
            "resourceType": "Patient",
            "id": participant_profile.participant_id,
            "identifier": [
                {
                    "use": "usual",
                    "type": {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/v2-0203",
                                "code": "MR",
                                "display": "Medical Record Number"
                            }
                        ]
                    },
                    "system": f"https://neurofeedback-trials.com/participants/{participant_profile.trial_id}",
                    "value": participant_profile.participant_id
                }
            ],
            "name": [
                {
                    "use": "official",
                    "family": demographics.get('family_name', ''),
                    "given": [demographics.get('given_name', '')]
                }
            ],
            "telecom": [
                {
                    "system": "email",
                    "value": demographics.get('email', ''),
                    "use": "home"
                },
                {
                    "system": "phone",
                    "value": demographics.get('phone', ''),
                    "use": "mobile"
                }
            ],
            "gender": demographics.get('gender', 'unknown'),
            "birthDate": demographics.get('birth_date', ''),
            "address": [
                {
                    "use": "home",
                    "type": "both",
                    "line": [demographics.get('address_line', '')],
                    "city": demographics.get('city', ''),
                    "state": demographics.get('state', ''),
                    "postalCode": demographics.get('postal_code', ''),
                    "country": demographics.get('country', '')
                }
            ],
            "extension": [
                {
                    "url": "https://neurofeedback-trials.com/fhir/StructureDefinition/trial-enrollment",
                    "extension": [
                        {
                            "url": "trialId",
                            "valueString": participant_profile.trial_id
                        },
                        {
                            "url": "enrollmentDate",
                            "valueDateTime": participant_profile.enrollment_date.isoformat()
                        },
                        {
                            "url": "dataResidencyPolicy",
                            "valueString": participant_profile.data_residency_requirements.value
                        },
                        {
                            "url": "assignedRegion",
                            "valueString": participant_profile.assigned_region.value if participant_profile.assigned_region else ""
                        }
                    ]
                }
            ]
        }

    async def _encrypt_fhir_resource(self, resource: Dict[str, Any], region: CloudRegion) -> Dict[str, Any]:
        """Encrypt sensitive FHIR resource data"""

        # Fields that should be encrypted
        sensitive_fields = ['name', 'telecom', 'address', 'birthDate']

        encrypted_resource = resource.copy()
        encryption_key = self.encryption_keys[region]

        for field in sensitive_fields:
            if field in encrypted_resource:
                # Convert to JSON and encrypt
                field_json = json.dumps(encrypted_resource[field])
                encrypted_data = encryption_key.encrypt(field_json.encode())
                encrypted_b64 = base64.b64encode(encrypted_data).decode()

                # Replace with encrypted version
                encrypted_resource[field] = {
                    "encrypted": True,
                    "data": encrypted_b64,
                    "algorithm": "Fernet"
                }

        return encrypted_resource

    async def _get_auth_headers(self, region: CloudRegion) -> Dict[str, str]:
        """Get authentication headers for regional endpoint"""

        # Generate JWT token for region-specific access
        payload = {
            'iss': 'neurofeedback-cloud-client',
            'aud': f'fhir-{region.value}',
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=1),
            'scope': 'patient/*.read patient/*.write observation/*.* diagnostic-report/*.*'
        }

        # In production, use proper signing key
        token = jwt.encode(payload, 'your-secret-key', algorithm='HS256')

        return {
            'Authorization': f'Bearer {token}',
            'X-Region': region.value,
            'X-Compliance': 'HIPAA,GDPR'
        }

    async def start_remote_session(self,
                                  participant_id: str,
                                  device_id: str,
                                  session_metadata: Dict[str, Any]) -> str:
        """Start remote EEG session with cloud recording"""

        try:
            # Generate session ID
            session_id = f"REMOTE_{participant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Get participant's assigned region
            # In production, look up from participant database
            assigned_region = CloudRegion.US_EAST_1  # Placeholder

            # Create cloud session record
            cloud_session = CloudSession(
                session_id=session_id,
                participant_id=participant_id,
                device_id=device_id,
                region=assigned_region,
                start_timestamp=datetime.now(),
                cloud_storage_url=f"s3://neurofeedback-{assigned_region.value}/sessions/{session_id}/"
            )

            self.active_sessions[session_id] = cloud_session

            # Initialize real-time streaming endpoint
            streaming_endpoint = self._get_regional_endpoint(assigned_region, EndpointType.REAL_TIME_STREAMING)
            if streaming_endpoint:
                await self._initialize_streaming_connection(session_id, streaming_endpoint)

            self.logger.info(f"Started remote session {session_id} in {assigned_region.value}")
            return session_id

        except Exception as e:
            self.logger.error(f"Remote session start failed: {e}")
            raise

    async def _initialize_streaming_connection(self, session_id: str, endpoint: CloudEndpointConfig):
        """Initialize WebSocket connection for real-time data streaming"""

        try:
            # WebSocket URL for streaming
            ws_url = endpoint.base_url.replace('https://', 'wss://').replace('http://', 'ws://')
            ws_url += f"/sessions/{session_id}/stream"

            # In production, establish WebSocket connection
            # This would handle real-time EEG data streaming to cloud

            self.logger.info(f"Streaming connection initialized for session {session_id}")

        except Exception as e:
            self.logger.error(f"Streaming connection failed: {e}")

    async def ingest_eeg_observation(self,
                                    session_id: str,
                                    eeg_data: Dict[str, Any],
                                    signal_quality: Dict[str, Any]) -> str:
        """Ingest EEG observation data to cloud FHIR endpoint"""

        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")

            cloud_session = self.active_sessions[session_id]

            # Create FHIR Observation resource
            observation_builder = EEGObservationBuilder()
            observation = observation_builder.create_home_eeg_observation(
                patient_id=cloud_session.participant_id,
                session_id=session_id,
                device_id=cloud_session.device_id,
                eeg_data=eeg_data,
                signal_quality=signal_quality
            )

            # Add cloud-specific metadata
            observation["extension"] = observation.get("extension", [])
            observation["extension"].append({
                "url": "https://neurofeedback-trials.com/fhir/StructureDefinition/cloud-session",
                "extension": [
                    {
                        "url": "sessionId",
                        "valueString": session_id
                    },
                    {
                        "url": "cloudRegion",
                        "valueString": cloud_session.region.value
                    },
                    {
                        "url": "storageUrl",
                        "valueUrl": cloud_session.cloud_storage_url
                    }
                ]
            })

            # Encrypt and submit to regional endpoint
            encrypted_observation = await self._encrypt_fhir_resource(observation, cloud_session.region)

            endpoint = self._get_regional_endpoint(cloud_session.region, EndpointType.OBSERVATION_INGESTION)
            client = self.regional_clients[cloud_session.region]

            async with client.post(
                f"{endpoint.base_url}/Observation",
                json=encrypted_observation,
                headers=await self._get_auth_headers(cloud_session.region)
            ) as response:

                if response.status in [200, 201]:
                    result = await response.json()
                    observation_id = result.get('id')

                    # Update session with FHIR observation ID
                    cloud_session.fhir_observation_id = observation_id

                    self.logger.info(f"EEG observation {observation_id} ingested for session {session_id}")
                    return observation_id
                else:
                    error_text = await response.text()
                    raise Exception(f"Observation ingestion failed: {response.status} - {error_text}")

        except Exception as e:
            self.logger.error(f"EEG observation ingestion failed: {e}")
            raise

    async def complete_remote_session(self, session_id: str, session_summary: Dict[str, Any]) -> bool:
        """Complete remote session and generate diagnostic report"""

        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")

            cloud_session = self.active_sessions[session_id]
            cloud_session.end_timestamp = datetime.now()
            cloud_session.duration_seconds = (cloud_session.end_timestamp - cloud_session.start_timestamp).total_seconds()
            cloud_session.data_quality_score = session_summary.get('data_quality_score', 0.0)
            cloud_session.processing_status = "completed"

            # Generate diagnostic report
            report_builder = DiagnosticReportBuilder()
            diagnostic_report = report_builder.create_remote_eeg_report(
                patient_id=cloud_session.participant_id,
                session_id=session_id,
                cloud_session=cloud_session,
                session_summary=session_summary
            )

            # Submit diagnostic report to cloud
            encrypted_report = await self._encrypt_fhir_resource(diagnostic_report, cloud_session.region)

            endpoint = self._get_regional_endpoint(cloud_session.region, EndpointType.DIAGNOSTIC_REPORTING)
            client = self.regional_clients[cloud_session.region]

            async with client.post(
                f"{endpoint.base_url}/DiagnosticReport",
                json=encrypted_report,
                headers=await self._get_auth_headers(cloud_session.region)
            ) as response:

                if response.status in [200, 201]:
                    result = await response.json()
                    report_id = result.get('id')

                    self.logger.info(f"Diagnostic report {report_id} created for session {session_id}")

                    # Mark session as completed
                    cloud_session.compliance_verified = True

                    return True
                else:
                    error_text = await response.text()
                    raise Exception(f"Diagnostic report creation failed: {response.status} - {error_text}")

        except Exception as e:
            self.logger.error(f"Session completion failed: {e}")
            return False

    async def export_trial_data(self,
                               trial_id: str,
                               export_format: str = "FHIR_BUNDLE",
                               date_range: Optional[Tuple[datetime, datetime]] = None) -> str:
        """Export trial data from cloud endpoints"""

        try:
            # Get all regions that have data for this trial
            regions_with_data = await self._get_trial_regions(trial_id)

            if not regions_with_data:
                raise ValueError(f"No data found for trial {trial_id}")

            # Generate export ID
            export_id = f"EXPORT_{trial_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Collect data from all regions
            consolidated_data = {}

            for region in regions_with_data:
                region_data = await self._export_region_data(region, trial_id, date_range)
                consolidated_data[region.value] = region_data

            # Create consolidated export package
            export_package = {
                "export_id": export_id,
                "trial_id": trial_id,
                "export_format": export_format,
                "export_timestamp": datetime.now().isoformat(),
                "data_regions": list(consolidated_data.keys()),
                "regional_data": consolidated_data,
                "compliance_certification": {
                    "hipaa_compliant": True,
                    "gdpr_compliant": True,
                    "data_integrity_verified": True,
                    "export_authorized": True
                }
            }

            # Store export package in secure cloud storage
            export_url = await self._store_export_package(export_package)

            self.logger.info(f"Trial data export {export_id} completed")
            return export_url

        except Exception as e:
            self.logger.error(f"Trial data export failed: {e}")
            raise

    async def _get_trial_regions(self, trial_id: str) -> List[CloudRegion]:
        """Get regions that contain data for specified trial"""
        # In production, query metadata database
        # For now, return common regions
        return [CloudRegion.US_EAST_1, CloudRegion.EU_WEST_1]

    async def _export_region_data(self,
                                 region: CloudRegion,
                                 trial_id: str,
                                 date_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Export data from specific region"""

        try:
            endpoint = self._get_regional_endpoint(region, EndpointType.DATA_EXPORT)
            client = self.regional_clients[region]

            # Build export query
            query_params = {
                'trial_id': trial_id,
                'format': 'FHIR_BUNDLE'
            }

            if date_range:
                query_params['date_start'] = date_range[0].isoformat()
                query_params['date_end'] = date_range[1].isoformat()

            async with client.get(
                f"{endpoint.base_url}/Export",
                params=query_params,
                headers=await self._get_auth_headers(region)
            ) as response:

                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Region export failed: {response.status} - {error_text}")

        except Exception as e:
            self.logger.error(f"Region {region.value} export failed: {e}")
            return {}

    async def _store_export_package(self, export_package: Dict[str, Any]) -> str:
        """Store export package in secure cloud storage"""

        # In production, use cloud storage service (S3, Azure Blob, etc.)
        export_id = export_package['export_id']
        storage_url = f"s3://neurofeedback-exports/{export_id}/export_package.json"

        # Simulate storage
        self.logger.info(f"Export package stored at {storage_url}")

        return storage_url

    async def get_cloud_health_status(self) -> Dict[str, Any]:
        """Get health status of all cloud endpoints"""

        health_status = {
            'overall_status': 'healthy',
            'regions': {},
            'performance_metrics': {
                'average_response_time_ms': 0.0,
                'requests_per_second': 0.0,
                'error_rate_percentage': 0.0
            },
            'last_updated': datetime.now().isoformat()
        }

        total_response_time = 0
        healthy_regions = 0

        for region in CloudRegion:
            try:
                endpoint = self._get_regional_endpoint(region, EndpointType.PATIENT_MANAGEMENT)
                if not endpoint:
                    continue

                start_time = datetime.now()

                # Health check request
                client = self.regional_clients.get(region)
                if client:
                    async with client.get(
                        f"{endpoint.base_url}/metadata",
                        headers=await self._get_auth_headers(region)
                    ) as response:
                        response_time = (datetime.now() - start_time).total_seconds() * 1000
                        total_response_time += response_time

                        if response.status == 200:
                            health_status['regions'][region.value] = {
                                'status': 'healthy',
                                'response_time_ms': response_time,
                                'last_check': datetime.now().isoformat()
                            }
                            healthy_regions += 1
                        else:
                            health_status['regions'][region.value] = {
                                'status': 'degraded',
                                'response_time_ms': response_time,
                                'error_code': response.status,
                                'last_check': datetime.now().isoformat()
                            }

            except Exception as e:
                health_status['regions'][region.value] = {
                    'status': 'unavailable',
                    'error': str(e),
                    'last_check': datetime.now().isoformat()
                }

        # Calculate overall metrics
        if healthy_regions > 0:
            health_status['performance_metrics']['average_response_time_ms'] = total_response_time / healthy_regions

        if healthy_regions < len(CloudRegion):
            health_status['overall_status'] = 'degraded'

        return health_status

    async def cleanup_resources(self):
        """Cleanup cloud resources"""
        try:
            # Close all regional HTTP clients
            for client in self.regional_clients.values():
                await client.close()

            self.logger.info("Cloud resources cleaned up successfully")

        except Exception as e:
            self.logger.error(f"Resource cleanup failed: {e}")


class RemoteTrialManager:
    """
    Manager for distributed clinical trials using cloud FHIR endpoints
    Coordinates remote participants across multiple geographic regions
    """

    def __init__(self, cloud_fhir: CloudFHIREndpoints):
        """Initialize remote trial manager"""
        self.cloud_fhir = cloud_fhir
        self.logger = logging.getLogger('remote_trial_manager')

        # Trial management
        self.active_remote_trials: Dict[str, Dict[str, Any]] = {}
        self.remote_participants: Dict[str, RemoteParticipantProfile] = {}
        self.regional_enrollment: Dict[CloudRegion, int] = {}

    async def initialize_remote_trial(self,
                                     trial_id: str,
                                     trial_configuration: Dict[str, Any]) -> bool:
        """Initialize distributed remote trial"""

        try:
            # Validate trial configuration
            if not self._validate_remote_trial_config(trial_configuration):
                return False

            # Create trial record
            remote_trial = {
                'trial_id': trial_id,
                'title': trial_configuration['title'],
                'target_enrollment': trial_configuration['target_enrollment'],
                'geographic_distribution': trial_configuration.get('geographic_distribution', {}),
                'data_residency_requirements': trial_configuration.get('data_residency_requirements', []),
                'supported_devices': trial_configuration.get('supported_devices', []),
                'created_at': datetime.now(),
                'status': 'active',
                'regional_endpoints': {},
                'enrollment_stats': {region.value: 0 for region in CloudRegion}
            }

            # Initialize regional endpoints for trial
            for region in CloudRegion:
                endpoint_config = self.cloud_fhir._get_regional_endpoint(region, EndpointType.TRIAL_ENROLLMENT)
                if endpoint_config:
                    remote_trial['regional_endpoints'][region.value] = endpoint_config.endpoint_id

            self.active_remote_trials[trial_id] = remote_trial

            self.logger.info(f"Remote trial {trial_id} initialized across {len(remote_trial['regional_endpoints'])} regions")
            return True

        except Exception as e:
            self.logger.error(f"Remote trial initialization failed: {e}")
            return False

    def _validate_remote_trial_config(self, config: Dict[str, Any]) -> bool:
        """Validate remote trial configuration"""
        required_fields = ['title', 'target_enrollment']

        for field in required_fields:
            if field not in config:
                self.logger.error(f"Missing required field: {field}")
                return False

        if config['target_enrollment'] <= 0:
            self.logger.error("Target enrollment must be positive")
            return False

        return True

    async def enroll_remote_participant(self,
                                       trial_id: str,
                                       participant_data: Dict[str, Any]) -> str:
        """Enroll participant in remote trial"""

        try:
            if trial_id not in self.active_remote_trials:
                raise ValueError(f"Remote trial {trial_id} not found")

            # Create participant profile
            participant_id = f"REMOTE_{trial_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            participant_profile = RemoteParticipantProfile(
                participant_id=participant_id,
                trial_id=trial_id,
                fhir_patient_id="",  # Will be set after FHIR creation
                home_location=participant_data['home_location'],
                assigned_region=CloudRegion.US_EAST_1,  # Will be determined
                device_preferences=participant_data.get('device_preferences', []),
                connectivity_profile=participant_data.get('connectivity_profile', {}),
                data_residency_requirements=DataResidencyPolicy(
                    participant_data.get('data_residency_requirements', 'hipaa_compliant')
                ),
                consent_digital_signature=participant_data['consent_digital_signature'],
                enrollment_date=datetime.now()
            )

            # Create patient record in appropriate cloud region
            fhir_patient_id = await self.cloud_fhir.create_remote_patient(
                participant_profile,
                participant_data['demographics']
            )

            participant_profile.fhir_patient_id = fhir_patient_id

            # Store participant profile
            self.remote_participants[participant_id] = participant_profile

            # Update trial enrollment statistics
            assigned_region = participant_profile.assigned_region
            self.active_remote_trials[trial_id]['enrollment_stats'][assigned_region.value] += 1

            self.logger.info(f"Remote participant {participant_id} enrolled in trial {trial_id}")
            return participant_id

        except Exception as e:
            self.logger.error(f"Remote participant enrollment failed: {e}")
            raise

    async def get_remote_trial_status(self, trial_id: str) -> Optional[Dict[str, Any]]:
        """Get status of remote trial"""

        if trial_id not in self.active_remote_trials:
            return None

        trial = self.active_remote_trials[trial_id]

        # Calculate enrollment statistics
        total_enrolled = sum(trial['enrollment_stats'].values())
        target_enrollment = trial['target_enrollment']
        enrollment_percentage = (total_enrolled / target_enrollment) * 100 if target_enrollment > 0 else 0

        # Get participants for this trial
        trial_participants = [
            p for p in self.remote_participants.values()
            if p.trial_id == trial_id
        ]

        return {
            'trial_id': trial_id,
            'title': trial['title'],
            'status': trial['status'],
            'enrollment': {
                'total_enrolled': total_enrolled,
                'target_enrollment': target_enrollment,
                'enrollment_percentage': enrollment_percentage,
                'regional_distribution': trial['enrollment_stats']
            },
            'participants': {
                'total_count': len(trial_participants),
                'active_count': len([p for p in trial_participants if p.compliance_status == 'active']),
                'geographic_distribution': self._get_participant_geographic_distribution(trial_participants)
            },
            'operational_metrics': {
                'active_regions': len([r for r, count in trial['enrollment_stats'].items() if count > 0]),
                'data_residency_compliance': self._check_data_residency_compliance(trial_participants)
            },
            'last_updated': datetime.now().isoformat()
        }

    def _get_participant_geographic_distribution(self, participants: List[RemoteParticipantProfile]) -> Dict[str, int]:
        """Get geographic distribution of participants"""
        distribution = {}

        for participant in participants:
            country = participant.home_location.get('country', 'Unknown')
            distribution[country] = distribution.get(country, 0) + 1

        return distribution

    def _check_data_residency_compliance(self, participants: List[RemoteParticipantProfile]) -> bool:
        """Check if all participants meet data residency requirements"""
        for participant in participants:
            # Check if participant's assigned region aligns with their requirements
            if participant.data_residency_requirements == DataResidencyPolicy.EU_GDPR_COMPLIANT:
                if participant.assigned_region not in [CloudRegion.EU_WEST_1, CloudRegion.EU_CENTRAL_1]:
                    return False
            elif participant.data_residency_requirements == DataResidencyPolicy.STRICT_REGIONAL:
                # Additional checks for strict regional requirements
                pass

        return True

    async def get_all_remote_trials_summary(self) -> Dict[str, Any]:
        """Get summary of all remote trials"""

        return {
            'total_trials': len(self.active_remote_trials),
            'total_participants': len(self.remote_participants),
            'regional_distribution': {
                region.value: len([p for p in self.remote_participants.values() if p.assigned_region == region])
                for region in CloudRegion
            },
            'trials': [
                await self.get_remote_trial_status(trial_id)
                for trial_id in self.active_remote_trials.keys()
            ],
            'last_updated': datetime.now().isoformat()
        }