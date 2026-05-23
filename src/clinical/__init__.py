"""
Phase IX: Clinical Workflow Integration
Hospital-grade systems for clinical deployment and EHR integration
"""

from .fhir_integration import FHIRClient, EEGObservationBuilder, DiagnosticReportBuilder
from .authentication import EnterpriseAuthManager, RoleBasedAccessControl
from .workflow_orchestration import ClinicalWorkflowManager, WorkflowStatus
from .hospital_dashboard import HospitalDashboard, ClinicalInterface
from .hospital_integration import HospitalIntegrationManager
from .consort_compliance import CONSORTManager, TrialDocumentationSystem

__all__ = [
    'FHIRClient',
    'EEGObservationBuilder',
    'DiagnosticReportBuilder',
    'EnterpriseAuthManager',
    'RoleBasedAccessControl',
    'ClinicalWorkflowManager',
    'WorkflowStatus',
    'HospitalDashboard',
    'ClinicalInterface',
    'HospitalIntegrationManager',
    'CONSORTManager',
    'TrialDocumentationSystem'
]