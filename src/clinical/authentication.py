"""
Enterprise Authentication and Role-Based Access Control
HIPAA-compliant authentication with hospital SSO integration
"""

import hashlib
import jwt
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import bcrypt
import logging
from pathlib import Path
import json
import requests
from functools import wraps

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles with hierarchical permissions"""
    SUPER_ADMIN = "super_admin"
    SYSTEM_ADMIN = "system_admin"
    CLINICAL_ADMIN = "clinical_admin"
    PRINCIPAL_INVESTIGATOR = "principal_investigator"
    NEUROLOGIST = "neurologist"
    CLINICIAN = "clinician"
    RESEARCHER = "researcher"
    TECHNICIAN = "technician"
    AUDITOR = "auditor"
    READ_ONLY = "read_only"

class Permission(Enum):
    """System permissions"""
    # System administration
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    MANAGE_SYSTEM_CONFIG = "manage_system_config"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    MANAGE_SITES = "manage_sites"

    # Clinical operations
    MANAGE_TRIALS = "manage_trials"
    ENROLL_SUBJECTS = "enroll_subjects"
    VIEW_PATIENT_DATA = "view_patient_data"
    MODIFY_PATIENT_DATA = "modify_patient_data"
    VIEW_EEG_DATA = "view_eeg_data"
    UPLOAD_EEG_DATA = "upload_eeg_data"

    # AI/ML operations
    RUN_AI_ANALYSIS = "run_ai_analysis"
    VIEW_AI_PREDICTIONS = "view_ai_predictions"
    MANAGE_AI_MODELS = "manage_ai_models"
    VIEW_EXPLANATIONS = "view_explanations"

    # Data management
    EXPORT_DATA = "export_data"
    DELETE_DATA = "delete_data"
    BULK_OPERATIONS = "bulk_operations"

    # Quality control
    APPROVE_RESULTS = "approve_results"
    OVERRIDE_QC = "override_qc"
    MANAGE_THRESHOLDS = "manage_thresholds"

    # Reporting
    GENERATE_REPORTS = "generate_reports"
    VIEW_REPORTS = "view_reports"
    REGULATORY_SUBMISSION = "regulatory_submission"

@dataclass
class UserProfile:
    """User profile with clinical information"""
    user_id: str
    username: str
    email: str
    full_name: str
    roles: List[UserRole]
    site_access: List[str]  # Site IDs user can access
    trial_access: List[str]  # Trial IDs user can access
    department: Optional[str] = None
    license_number: Optional[str] = None
    credentials: Optional[str] = None  # MD, PhD, etc.
    phone: Optional[str] = None
    created_at: datetime = None
    last_login: Optional[datetime] = None
    is_active: bool = True
    must_change_password: bool = False
    mfa_enabled: bool = False
    password_expires_at: Optional[datetime] = None

@dataclass
class SessionInfo:
    """Active session information"""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    last_activity: datetime
    permissions: Set[Permission]

@dataclass
class SSOConfig:
    """Single Sign-On configuration"""
    provider: str  # "saml", "oauth2", "ldap"
    endpoint_url: str
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    certificate_path: Optional[str] = None
    ldap_base_dn: Optional[str] = None
    attribute_mapping: Dict[str, str] = None

class EnterpriseAuthManager:
    """Enterprise authentication manager with hospital integration"""

    def __init__(self, db_path: str, jwt_secret: str = None):
        self.db_path = Path(db_path)
        self.jwt_secret = jwt_secret or secrets.token_urlsafe(32)
        self.session_timeout = timedelta(hours=8)
        self.password_policy = {
            'min_length': 12,
            'require_uppercase': True,
            'require_lowercase': True,
            'require_numbers': True,
            'require_special': True,
            'max_age_days': 90
        }

        self._init_auth_database()
        self._setup_role_permissions()

    def _init_auth_database(self):
        """Initialize authentication database tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Users table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS auth_users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    roles TEXT NOT NULL, -- JSON array
                    site_access TEXT, -- JSON array
                    trial_access TEXT, -- JSON array
                    department TEXT,
                    license_number TEXT,
                    credentials TEXT,
                    phone TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    must_change_password BOOLEAN DEFAULT FALSE,
                    mfa_enabled BOOLEAN DEFAULT FALSE,
                    mfa_secret TEXT,
                    password_expires_at TIMESTAMP,
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP
                )
            """)

            # Active sessions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (user_id) REFERENCES auth_users (user_id)
                )
            """)

            # Login audit
            conn.execute("""
                CREATE TABLE IF NOT EXISTS auth_audit (
                    audit_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    username TEXT,
                    action TEXT, -- 'login_success', 'login_failed', 'logout', 'password_change', etc.
                    ip_address TEXT,
                    user_agent TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    details TEXT -- JSON
                )
            """)

            # SSO configurations
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sso_configs (
                    config_id TEXT PRIMARY KEY,
                    provider_name TEXT NOT NULL,
                    provider_type TEXT CHECK(provider_type IN ('saml', 'oauth2', 'ldap')),
                    configuration TEXT NOT NULL, -- JSON
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Role permissions cache
            conn.execute("""
                CREATE TABLE IF NOT EXISTS role_permissions (
                    role_name TEXT,
                    permission_name TEXT,
                    granted BOOLEAN DEFAULT TRUE,
                    PRIMARY KEY (role_name, permission_name)
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON auth_sessions (user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_sessions_expires ON auth_sessions (expires_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_audit_user ON auth_audit (user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_audit_timestamp ON auth_audit (timestamp)")

    def _setup_role_permissions(self):
        """Set up default role-permission mappings"""
        role_permissions_map = {
            UserRole.SUPER_ADMIN: [p for p in Permission],  # All permissions

            UserRole.SYSTEM_ADMIN: [
                Permission.MANAGE_USERS, Permission.MANAGE_ROLES, Permission.MANAGE_SYSTEM_CONFIG,
                Permission.VIEW_AUDIT_LOGS, Permission.MANAGE_SITES, Permission.MANAGE_TRIALS,
                Permission.VIEW_PATIENT_DATA, Permission.VIEW_EEG_DATA, Permission.RUN_AI_ANALYSIS,
                Permission.VIEW_AI_PREDICTIONS, Permission.MANAGE_AI_MODELS, Permission.VIEW_EXPLANATIONS,
                Permission.GENERATE_REPORTS, Permission.VIEW_REPORTS
            ],

            UserRole.CLINICAL_ADMIN: [
                Permission.MANAGE_TRIALS, Permission.ENROLL_SUBJECTS, Permission.VIEW_PATIENT_DATA,
                Permission.MODIFY_PATIENT_DATA, Permission.VIEW_EEG_DATA, Permission.UPLOAD_EEG_DATA,
                Permission.RUN_AI_ANALYSIS, Permission.VIEW_AI_PREDICTIONS, Permission.VIEW_EXPLANATIONS,
                Permission.APPROVE_RESULTS, Permission.GENERATE_REPORTS, Permission.VIEW_REPORTS
            ],

            UserRole.PRINCIPAL_INVESTIGATOR: [
                Permission.MANAGE_TRIALS, Permission.ENROLL_SUBJECTS, Permission.VIEW_PATIENT_DATA,
                Permission.MODIFY_PATIENT_DATA, Permission.VIEW_EEG_DATA, Permission.UPLOAD_EEG_DATA,
                Permission.RUN_AI_ANALYSIS, Permission.VIEW_AI_PREDICTIONS, Permission.VIEW_EXPLANATIONS,
                Permission.APPROVE_RESULTS, Permission.EXPORT_DATA, Permission.GENERATE_REPORTS,
                Permission.VIEW_REPORTS, Permission.REGULATORY_SUBMISSION
            ],

            UserRole.NEUROLOGIST: [
                Permission.VIEW_PATIENT_DATA, Permission.MODIFY_PATIENT_DATA, Permission.VIEW_EEG_DATA,
                Permission.UPLOAD_EEG_DATA, Permission.RUN_AI_ANALYSIS, Permission.VIEW_AI_PREDICTIONS,
                Permission.VIEW_EXPLANATIONS, Permission.APPROVE_RESULTS, Permission.OVERRIDE_QC,
                Permission.GENERATE_REPORTS, Permission.VIEW_REPORTS
            ],

            UserRole.CLINICIAN: [
                Permission.VIEW_PATIENT_DATA, Permission.VIEW_EEG_DATA, Permission.RUN_AI_ANALYSIS,
                Permission.VIEW_AI_PREDICTIONS, Permission.VIEW_EXPLANATIONS, Permission.VIEW_REPORTS
            ],

            UserRole.RESEARCHER: [
                Permission.VIEW_PATIENT_DATA, Permission.VIEW_EEG_DATA, Permission.RUN_AI_ANALYSIS,
                Permission.VIEW_AI_PREDICTIONS, Permission.VIEW_EXPLANATIONS, Permission.EXPORT_DATA,
                Permission.GENERATE_REPORTS, Permission.VIEW_REPORTS
            ],

            UserRole.TECHNICIAN: [
                Permission.VIEW_EEG_DATA, Permission.UPLOAD_EEG_DATA, Permission.RUN_AI_ANALYSIS,
                Permission.VIEW_AI_PREDICTIONS
            ],

            UserRole.AUDITOR: [
                Permission.VIEW_AUDIT_LOGS, Permission.VIEW_PATIENT_DATA, Permission.VIEW_EEG_DATA,
                Permission.VIEW_AI_PREDICTIONS, Permission.VIEW_REPORTS
            ],

            UserRole.READ_ONLY: [
                Permission.VIEW_REPORTS
            ]
        }

        with sqlite3.connect(self.db_path) as conn:
            # Clear existing permissions
            conn.execute("DELETE FROM role_permissions")

            # Insert role permissions
            for role, permissions in role_permissions_map.items():
                for permission in permissions:
                    conn.execute("""
                        INSERT INTO role_permissions (role_name, permission_name, granted)
                        VALUES (?, ?, ?)
                    """, (role.value, permission.value, True))

    def create_user(self, username: str, email: str, password: str,
                   full_name: str, roles: List[UserRole],
                   site_access: List[str] = None,
                   trial_access: List[str] = None,
                   **kwargs) -> Optional[str]:
        """Create new user account"""

        # Validate password
        if not self._validate_password(password):
            raise ValueError("Password does not meet policy requirements")

        # Generate salt and hash password
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)

        user_id = f"user_{secrets.token_urlsafe(8)}"
        password_expires_at = datetime.now() + timedelta(days=self.password_policy['max_age_days'])

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO auth_users (
                        user_id, username, email, password_hash, salt, full_name,
                        roles, site_access, trial_access, department, license_number,
                        credentials, phone, password_expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id, username, email, password_hash.decode('utf-8'), salt.decode('utf-8'),
                    full_name, json.dumps([role.value for role in roles]),
                    json.dumps(site_access or []), json.dumps(trial_access or []),
                    kwargs.get('department'), kwargs.get('license_number'),
                    kwargs.get('credentials'), kwargs.get('phone'), password_expires_at
                ))

            logger.info(f"Created user account: {username}")
            return user_id

        except sqlite3.IntegrityError as e:
            logger.error(f"Failed to create user {username}: {e}")
            return None

    def authenticate_user(self, username: str, password: str,
                         ip_address: str = None, user_agent: str = None) -> Optional[str]:
        """Authenticate user and create session"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT user_id, password_hash, salt, is_active, failed_login_attempts,
                       locked_until, password_expires_at
                FROM auth_users
                WHERE username = ? OR email = ?
            """, (username, username))

            user_data = cursor.fetchone()

            if not user_data:
                self._audit_login_attempt(None, username, 'login_failed', ip_address, user_agent,
                                        {'reason': 'user_not_found'})
                return None

            user_id, stored_hash, salt, is_active, failed_attempts, locked_until, password_expires = user_data

            # Check if account is locked
            if locked_until and datetime.fromisoformat(locked_until) > datetime.now():
                self._audit_login_attempt(user_id, username, 'login_failed', ip_address, user_agent,
                                        {'reason': 'account_locked'})
                return None

            # Check if account is active
            if not is_active:
                self._audit_login_attempt(user_id, username, 'login_failed', ip_address, user_agent,
                                        {'reason': 'account_disabled'})
                return None

            # Verify password
            if not bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
                # Increment failed attempts
                failed_attempts += 1
                locked_until_time = None

                if failed_attempts >= 5:  # Lock after 5 failed attempts
                    locked_until_time = datetime.now() + timedelta(minutes=30)

                conn.execute("""
                    UPDATE auth_users
                    SET failed_login_attempts = ?, locked_until = ?
                    WHERE user_id = ?
                """, (failed_attempts, locked_until_time, user_id))

                self._audit_login_attempt(user_id, username, 'login_failed', ip_address, user_agent,
                                        {'reason': 'invalid_password', 'failed_attempts': failed_attempts})
                return None

            # Check password expiration
            if password_expires and datetime.fromisoformat(password_expires) < datetime.now():
                self._audit_login_attempt(user_id, username, 'login_failed', ip_address, user_agent,
                                        {'reason': 'password_expired'})
                return None

            # Successful authentication - reset failed attempts
            conn.execute("""
                UPDATE auth_users
                SET failed_login_attempts = 0, locked_until = NULL, last_login = ?
                WHERE user_id = ?
            """, (datetime.now(), user_id))

            # Create session
            session_id = self._create_session(user_id, ip_address, user_agent)

            self._audit_login_attempt(user_id, username, 'login_success', ip_address, user_agent,
                                    {'session_id': session_id})

            return session_id

    def _create_session(self, user_id: str, ip_address: str = None,
                       user_agent: str = None) -> str:
        """Create new user session"""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + self.session_timeout

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO auth_sessions (
                    session_id, user_id, expires_at, ip_address, user_agent
                ) VALUES (?, ?, ?, ?, ?)
            """, (session_id, user_id, expires_at, ip_address, user_agent))

        return session_id

    def validate_session(self, session_id: str) -> Optional[SessionInfo]:
        """Validate session and return session info"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT s.session_id, s.user_id, s.created_at, s.expires_at,
                       s.ip_address, s.user_agent, s.last_activity,
                       u.roles, u.is_active
                FROM auth_sessions s
                JOIN auth_users u ON s.user_id = u.user_id
                WHERE s.session_id = ? AND s.is_active = TRUE
            """, (session_id,))

            session_data = cursor.fetchone()

            if not session_data:
                return None

            (session_id, user_id, created_at, expires_at, ip_address,
             user_agent, last_activity, roles_json, is_active) = session_data

            # Check expiration
            if datetime.fromisoformat(expires_at) < datetime.now():
                self._deactivate_session(session_id)
                return None

            # Check user is still active
            if not is_active:
                self._deactivate_session(session_id)
                return None

            # Update last activity
            conn.execute("""
                UPDATE auth_sessions SET last_activity = ? WHERE session_id = ?
            """, (datetime.now(), session_id))

            # Get permissions for user roles
            roles = [UserRole(role) for role in json.loads(roles_json)]
            permissions = self._get_permissions_for_roles(roles)

            return SessionInfo(
                session_id=session_id,
                user_id=user_id,
                created_at=datetime.fromisoformat(created_at),
                expires_at=datetime.fromisoformat(expires_at),
                ip_address=ip_address,
                user_agent=user_agent,
                last_activity=datetime.fromisoformat(last_activity),
                permissions=permissions
            )

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by user ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT user_id, username, email, full_name, roles, site_access,
                       trial_access, department, license_number, credentials,
                       phone, created_at, last_login, is_active, must_change_password,
                       mfa_enabled, password_expires_at
                FROM auth_users
                WHERE user_id = ?
            """, (user_id,))

            user_data = cursor.fetchone()

            if not user_data:
                return None

            return UserProfile(
                user_id=user_data[0],
                username=user_data[1],
                email=user_data[2],
                full_name=user_data[3],
                roles=[UserRole(role) for role in json.loads(user_data[4])],
                site_access=json.loads(user_data[5]) if user_data[5] else [],
                trial_access=json.loads(user_data[6]) if user_data[6] else [],
                department=user_data[7],
                license_number=user_data[8],
                credentials=user_data[9],
                phone=user_data[10],
                created_at=datetime.fromisoformat(user_data[11]) if user_data[11] else None,
                last_login=datetime.fromisoformat(user_data[12]) if user_data[12] else None,
                is_active=bool(user_data[13]),
                must_change_password=bool(user_data[14]),
                mfa_enabled=bool(user_data[15]),
                password_expires_at=datetime.fromisoformat(user_data[16]) if user_data[16] else None
            )

    def _get_permissions_for_roles(self, roles: List[UserRole]) -> Set[Permission]:
        """Get all permissions for given roles"""
        permissions = set()

        with sqlite3.connect(self.db_path) as conn:
            for role in roles:
                cursor = conn.execute("""
                    SELECT permission_name FROM role_permissions
                    WHERE role_name = ? AND granted = TRUE
                """, (role.value,))

                for (permission_name,) in cursor.fetchall():
                    permissions.add(Permission(permission_name))

        return permissions

    def _validate_password(self, password: str) -> bool:
        """Validate password against policy"""
        policy = self.password_policy

        if len(password) < policy['min_length']:
            return False

        if policy['require_uppercase'] and not any(c.isupper() for c in password):
            return False

        if policy['require_lowercase'] and not any(c.islower() for c in password):
            return False

        if policy['require_numbers'] and not any(c.isdigit() for c in password):
            return False

        if policy['require_special'] and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            return False

        return True

    def _audit_login_attempt(self, user_id: Optional[str], username: str,
                           action: str, ip_address: str = None,
                           user_agent: str = None, details: Dict[str, Any] = None):
        """Audit login attempt"""
        audit_id = f"auth_{secrets.token_urlsafe(8)}"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO auth_audit (
                    audit_id, user_id, username, action, ip_address, user_agent, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                audit_id, user_id, username, action, ip_address, user_agent,
                json.dumps(details) if details else None
            ))

    def _deactivate_session(self, session_id: str):
        """Deactivate session"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE auth_sessions SET is_active = FALSE WHERE session_id = ?
            """, (session_id,))

    def logout_user(self, session_id: str, user_id: str = None):
        """Logout user and deactivate session"""
        self._deactivate_session(session_id)

        if user_id:
            self._audit_login_attempt(user_id, None, 'logout', None, None,
                                    {'session_id': session_id})

class RoleBasedAccessControl:
    """Role-based access control decorator and middleware"""

    def __init__(self, auth_manager: EnterpriseAuthManager):
        self.auth_manager = auth_manager

    def require_permission(self, permission: Permission):
        """Decorator to require specific permission"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract session from request context
                session_id = kwargs.get('session_id') or getattr(func, '_session_id', None)

                if not session_id:
                    raise PermissionError("No active session")

                session = self.auth_manager.validate_session(session_id)
                if not session:
                    raise PermissionError("Invalid session")

                if permission not in session.permissions:
                    raise PermissionError(f"Permission denied: {permission.value}")

                return func(*args, **kwargs)
            return wrapper
        return decorator

    def require_any_permission(self, permissions: List[Permission]):
        """Decorator to require any of the specified permissions"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                session_id = kwargs.get('session_id') or getattr(func, '_session_id', None)

                if not session_id:
                    raise PermissionError("No active session")

                session = self.auth_manager.validate_session(session_id)
                if not session:
                    raise PermissionError("Invalid session")

                if not any(perm in session.permissions for perm in permissions):
                    required = ', '.join([p.value for p in permissions])
                    raise PermissionError(f"Permission denied: requires one of {required}")

                return func(*args, **kwargs)
            return wrapper
        return decorator

    def require_role(self, role: UserRole):
        """Decorator to require specific role"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                session_id = kwargs.get('session_id') or getattr(func, '_session_id', None)

                if not session_id:
                    raise PermissionError("No active session")

                session = self.auth_manager.validate_session(session_id)
                if not session:
                    raise PermissionError("Invalid session")

                user_profile = self.auth_manager.get_user_profile(session.user_id)
                if not user_profile or role not in user_profile.roles:
                    raise PermissionError(f"Role required: {role.value}")

                return func(*args, **kwargs)
            return wrapper
        return decorator