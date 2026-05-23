#!/usr/bin/env python3
"""
Clinical Platform API Endpoints for Trial Management and Audit
Phase VI: Persistent job tracking and regulatory compliance
"""

from flask import Flask, request, jsonify, g
from functools import wraps
import sqlite3
import json
from datetime import datetime, timedelta
import uuid
import hashlib
from pathlib import Path

app = Flask(__name__)

# Database configuration
DATABASE = 'data/eeg_platform.db'

def get_db():
    """Get database connection."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
    return db

@app.teardown_appcontext
def close_connection(exception):
    """Close database connection."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def audit_log(event_type, entity_type, entity_id, action, details=None, user_id=None):
    """Log audit event to database."""
    db = get_db()

    audit_data = {
        'event_type': event_type,
        'entity_type': entity_type,
        'entity_id': entity_id,
        'user_id': user_id or 'system',
        'action': action,
        'details': details,
        'ip_address': request.environ.get('REMOTE_ADDR', 'unknown'),
        'user_agent': request.environ.get('HTTP_USER_AGENT', 'unknown'),
        'success': True
    }

    db.execute("""
        INSERT INTO audit_events
        (event_type, entity_type, entity_id, user_id, action, details, ip_address, user_agent, success)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        audit_data['event_type'], audit_data['entity_type'], audit_data['entity_id'],
        audit_data['user_id'], audit_data['action'], audit_data['details'],
        audit_data['ip_address'], audit_data['user_agent'], audit_data['success']
    ))
    db.commit()

def require_auth(f):
    """Decorator for authentication (simplified for demo)."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # In production, implement proper JWT/OAuth authentication
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authentication required'}), 401

        # Mock user validation
        token = auth_header.split(' ')[1]
        if token != 'demo_token_123':
            return jsonify({'error': 'Invalid token'}), 401

        g.current_user = 'demo_user'
        return f(*args, **kwargs)
    return decorated_function

# ================================================================================
# TRIAL MANAGEMENT ENDPOINTS
# ================================================================================

@app.route('/api/trials', methods=['GET'])
@require_auth
def get_trials():
    """Get all trials with optional filtering."""

    status_filter = request.args.get('status')
    institution_filter = request.args.get('institution')

    db = get_db()
    query = "SELECT * FROM trials WHERE 1=1"
    params = []

    if status_filter:
        query += " AND status = ?"
        params.append(status_filter)

    if institution_filter:
        query += " AND institution LIKE ?"
        params.append(f"%{institution_filter}%")

    query += " ORDER BY created_at DESC"

    cursor = db.execute(query, params)
    trials = [dict(row) for row in cursor.fetchall()]

    audit_log('data_access', 'trial', 'multiple', 'list_trials',
             json.dumps({'filters': {'status': status_filter, 'institution': institution_filter}}),
             g.current_user)

    return jsonify({
        'trials': trials,
        'count': len(trials),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/trials', methods=['POST'])
@require_auth
def create_trial():
    """Create a new clinical trial."""

    data = request.get_json()

    # Validate required fields
    required_fields = ['trial_name', 'protocol_version', 'pi_name', 'institution']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400

    trial_id = data.get('trial_id') or f"TRIAL_{uuid.uuid4().hex[:8].upper()}"

    db = get_db()
    try:
        db.execute("""
            INSERT INTO trials
            (trial_id, trial_name, protocol_version, irb_number, pi_name, institution,
             start_date, end_date, status, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trial_id,
            data['trial_name'],
            data['protocol_version'],
            data.get('irb_number'),
            data['pi_name'],
            data['institution'],
            data.get('start_date'),
            data.get('end_date'),
            data.get('status', 'planning'),
            data.get('description')
        ))
        db.commit()

        audit_log('trial_create', 'trial', trial_id, 'create_trial',
                 json.dumps({'trial_name': data['trial_name']}), g.current_user)

        return jsonify({
            'trial_id': trial_id,
            'message': 'Trial created successfully',
            'timestamp': datetime.now().isoformat()
        }), 201

    except sqlite3.IntegrityError as e:
        return jsonify({'error': 'Trial ID already exists'}), 409

@app.route('/api/trials/<trial_id>', methods=['GET'])
@require_auth
def get_trial(trial_id):
    """Get specific trial details."""

    db = get_db()
    cursor = db.execute("SELECT * FROM trials WHERE trial_id = ?", (trial_id,))
    trial = cursor.fetchone()

    if not trial:
        return jsonify({'error': 'Trial not found'}), 404

    # Get associated sites
    cursor = db.execute("""
        SELECT ts.*, s.site_name, s.institution as site_institution
        FROM trial_sites ts
        JOIN sites s ON ts.site_id = s.site_id
        WHERE ts.trial_id = ?
    """, (trial_id,))
    sites = [dict(row) for row in cursor.fetchall()]

    # Get enrollment summary
    cursor = db.execute("""
        SELECT COUNT(*) as total_subjects,
               COUNT(CASE WHEN status = 'enrolled' THEN 1 END) as enrolled,
               COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed
        FROM subjects WHERE trial_id = ?
    """, (trial_id,))
    enrollment = dict(cursor.fetchone())

    audit_log('data_access', 'trial', trial_id, 'view_trial', None, g.current_user)

    return jsonify({
        'trial': dict(trial),
        'sites': sites,
        'enrollment': enrollment,
        'timestamp': datetime.now().isoformat()
    })

# ================================================================================
# JOB MANAGEMENT ENDPOINTS
# ================================================================================

@app.route('/api/jobs', methods=['GET'])
@require_auth
def get_jobs():
    """Get jobs with filtering and pagination."""

    # Query parameters
    trial_id = request.args.get('trial_id')
    site_id = request.args.get('site_id')
    status = request.args.get('status')
    job_type = request.args.get('job_type')
    feature_set = request.args.get('feature_set')

    # Pagination
    page = int(request.args.get('page', 1))
    per_page = min(int(request.args.get('per_page', 50)), 100)
    offset = (page - 1) * per_page

    # Date range
    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')

    db = get_db()

    # Build query with filters
    query = "SELECT * FROM job_summary WHERE 1=1"
    params = []

    if trial_id:
        query += " AND trial_id = ?"
        params.append(trial_id)

    if site_id:
        query += " AND site_id = ?"
        params.append(site_id)

    if status:
        query += " AND status = ?"
        params.append(status)

    if job_type:
        query += " AND job_type = ?"
        params.append(job_type)

    if feature_set:
        query += " AND feature_set = ?"
        params.append(feature_set)

    if from_date:
        query += " AND submitted_at >= ?"
        params.append(from_date)

    if to_date:
        query += " AND submitted_at <= ?"
        params.append(to_date)

    # Get total count
    count_query = query.replace("SELECT *", "SELECT COUNT(*)")
    cursor = db.execute(count_query, params)
    total_count = cursor.fetchone()[0]

    # Add pagination
    query += " ORDER BY submitted_at DESC LIMIT ? OFFSET ?"
    params.extend([per_page, offset])

    cursor = db.execute(query, params)
    jobs = [dict(row) for row in cursor.fetchall()]

    audit_log('data_access', 'job', 'multiple', 'list_jobs',
             json.dumps({'filters': dict(request.args)}), g.current_user)

    return jsonify({
        'jobs': jobs,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': total_count,
            'pages': (total_count + per_page - 1) // per_page
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/jobs', methods=['POST'])
@require_auth
def create_job():
    """Submit a new processing job."""

    data = request.get_json()

    # Validate required fields
    required_fields = ['subject_id', 'trial_id', 'site_id', 'job_type', 'input_file_path']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400

    job_id = f"JOB_{uuid.uuid4().hex[:12].upper()}"

    db = get_db()
    try:
        db.execute("""
            INSERT INTO jobs
            (job_id, subject_id, trial_id, site_id, session_name, job_type,
             feature_set, status, priority, input_file_path, config_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job_id,
            data['subject_id'],
            data['trial_id'],
            data['site_id'],
            data.get('session_name'),
            data['job_type'],
            data.get('feature_set', 'Core15+'),
            'queued',
            data.get('priority', 0),
            data['input_file_path'],
            json.dumps(data.get('config', {}))
        ))
        db.commit()

        audit_log('job_submit', 'job', job_id, 'submit_job',
                 json.dumps({'job_type': data['job_type'], 'subject_id': data['subject_id']}),
                 g.current_user)

        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Job submitted successfully',
            'timestamp': datetime.now().isoformat()
        }), 201

    except sqlite3.IntegrityError as e:
        return jsonify({'error': 'Invalid subject_id, trial_id, or site_id'}), 400

@app.route('/api/jobs/<job_id>', methods=['GET'])
@require_auth
def get_job(job_id):
    """Get specific job details."""

    db = get_db()
    cursor = db.execute("SELECT * FROM job_summary WHERE job_id = ?", (job_id,))
    job = cursor.fetchone()

    if not job:
        return jsonify({'error': 'Job not found'}), 404

    # Get results if available
    cursor = db.execute("SELECT * FROM results WHERE job_id = ?", (job_id,))
    results = [dict(row) for row in cursor.fetchall()]

    audit_log('data_access', 'job', job_id, 'view_job', None, g.current_user)

    return jsonify({
        'job': dict(job),
        'results': results,
        'timestamp': datetime.now().isoformat()
    })

# ================================================================================
# AUDIT AND MONITORING ENDPOINTS
# ================================================================================

@app.route('/api/audit/events', methods=['GET'])
@require_auth
def get_audit_events():
    """Get audit events with filtering."""

    # Query parameters
    event_type = request.args.get('event_type')
    entity_type = request.args.get('entity_type')
    entity_id = request.args.get('entity_id')
    user_id = request.args.get('user_id')

    # Date range
    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')

    # Pagination
    page = int(request.args.get('page', 1))
    per_page = min(int(request.args.get('per_page', 100)), 500)
    offset = (page - 1) * per_page

    db = get_db()

    # Build query
    query = "SELECT * FROM audit_events WHERE 1=1"
    params = []

    if event_type:
        query += " AND event_type = ?"
        params.append(event_type)

    if entity_type:
        query += " AND entity_type = ?"
        params.append(entity_type)

    if entity_id:
        query += " AND entity_id = ?"
        params.append(entity_id)

    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)

    if from_date:
        query += " AND timestamp >= ?"
        params.append(from_date)

    if to_date:
        query += " AND timestamp <= ?"
        params.append(to_date)

    # Get total count
    count_query = query.replace("SELECT *", "SELECT COUNT(*)")
    cursor = db.execute(count_query, params)
    total_count = cursor.fetchone()[0]

    # Add pagination
    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([per_page, offset])

    cursor = db.execute(query, params)
    events = [dict(row) for row in cursor.fetchall()]

    # Log this audit access (meta-audit)
    audit_log('data_access', 'audit', 'events', 'view_audit_events',
             json.dumps({'filters': dict(request.args)}), g.current_user)

    return jsonify({
        'events': events,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': total_count,
            'pages': (total_count + per_page - 1) // per_page
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/monitoring/dashboard', methods=['GET'])
@require_auth
def get_dashboard():
    """Get monitoring dashboard data."""

    trial_id = request.args.get('trial_id')
    site_id = request.args.get('site_id')

    db = get_db()

    # Job statistics
    job_query = "SELECT status, COUNT(*) as count FROM jobs WHERE 1=1"
    params = []

    if trial_id:
        job_query += " AND trial_id = ?"
        params.append(trial_id)

    if site_id:
        job_query += " AND site_id = ?"
        params.append(site_id)

    job_query += " GROUP BY status"

    cursor = db.execute(job_query, params)
    job_stats = {row[0]: row[1] for row in cursor.fetchall()}

    # QC statistics
    qc_query = """
        SELECT
            COUNT(*) as total_results,
            COUNT(CASE WHEN qc_pass = 1 THEN 1 END) as qc_passed,
            AVG(qc_score) as avg_qc_score,
            AVG(balanced_accuracy) as avg_accuracy
        FROM results r
        JOIN jobs j ON r.job_id = j.job_id
        WHERE j.status = 'completed'
    """
    qc_params = []

    if trial_id:
        qc_query += " AND j.trial_id = ?"
        qc_params.append(trial_id)

    if site_id:
        qc_query += " AND j.site_id = ?"
        qc_params.append(site_id)

    cursor = db.execute(qc_query, qc_params)
    qc_stats = dict(cursor.fetchone())

    # Recent activity (last 24 hours)
    yesterday = (datetime.now() - timedelta(days=1)).isoformat()
    activity_query = """
        SELECT event_type, COUNT(*) as count
        FROM audit_events
        WHERE timestamp >= ?
    """
    activity_params = [yesterday]

    if trial_id:
        activity_query += " AND entity_id LIKE ?"
        activity_params.append(f"%{trial_id}%")

    activity_query += " GROUP BY event_type"

    cursor = db.execute(activity_query, activity_params)
    recent_activity = {row[0]: row[1] for row in cursor.fetchall()}

    audit_log('data_access', 'dashboard', 'monitoring', 'view_dashboard',
             json.dumps({'trial_id': trial_id, 'site_id': site_id}), g.current_user)

    return jsonify({
        'job_statistics': job_stats,
        'qc_statistics': qc_stats,
        'recent_activity': recent_activity,
        'timestamp': datetime.now().isoformat()
    })

# ================================================================================
# HEALTH CHECK AND SYSTEM ENDPOINTS
# ================================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """System health check."""

    db = get_db()

    try:
        # Test database connection
        cursor = db.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        table_count = cursor.fetchone()[0]

        # Check recent activity
        cursor = db.execute("SELECT COUNT(*) FROM audit_events WHERE timestamp >= datetime('now', '-1 hour')")
        recent_events = cursor.fetchone()[0]

        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'tables': table_count,
            'recent_events': recent_events,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/system/config', methods=['GET'])
@require_auth
def get_system_config():
    """Get system configuration (non-sensitive values only)."""

    db = get_db()
    cursor = db.execute("""
        SELECT config_category, config_key, config_value, config_type, description
        FROM system_config
        WHERE is_sensitive = 0
        ORDER BY config_category, config_key
    """)

    configs = {}
    for row in cursor.fetchall():
        category = row[0]
        if category not in configs:
            configs[category] = {}

        configs[category][row[1]] = {
            'value': row[2],
            'type': row[3],
            'description': row[4]
        }

    audit_log('data_access', 'system', 'config', 'view_config', None, g.current_user)

    return jsonify({
        'configuration': configs,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Ensure database exists
    if not Path(DATABASE).exists():
        print("Database not found. Please run create_database_schema.py first.")
        exit(1)

    print("🚀 Starting EEG Clinical Platform API")
    print(f"📊 Database: {DATABASE}")
    print("🔐 Authentication: Bearer token required")
    print("📡 Endpoints: /api/trials, /api/jobs, /api/audit, /api/monitoring")

    app.run(debug=True, host='0.0.0.0', port=5000)