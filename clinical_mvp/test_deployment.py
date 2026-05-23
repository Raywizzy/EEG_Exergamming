#!/usr/bin/env python3
"""
Core15+ Clinical MVP Deployment Test
Validates that all components are properly configured for deployment.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path

def test_file_structure():
    """Test that all required files exist."""
    print("🔍 Testing file structure...")

    required_files = [
        "docker-compose.yml",
        "Dockerfile",
        "requirements.txt",
        "app/main.py",
        "app/routes/ui.py",
        "app/routes/jobs.py",
        "app/routes/health.py",
        "app/services/jobs.py",
        "app/services/report_pdf.py",
        "app/templates/base.html",
        "app/templates/dashboard.html",
        "app/static/style.css"
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False

    print("✅ All required files present")
    return True

def test_docker_config():
    """Test Docker configuration validity."""
    print("🐳 Testing Docker configuration...")

    # Check docker-compose.yml
    if not Path("docker-compose.yml").exists():
        print("❌ docker-compose.yml not found")
        return False

    # Check Dockerfile
    if not Path("Dockerfile").exists():
        print("❌ Dockerfile not found")
        return False

    print("✅ Docker configuration files present")
    return True

def test_core15_integration():
    """Test Core15+ pipeline integration."""
    print("🧠 Testing Core15+ integration...")

    # Check that Core15+ scripts exist
    core15_files = [
        "../scripts/clinical_trial_processor.py",
        "../config/preprocessing_config.yaml"
    ]

    missing_core15 = []
    for file_path in core15_files:
        if not Path(file_path).exists():
            missing_core15.append(file_path)

    if missing_core15:
        print(f"⚠️  Missing Core15+ files: {missing_core15}")
        print("    These are required for full functionality")
        return False

    print("✅ Core15+ integration ready")
    return True

def test_requirements():
    """Test Python requirements."""
    print("📦 Testing Python requirements...")

    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False

    with open("requirements.txt") as f:
        requirements = f.read()

    required_packages = [
        "fastapi",
        "uvicorn",
        "jinja2",
        "mne",
        "numpy",
        "pandas",
        "scikit-learn"
    ]

    missing_packages = []
    for package in required_packages:
        if package not in requirements:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ Missing packages in requirements.txt: {missing_packages}")
        return False

    print("✅ All required packages listed")
    return True

def test_app_startup():
    """Test application startup (if possible)."""
    print("🚀 Testing application startup...")

    try:
        # Try importing the main app
        sys.path.insert(0, str(Path.cwd()))
        from app.main import app
        print("✅ Application imports successfully")
        return True
    except ImportError as e:
        print(f"⚠️  Application import failed: {e}")
        print("    This is expected if dependencies aren't installed")
        return False

def test_deployment_readiness():
    """Comprehensive deployment readiness test."""
    print("=" * 60)
    print("🏥 Core15+ Clinical MVP Deployment Test")
    print("=" * 60)

    tests = [
        ("File Structure", test_file_structure),
        ("Docker Configuration", test_docker_config),
        ("Core15+ Integration", test_core15_integration),
        ("Python Requirements", test_requirements),
        ("Application Startup", test_app_startup)
    ]

    results = {}
    all_passed = True

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results[test_name] = result
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results[test_name] = False
            all_passed = False

    print("\n" + "=" * 60)
    print("📊 DEPLOYMENT READINESS SUMMARY")
    print("=" * 60)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 DEPLOYMENT READY! All tests passed.")
        print("\nTo deploy:")
        print("  docker-compose up -d")
        print("\nAccess at:")
        print("  Web UI: http://localhost:8080/ui/")
        print("  API Docs: http://localhost:8080/api/docs")
    else:
        print("⚠️  DEPLOYMENT NOT READY. Fix failing tests first.")

    print("=" * 60)
    return all_passed

if __name__ == "__main__":
    success = test_deployment_readiness()
    sys.exit(0 if success else 1)