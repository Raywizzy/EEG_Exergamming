#!/usr/bin/env python3
"""
Test robustness integration in clinical MVP platform
"""

import sys
import json
import time
import requests
from pathlib import Path

# Test configuration
API_BASE = "http://localhost:8080/api"
TEST_BIDS_DIR = "/app/data/in"

def test_robustness_api():
    """Test robustness API endpoints."""
    print("=" * 60)
    print("🧪 Testing Robustness Integration")
    print("=" * 60)

    # Test 1: Check robustness status (should be none initially)
    print("\n1. Testing robustness status endpoint...")
    try:
        response = requests.get(f"{API_BASE}/robustness/status")
        if response.status_code == 200:
            status_data = response.json()
            print(f"   ✅ Status endpoint working: {status_data['status']}")
        else:
            print(f"   ❌ Status endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Status endpoint error: {e}")

    # Test 2: Check robustness results (should be 404 initially)
    print("\n2. Testing robustness results endpoint...")
    try:
        response = requests.get(f"{API_BASE}/robustness/results")
        if response.status_code == 404:
            print("   ✅ Results endpoint correctly returns 404 (no results yet)")
        else:
            print(f"   ⚠️  Unexpected response: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Results endpoint error: {e}")

    # Test 3: Start robustness test (requires valid BIDS data)
    print("\n3. Testing robustness test creation...")
    test_payload = {
        "bids_dir": TEST_BIDS_DIR,
        "site_id": "TEST_SITE",
        "trial_id": "TEST_ROBUSTNESS"
    }

    try:
        response = requests.post(f"{API_BASE}/robustness", json=test_payload)
        if response.status_code == 200:
            job_data = response.json()
            print(f"   ✅ Robustness test created: Job ID {job_data['job_id']}")
            print(f"   📄 Message: {job_data['message']}")
            return job_data['job_id']
        else:
            print(f"   ❌ Failed to create robustness test: {response.status_code}")
            if response.text:
                print(f"   Error details: {response.text}")
    except Exception as e:
        print(f"   ❌ Robustness test creation error: {e}")

    return None

def test_job_monitoring(job_id):
    """Monitor a robustness job."""
    if not job_id:
        print("\n⏭️  Skipping job monitoring (no job ID)")
        return

    print(f"\n4. Monitoring robustness job {job_id}...")

    for i in range(5):  # Check status 5 times
        try:
            response = requests.get(f"{API_BASE}/jobs/{job_id}")
            if response.status_code == 200:
                job_data = response.json()
                status = job_data.get('status', 'unknown')
                print(f"   Check {i+1}: Status = {status}")

                if status == 'succeeded':
                    print("   ✅ Job completed successfully!")
                    break
                elif status == 'failed':
                    error = job_data.get('error', 'Unknown error')
                    print(f"   ❌ Job failed: {error}")
                    break
                elif status in ['queued', 'running']:
                    print(f"   ⏳ Job is {status}, waiting...")
                    time.sleep(2)
                else:
                    print(f"   ⚠️  Unknown status: {status}")

            else:
                print(f"   ❌ Failed to get job status: {response.status_code}")

        except Exception as e:
            print(f"   ❌ Job monitoring error: {e}")

        time.sleep(1)

def test_file_structure():
    """Test that robustness files are created correctly."""
    print("\n5. Testing file structure...")

    # Check robustness harness exists
    harness_path = Path(__file__).parent / "robustness" / "robustness_harness.py"
    if harness_path.exists():
        print("   ✅ Robustness harness script found")
    else:
        print(f"   ❌ Robustness harness script not found at {harness_path}")

    # Check API endpoints are included
    routes_path = Path(__file__).parent / "app" / "routes" / "jobs.py"
    if routes_path.exists():
        with routes_path.open() as f:
            content = f.read()
            if "/robustness" in content:
                print("   ✅ Robustness API endpoints found in routes")
            else:
                print("   ❌ Robustness API endpoints not found in routes")
    else:
        print(f"   ❌ Routes file not found at {routes_path}")

def test_ui_integration():
    """Test UI integration."""
    print("\n6. Testing UI integration...")

    jobs_template = Path(__file__).parent / "app" / "templates" / "jobs.html"
    if jobs_template.exists():
        with jobs_template.open() as f:
            content = f.read()
            if "startRobustnessTest" in content:
                print("   ✅ Robustness button found in jobs template")
            else:
                print("   ❌ Robustness button not found in jobs template")
    else:
        print(f"   ❌ Jobs template not found at {jobs_template}")

def main():
    """Run all robustness integration tests."""
    print("Starting robustness integration tests...")

    # Test file structure first
    test_file_structure()
    test_ui_integration()

    # Test API endpoints
    job_id = test_robustness_api()

    # Monitor job if created
    test_job_monitoring(job_id)

    print("\n" + "=" * 60)
    print("🧪 Robustness Integration Test Complete")
    print("=" * 60)

    print("\nNext steps:")
    print("1. Deploy the platform: docker-compose up -d")
    print("2. Access the web UI: http://localhost:8080/ui/jobs")
    print("3. Click 'Run Robustness Test' to test with real data")
    print("4. Monitor progress and view results in the dashboard")

if __name__ == "__main__":
    main()