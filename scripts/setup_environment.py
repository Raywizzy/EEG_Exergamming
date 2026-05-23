#!/usr/bin/env python3
"""
Environment setup and version checking script for EEG Exergaming Project.
Ensures all dependencies are installed and prints version information.
"""

import sys
import platform
import subprocess
from datetime import datetime, timezone

def get_package_version(package_name):
    """Get version of installed package."""
    try:
        result = subprocess.run([sys.executable, "-c", f"import {package_name}; print({package_name}.__version__)"],
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "NOT INSTALLED"
    except:
        return "ERROR"

def main():
    """Print environment information and check dependencies."""
    print("=" * 60)
    print("EEG EXERGAMING PROJECT - ENVIRONMENT CHECK")
    print("=" * 60)

    # Timestamp
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"Timestamp: {timestamp}")
    print(f"ISO Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
    print()

    # System information
    print("SYSTEM INFORMATION:")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print()

    # Package versions
    packages = [
        'mne', 'numpy', 'scipy', 'pandas', 'sklearn',
        'matplotlib', 'seaborn', 'joblib', 'h5py', 'yaml'
    ]

    print("PACKAGE VERSIONS:")
    for package in packages:
        if package == 'sklearn':
            version = get_package_version('sklearn')
        elif package == 'yaml':
            version = get_package_version('yaml')
        else:
            version = get_package_version(package)
        print(f"{package}: {version}")

    print("\n" + "=" * 60)
    print("Environment check completed.")
    print("=" * 60)

if __name__ == "__main__":
    main()