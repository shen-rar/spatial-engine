#!/usr/bin/env python
"""
Spatial Engine Launcher - One Command to Rule Them All

Runs:
  1. End-to-end verification tests
  2. Backend FastAPI server (port 8000)
  3. Frontend Vite dev server (port 5173)

Usage:
  python launch.py              # Run everything
  python launch.py --test-only  # Run tests only
  python launch.py --no-test    # Skip tests, start servers
"""

import sys
import subprocess
import time
import os
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(msg):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}  {msg}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.END}\n")

def print_step(msg):
    print(f"{Colors.CYAN}{Colors.BOLD}-> {msg}{Colors.END}")

def print_success(msg):
    print(f"{Colors.GREEN}[OK] {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}[FAIL] {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}[WARN] {msg}{Colors.END}")

def run_tests():
    """Run the end-to-end verification suite."""
    print_header("Running End-to-End Verification Tests")

    result = subprocess.run(
        [sys.executable, "tests/test_e2e_verification.py"],
        cwd=Path(__file__).parent,
    )

    if result.returncode == 0:
        print_success("All tests passed!")
        return True
    else:
        print_error("Tests failed!")
        return False

def start_backend():
    """Start the FastAPI backend server."""
    print_step("Starting Backend (FastAPI on http://localhost:8000)")

    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
        cwd=Path(__file__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    # Wait for backend to start
    time.sleep(2)

    if backend_process.poll() is None:
        print_success("Backend started successfully")
        return backend_process
    else:
        print_error("Backend failed to start")
        return None

def start_frontend():
    """Start the Vite frontend dev server."""
    print_step("Starting Frontend (Vite on http://localhost:5173)")

    frontend_dir = Path(__file__).parent / "frontend"

    # On Windows, npm needs shell=True to be found in PATH
    is_windows = sys.platform == "win32"

    # Check if node_modules exists
    if not (frontend_dir / "node_modules").exists():
        print_warning("node_modules not found. Running npm install...")
        install_result = subprocess.run(
            ["npm", "install"],
            cwd=frontend_dir,
            shell=is_windows,
        )
        if install_result.returncode != 0:
            print_error("npm install failed")
            return None

    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        shell=is_windows,
    )

    # Wait for frontend to start
    time.sleep(3)

    if frontend_process.poll() is None:
        print_success("Frontend started successfully")
        return frontend_process
    else:
        print_error("Frontend failed to start")
        return None

def main():
    print_header("3D Spatial Engine - Launch Suite")

    # Parse arguments
    args = sys.argv[1:]
    test_only = "--test-only" in args
    no_test = "--no-test" in args

    # Step 1: Run tests (unless --no-test)
    if not no_test:
        tests_passed = run_tests()
        if not tests_passed:
            print_error("Tests failed. Fix issues before starting servers.")
            sys.exit(1)

        if test_only:
            print_success("Tests complete. Exiting (--test-only flag).")
            sys.exit(0)

    # Step 2: Start backend
    backend_process = start_backend()
    if not backend_process:
        sys.exit(1)

    # Step 3: Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        backend_process.terminate()
        sys.exit(1)

    # Print success message
    print_header("Spatial Engine Running")
    print(f"{Colors.GREEN}Backend API:{Colors.END}  http://localhost:8000")
    print(f"{Colors.GREEN}API Docs:{Colors.END}     http://localhost:8000/docs")
    print(f"{Colors.GREEN}Frontend:{Colors.END}     http://localhost:5173")
    print(f"\n{Colors.YELLOW}Press Ctrl+C to stop all services{Colors.END}\n")

    # Keep running until Ctrl+C
    try:
        while True:
            time.sleep(1)

            # Check if processes are still alive
            if backend_process.poll() is not None:
                print_error("Backend process died unexpectedly")
                break
            if frontend_process.poll() is not None:
                print_error("Frontend process died unexpectedly")
                break

    except KeyboardInterrupt:
        print_header("Shutting Down")
        print_step("Stopping backend...")
        backend_process.terminate()
        print_step("Stopping frontend...")
        frontend_process.terminate()

        # Wait for clean shutdown
        backend_process.wait(timeout=5)
        frontend_process.wait(timeout=5)

        print_success("All services stopped")
        sys.exit(0)

if __name__ == "__main__":
    main()
