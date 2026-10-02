"""
run.py
------
Convenience launcher for the Student Performance Analyzer Streamlit application.
Ensures modules are accessible and starts Streamlit directly with native Python.
"""

import os
import sys
import subprocess

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(project_dir, "app.py")
    
    print("=" * 65)
    print("🚀 Launching EduMetrics AI - Student Performance Analyzer")
    print("=" * 65)
    print(f"Project Directory : {project_dir}")
    print(f"Streamlit App     : {app_path}")
    print("\nStarting local server on http://localhost:8501 ...")
    print("Press Ctrl + C in this terminal to stop the server.\n")
    
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[INFO] Application stopped by user.")

if __name__ == "__main__":
    main()
