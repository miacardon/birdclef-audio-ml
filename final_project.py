"""
Final Project Entry Point
Runs analysis and regenerates results and figures.
"""

import subprocess
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    print("Running BirdCLEF Final Project analysis...")

    script_path = os.path.join(BASE_DIR, "src", "visualize_training.py")

    # Run the analysis / visualization script
    subprocess.run(
        [sys.executable, script_path],
        check=True
    )

    print("Analysis complete. Results and figures updated.")

if __name__ == "__main__":
    main()
