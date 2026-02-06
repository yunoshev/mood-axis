#!/usr/bin/env python3
"""Run the Mood Axis Gradio application.

Usage:
    python scripts/run_app.py
    python scripts/run_app.py --model meta-llama/Llama-3.1-8B-Instruct
    python scripts/run_app.py --share --port 7861
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import DEFAULT_MODEL, AXES_FILE
from src.ui.app import launch_app


def main():
    parser = argparse.ArgumentParser(description="Run the Mood Axis application")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on (default: 7860)",
    )
    args = parser.parse_args()

    # Check if calibration has been run
    if not AXES_FILE.exists():
        print("Warning: Axis vectors not found!")
        print(f"Expected file: {AXES_FILE}")
        print("\nPlease run calibration first:")
        print(f"  python scripts/calibrate.py --model {args.model}")
        print("\nContinuing without mood indicators...")
        print()

    print("=" * 60)
    print("Mood Axis Application")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Port: {args.port}")
    print(f"Share: {args.share}")
    print("=" * 60)

    launch_app(
        model_name=args.model,
        share=args.share,
        server_port=args.port,
    )


if __name__ == "__main__":
    main()
