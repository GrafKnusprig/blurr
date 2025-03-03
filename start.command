#!/bin/bash
# Activate the virtual environment and run blurr.py

# Change to the directory where this script and your venv are located.
cd "$(dirname "$0")"

source .venv/bin/activate
python blurr.py