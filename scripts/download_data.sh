#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
# Install the optional downloader with: python -m pip install kaggle
# Configure Kaggle authentication locally before running this script.
# download_data.py runs the exact Kaggle command:
# python -m kaggle datasets download -d ipythonx/mvtec-ad -p data/
# It then extracts the five configured categories into data/mvtec_ad/.
# Unlike --unzip, this avoids extracting the other ten categories.
python scripts/download_data.py "$@"
