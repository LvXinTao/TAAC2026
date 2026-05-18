#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- EDA mode: analyze dataset and output to log ----
python3 -u "${SCRIPT_DIR}/eda.py" \
    "$@"
