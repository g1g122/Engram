#!/usr/bin/env bash

# Change this to the directory containing the moa/ package.
ROOT_DIR="path/to/your/MoA_Lee2025_WACV"
EXAMPLE_DIR="${ROOT_DIR}/examples"
CONFIG_PATH="${EXAMPLE_DIR}/example_config.yaml"
LOG_FILE="${EXAMPLE_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${EXAMPLE_DIR}/run.pid"

cd "${ROOT_DIR}"
nohup python -u -m moa.main --config "${CONFIG_PATH}" > "${LOG_FILE}" 2>&1 &
echo "$!" > "${PID_FILE}"

echo "[run] pid=$(cat "${PID_FILE}")"
echo "[run] log=${LOG_FILE}"
