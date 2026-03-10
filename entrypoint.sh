#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] Starting YOLO training entrypoint"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "[entrypoint] Environment variable '$name' is required but not set" >&2
    exit 1
  fi
}

require_env "ROBOFLOW_DATASET_URL"
require_env "RUN_NAME"

# Optional overrides with sensible defaults
export DATA_ROOT="${DATA_ROOT:-/workspace/data}"
export MODEL_NAME="${MODEL_NAME:-yolo9-t}"
export IMGSZ="${IMGSZ:-640}"
export BATCH="${BATCH:-16}"
export EPOCHS="${EPOCHS:-100}"
export WORKERS="${WORKERS:-8}"
export PROJECT_DIR="${PROJECT_DIR:-/workspace/runs}"

mkdir -p "${DATA_ROOT}" /workspace/output "${PROJECT_DIR}"

echo "[entrypoint] Downloading dataset..."
python /workspace/download_dataset.py

# Try to infer data.yaml path if not explicitly set
if [[ -z "${DATA_YAML_PATH:-}" ]]; then
  CANDIDATE=$(python - << 'EOF'
from pathlib import Path
root = Path("/workspace/data/dataset")
candidate = next(root.rglob("data.yaml"), None)
print(candidate if candidate else "")
EOF
)
  if [[ -n "${CANDIDATE}" ]]; then
    export DATA_YAML_PATH="${CANDIDATE}"
  else
    echo "[entrypoint] WARNING: Could not automatically locate data.yaml; using default /workspace/data/dataset/data.yaml" >&2
    export DATA_YAML_PATH="/workspace/data/dataset/data.yaml"
  fi
fi

echo "[entrypoint] Training model '${MODEL_NAME}' with data '${DATA_YAML_PATH}'"
python /workspace/train_yolo.py

TAR_PATH="/workspace/output/${RUN_NAME}.tar.gz"
echo "[entrypoint] Packaging training artifacts into ${TAR_PATH}"

tar -czf "${TAR_PATH}" -C /workspace runs || {
  echo "[entrypoint] WARNING: Failed to create tarball from /workspace/runs" >&2
}

echo "[entrypoint] Entry point completed successfully"

