#!/usr/bin/env bash
# Run preprocessing for all dataset configs under configs/datasets for a given server.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <server>"
  exit 1
fi

SERVER="$1"

CONFIG_DIR="configs/datasets"
if [[ ! -d "${CONFIG_DIR}" ]]; then
  echo "Config directory not found: ${CONFIG_DIR}"
  exit 1
fi

for config in "${CONFIG_DIR}"/*.yaml; do
  [ -e "$config" ] || continue
  echo ">>> Preprocessing ${config} for server ${SERVER}"
  uv run python scripts/preprocess.py --server "${SERVER}" --config "${config}"
done
