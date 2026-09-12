#!/usr/bin/env bash
# Synthetic end-to-end demo: no API keys, no cost.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m evalharness demo
