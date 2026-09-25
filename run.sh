#!/usr/bin/env bash
# Run a pipeline module (or a .py script) under a hard memory cap: an overrun kills
# only this job, never the desktop. Usage: ./run.sh ber.prepare --splits train
# Override with BER_MEM=10G BER_THREADS=12 ./run.sh ...
set -euo pipefail
cd "$(dirname "$0")"
MEM=${BER_MEM:-7G}
THREADS=${BER_THREADS:-8}
if [[ "$1" == *.py ]]; then MODE=(); else MODE=(-m); fi
exec systemd-run --user --scope --quiet \
  -p MemoryMax="$MEM" -p MemorySwapMax=0 -p CPUWeight=50 \
  env PYTHONPATH=business_entity_resolution/src POLARS_MAX_THREADS="$THREADS" \
  OMP_NUM_THREADS="$THREADS" RAYON_NUM_THREADS="$THREADS" \
  nice -n 10 .venv/bin/python "${MODE[@]}" "$@"
