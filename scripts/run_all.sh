#!/usr/bin/env bash
# Runs the experiments of Sec. V, one group per machine:
#
#   scripts/run_all.sh memory   c7g.large    memory rows of Tables II-IV, Tables VI-VIII, Sec. V-G
#   scripts/run_all.sh disk     c7gd.large   disk rows of Tables II-IV, Table V, Sec. V-G,
#                                            QVCache fidelity (Sec. V-A)
#   scripts/run_all.sh scale    c7g.4xlarge  Table X, CatapultDB fidelity (Sec. V-A)
#   scripts/run_all.sh e2e      g5.xlarge    Table IX
#   scripts/run_all.sh tables   any machine  prints the tables from results/
#
# Copy results/ between machines so that every group sees the same tuning
# cache (results/tuning/). Disk runs drop the page cache first and execute in
# a transient cgroup-v2 scope with the memory cap of the paper (2 GiB unless
# stated); this needs sudo.
#
# Environment: CONFIG (default configs/aws.yaml), RESULTS (default results),
# DATA_ROOT (default /data/trace, must match data_root in CONFIG).
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${CONFIG:-configs/aws.yaml}
RESULTS=${RESULTS:-results}
DATA_ROOT=${DATA_ROOT:-/data/trace}
PY="$(command -v python3)"
RUN=("$PY" scripts/run.py -c "$CONFIG" --results "$RESULTS")

# cap <GiB> <command...>: runs the command in a cgroup with memory.max = GiB.
cap() {
  local gib=$1
  shift
  sudo systemd-run --scope --quiet -p "MemoryMax=${gib}G" -p MemorySwapMax=0 "$@"
  sudo chown -R "$(id -u):$(id -g)" "$RESULTS"
}

# drop: writes back dirty pages and empties the page cache.
drop() {
  sync
  echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
}

train_darth() {  # train_darth <tier...>: DARTH models at 2x the cold tuned L
  local t
  for t in "$@"; do
    "$PY" scripts/train_darth.py -c "$CONFIG" --tier "$t" --results "$RESULTS"
  done
}

memory() {
  "${RUN[@]}" cost        --tiers mem_video mem_dialog
  train_darth mem_video mem_dialog
  "${RUN[@]}" baselines   --tiers mem_video mem_dialog
  "${RUN[@]}" attribution --tiers mem_video
  "${RUN[@]}" locality    --tiers mem_video
  "${RUN[@]}" dynamics    --tiers mem_video
  "${RUN[@]}" sensitivity --tiers mem_video
  "${RUN[@]}" guarantees  --tiers mem_video mem_dialog
}

disk() {
  drop; cap 2 "${RUN[@]}" cost --tiers disk_video
  train_darth disk_video
  drop; cap 2 "${RUN[@]}" baselines   --tiers disk_video
  drop; cap 2 "${RUN[@]}" attribution --tiers disk_video
  drop; cap 2 "${RUN[@]}" guarantees  --tiers disk_video --limit 5000
  # Table V: one run per caching configuration
  drop; cap 2 "${RUN[@]}" storage --tiers disk_video --label direct_16k
  drop; cap 2 "${RUN[@]}" storage --tiers disk_video --label direct_64k
  drop; cap 2 "${RUN[@]}" storage --tiers disk_video --label buffered_2g
  drop; cap 1 "${RUN[@]}" storage --tiers disk_video --label buffered_1g
  drop;       "${RUN[@]}" storage --tiers disk_video --label buffered_nocap
  # QVCache fidelity: DEEP-10M disk index on the local NVMe
  drop; cap 2 "${RUN[@]}" fidelity --tiers fid_qvcache_deep
}

scale() {
  "${RUN[@]}" scale    --tiers mem_video --threads 16
  "${RUN[@]}" width    --tiers mem_dialog mem_dialog_768 mem_dialog_1536
  # CatapultDB fidelity: 1M x 768-d float32 vectors (3 GB) held in RAM
  "${RUN[@]}" fidelity --tiers fid_catapult_pubmed
}

e2e() {
  "$PY" scripts/e2e_rag.py -c "$CONFIG" --tier disk_dialog \
    --workload "$DATA_ROOT/workloads/dialog_10m" \
    --passages "$DATA_ROOT/wiki/passages.tsv" --results "$RESULTS"
}

tables() {
  "$PY" scripts/make_tables.py --results "$RESULTS" | tee "$RESULTS/tables.txt"
}

case "${1:-}" in
  memory | disk | scale | e2e | tables) "$1" ;;
  *)
    echo "usage: $0 {memory|disk|scale|e2e|tables}" >&2
    exit 2
    ;;
esac
