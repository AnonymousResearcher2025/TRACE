#!/usr/bin/env bash
# Builds the C++ core into python/trace_harness/_trace*.so
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$(command -v "$PY")" \
  -Dpybind11_DIR="$("$PY" -m pybind11 --cmakedir)"
cmake --build build -j"$(nproc)"
echo "built: $(ls python/trace_harness/_trace*.so)"
