#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "=== Omega Poker Setup ==="

# Python deps
pip install -r requirements.txt

# Optional: build TexasSolver-Console (C++ postflop solver)
# Comment out if you don't need accurate postflop solver (heuristic fallback is used)
if command -v cmake &>/dev/null && command -v g++ &>/dev/null; then
  echo "--- Building TexasSolver-Console ---"
  if [ ! -d "TexasSolver" ]; then
    git clone --depth=1 https://github.com/bupticybee/TexasSolver.git
  fi
  cd TexasSolver
  cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
  cmake --build build --config Release --target console_solver -j4
  cp build/console_solver .
  cd ..
  echo "--- TexasSolver built at ./TexasSolver/console_solver ---"
else
  echo "--- cmake/g++ not found; skipping TexasSolver build (heuristic fallback active) ---"
fi

mkdir -p data/hand_histories

echo ""
echo "=== Setup complete ==="
echo "Run with:  python main.py"
echo "Then open: http://localhost:8765"
