#!/usr/bin/env bash
set -euo pipefail

echo "====================================="
echo " Building All Heimdall Benchmarks"
echo "====================================="

cd benchmarks

echo ""
echo "→ Building stateless gold"
python3 build_stateless_gold.py

echo ""
echo "→ Building stateful gold"
python3 build_stateful_gold.py

echo ""
echo "→ Building stateful adversarial gold"
python3 build_stateful_adversarial.py

echo ""
echo "✓ All benchmark datasets built."
