#!/usr/bin/env bash
set -euo pipefail

echo "====================================="
echo " Evaluating All Heimdall Benchmarks"
echo "====================================="

cd benchmarks

echo ""
echo "→ Evaluating stateless"
python3 evaluate_stateless_gold.py

echo ""
echo "→ Evaluating stateful"
python3 evaluate_stateful_gold.py

echo ""
echo "→ Evaluating stateful adversarial"
python3 evaluate_stateful_adversarial.py

echo ""
echo "✓ All evaluations complete."
