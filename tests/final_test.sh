#!/usr/bin/env bash
set -Eeuo pipefail

echo "=============================================="
echo " Test run started at: $(date)"
echo " Working directory: $(pwd)"
echo "=============================================="

cleanup_pycache() {
  echo "[INFO] Cleaning __pycache__ directories..."
  find . -type d -name "__pycache__" -print -exec rm -rf {} +
}

run_test() {
  local test_file="$1"
  echo
  echo "----------------------------------------------"
  echo "[INFO] Running: $test_file"
  echo "----------------------------------------------"
  python3 -m pytest -s "$test_file"
  echo "[INFO] Completed: $test_file"
}

trap 'echo "[ERROR] Script failed at line $LINENO"; exit 1' ERR

cleanup_pycache
run_test "tests/test_baseline.py"

cleanup_pycache
run_test "tests/test_51_message.py"

cleanup_pycache
run_test "tests/test_100_message.py"

cleanup_pycache
run_test "tests/test_normalization_regression.py"

cleanup_pycache
run_test "tests/test_online_learning_decay.py"

cleanup_pycache
run_test "tests/test_online_learning_resolves_contextual_reference.py"

cleanup_pycache
run_test "tests/test_dwell_stability.py"

cleanup_pycache
run_test "tests/test_hostile_soft_recovery.py"

cleanup_pycache

echo
echo "=============================================="
echo " ✅ All tests completed successfully"
echo " Finished at: $(date)"
echo "=============================================="
