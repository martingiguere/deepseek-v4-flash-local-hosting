#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "=== Building ccp ==="
go build -o ccp .
echo "=== Testing Ollama Cloud backend ==="
CCP_BACKEND=ollama python3 test_ccp.py
echo "=== Testing Bifrost backend ==="
CCP_BACKEND=bifrost python3 test_ccp.py
echo "=== All tests passed ==="