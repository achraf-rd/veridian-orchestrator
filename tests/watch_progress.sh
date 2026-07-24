#!/usr/bin/env bash
# Drives a full pipeline run end-to-end, timestamping every SSE line, so
# progress ticks / heartbeats during the multi-minute gen_tc/gen_xosc calls
# are visibly interleaved in real time (not buffered until the node ends).
set -u
ORCH="http://127.0.0.1:8200"
CID="watch-$(date +%s)"

stamp() { while IFS= read -r line; do echo "$(date +%T.%3N) $line"; done; }

echo "=== chat (conv $CID) ==="
curl -sN --max-time 600 -X POST "$ORCH/chat" -H "Content-Type: application/json" \
  -d "{\"conversation_id\":\"$CID\",\"message\":\"The AEB system shall stop the vehicle before a pedestrian crossing at 50 km/h.\\nThe AEB system shall react within 300 ms.\"}" \
  | stamp

for gate in requirements test_cases xosc; do
  echo "=== resume (past $gate) ==="
  curl -sN --max-time 600 -X POST "$ORCH/resume" -H "Content-Type: application/json" \
    -d "{\"conversation_id\":\"$CID\",\"decision\":{\"approved\":true}}" \
    | stamp
done

echo "=== DONE ==="
