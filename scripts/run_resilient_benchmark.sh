#!/usr/bin/env bash
set -euo pipefail

# Retry only transient infrastructure/API failures. Benchmark quality failures are
# enforced in a separate workflow step and are never retried or weakened here.
MAX_ATTEMPTS="${AI_LCA_INFRA_RETRY_ATTEMPTS:-3}"
BASE_DELAY="${AI_LCA_INFRA_RETRY_BASE_DELAY_SECONDS:-15}"
LOG_PATH="${AI_LCA_INFRA_RETRY_LOG:-benchmark_infra_attempt.log}"

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <command> [args ...]" >&2
  exit 2
fi

is_transient_failure() {
  grep -Eqi \
    'APITimeoutError|APIConnectionError|ReadTimeout|ConnectTimeout|RemoteProtocolError|RemoteDisconnected|RateLimitError|InternalServerError|Connection reset|Connection aborted|temporarily unavailable|server disconnected' \
    "$LOG_PATH"
}

attempt=1
while true; do
  : > "$LOG_PATH"
  echo "Infrastructure attempt ${attempt}/${MAX_ATTEMPTS}: $*"

  set +e
  "$@" 2>&1 | tee "$LOG_PATH"
  status=${PIPESTATUS[0]}
  set -e

  if [ "$status" -eq 0 ]; then
    exit 0
  fi

  if ! is_transient_failure; then
    echo "Command failed with a non-transient error; not retrying." >&2
    exit "$status"
  fi

  if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
    echo "Transient failure persisted for ${MAX_ATTEMPTS} attempts." >&2
    exit "$status"
  fi

  delay=$((BASE_DELAY * attempt))
  echo "Transient API/network failure detected; retrying in ${delay}s." >&2
  sleep "$delay"
  attempt=$((attempt + 1))
done
