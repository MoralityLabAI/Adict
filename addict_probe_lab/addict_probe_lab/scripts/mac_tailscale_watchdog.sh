#!/bin/bash
set -euo pipefail

# Lightweight watchdog for Mac remote nodes.
# - verifies tailscaled backend status
# - verifies node has a Tailscale IPv4
# - attempts local recovery if unhealthy

LOG_DIR="${HOME}/worker/logs"
LOG_FILE="${LOG_DIR}/tailscale_watchdog.log"
mkdir -p "${LOG_DIR}"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" >> "${LOG_FILE}"; }
kickstart_daemon() {
  # Prefer non-interactive sudo if available, then fallback to plain launchctl.
  /usr/bin/sudo -n /bin/launchctl kickstart -k system/com.tailscale.tailscaled >/dev/null 2>&1 && return 0
  /bin/launchctl kickstart -k system/com.tailscale.tailscaled >/dev/null 2>&1 && return 0
  return 1
}

if ! command -v tailscale >/dev/null 2>&1; then
  log "tailscale binary not found; skipping"
  exit 0
fi

JSON="$(tailscale status --json 2>/dev/null || true)"
if [[ -z "${JSON}" ]]; then
  log "status_json_empty: attempting tailscaled restart"
  kickstart_daemon || log "kickstart_failed"
  exit 0
fi

readarray -t HEALTH_FIELDS < <(
  TS_JSON="${JSON}" /usr/bin/python3 -c 'import json,os
try:
    d = json.loads(os.environ["TS_JSON"])
    print(d.get("BackendState",""))
    print("1" if d.get("Self",{}).get("InEngine", False) else "0")
except Exception:
    print("")
    print("0")
'
)
BACKEND_STATE="${HEALTH_FIELDS[0]:-}"
IN_ENGINE="${HEALTH_FIELDS[1]:-0}"

IPV4="$(tailscale ip -4 2>/dev/null | head -n 1 || true)"

if [[ "${BACKEND_STATE}" != "Running" || -z "${IPV4}" || "${IN_ENGINE}" != "1" ]]; then
  log "unhealthy backend=${BACKEND_STATE} in_engine=${IN_ENGINE} ip=${IPV4:-none}; recovering"
  kickstart_daemon || log "kickstart_failed"
  tailscale up --ssh >/dev/null 2>&1 || true
  sleep 2
  NEW_IP="$(tailscale ip -4 2>/dev/null | head -n 1 || true)"
  log "post_recovery ip=${NEW_IP:-none}"
else
  log "healthy backend=${BACKEND_STATE} in_engine=${IN_ENGINE} ip=${IPV4}"
fi
