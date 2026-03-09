#!/bin/bash
set -euo pipefail

# Tailscale watchdog for Mac remote nodes.
# - checks backend + engine health
# - optional peer ping probe
# - applies staged recovery with cooldown and state tracking

LOG_DIR="${AICOO_LOG_DIR:-${HOME}/worker/logs}"
LOG_FILE="${LOG_DIR}/tailscale_watchdog.log"
STATE_FILE="${LOG_DIR}/tailscale_watchdog.state"
LOCK_DIR="${LOG_DIR}/tailscale_watchdog.lock"
COOLDOWN_SEC="${AICOO_WATCHDOG_COOLDOWN_SEC:-180}"
TARGET_IP="${AICOO_WATCH_TARGET_IP:-}"
mkdir -p "${LOG_DIR}"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" >> "${LOG_FILE}"; }
now_epoch() { date +%s; }
find_tailscale_bin() {
  if command -v tailscale >/dev/null 2>&1; then
    command -v tailscale
    return 0
  fi
  for p in \
    /opt/homebrew/bin/tailscale \
    /usr/local/bin/tailscale \
    /Applications/Tailscale.app/Contents/MacOS/Tailscale
  do
    if [[ -x "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

if ! mkdir "${LOCK_DIR}" >/dev/null 2>&1; then
  log "lock_busy: another watchdog instance is running"
  exit 0
fi
trap 'rmdir "${LOCK_DIR}" >/dev/null 2>&1 || true' EXIT

FAILURES=0
LAST_RECOVERY=0
if [[ -f "${STATE_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${STATE_FILE}" || true
fi

write_state() {
  cat > "${STATE_FILE}" <<EOF
FAILURES=${FAILURES}
LAST_RECOVERY=${LAST_RECOVERY}
EOF
}

kickstart_daemon() {
  # Prefer non-interactive sudo if available, then fallback to plain launchctl.
  /usr/bin/sudo -n /bin/launchctl kickstart -k system/com.tailscale.tailscaled >/dev/null 2>&1 && return 0
  /bin/launchctl kickstart -k system/com.tailscale.tailscaled >/dev/null 2>&1 && return 0
  return 1
}

reconnect() {
  "${TS_BIN}" up --ssh >/dev/null 2>&1 || true
}

in_cooldown() {
  local now
  now="$(now_epoch)"
  [[ $(( now - LAST_RECOVERY )) -lt "${COOLDOWN_SEC}" ]]
}

TS_BIN="$(find_tailscale_bin || true)"
if [[ -z "${TS_BIN}" ]]; then
  log "tailscale binary not found; skipping"
  exit 0
fi

JSON="$("${TS_BIN}" status --json 2>/dev/null || true)"
if [[ -z "${JSON}" ]]; then
  log "status_json_empty: attempting tailscaled restart"
  kickstart_daemon || log "kickstart_failed"
  exit 0
fi

HEALTH_FIELDS="$(
  TS_JSON="${JSON}" /usr/bin/python3 -c 'import json,os
try:
    d = json.loads(os.environ["TS_JSON"])
    print(d.get("BackendState",""))
    print("1" if d.get("Self",{}).get("InEngine", False) else "0")
except Exception:
    print("")
    print("0")
'
)"
BACKEND_STATE="$(printf '%s\n' "${HEALTH_FIELDS}" | /usr/bin/sed -n '1p')"
IN_ENGINE="$(printf '%s\n' "${HEALTH_FIELDS}" | /usr/bin/sed -n '2p')"
IN_ENGINE="${IN_ENGINE:-0}"

IPV4="$("${TS_BIN}" ip -4 2>/dev/null | head -n 1 || true)"
PEER_OK="1"
if [[ -n "${TARGET_IP}" ]]; then
  if "${TS_BIN}" ping -c 1 --timeout 5s "${TARGET_IP}" >/dev/null 2>&1; then
    PEER_OK="1"
  else
    PEER_OK="0"
  fi
fi

if [[ "${BACKEND_STATE}" == "Running" && -n "${IPV4}" && "${PEER_OK}" == "1" ]]; then
  FAILURES=0
  write_state
  log "healthy backend=${BACKEND_STATE} in_engine=${IN_ENGINE} ip=${IPV4}"
  exit 0
fi

FAILURES=$((FAILURES + 1))
log "unhealthy failures=${FAILURES} backend=${BACKEND_STATE} in_engine=${IN_ENGINE} ip=${IPV4:-none} peer_ok=${PEER_OK}"

if in_cooldown; then
  write_state
  log "cooldown_active sec=${COOLDOWN_SEC}; skipping recovery"
  exit 0
fi

LAST_RECOVERY="$(now_epoch)"
if [[ "${FAILURES}" -le 2 ]]; then
  # First line recovery: refresh local session.
  reconnect
  log "recovery_action=tailscale_up"
else
  # Escalation recovery: restart daemon then reconnect.
  kickstart_daemon || log "kickstart_failed"
  reconnect
  log "recovery_action=daemon_kickstart_plus_up"
fi
sleep 3

NEW_IP="$("${TS_BIN}" ip -4 2>/dev/null | head -n 1 || true)"
POST_PEER_OK="1"
if [[ -n "${TARGET_IP}" ]]; then
  if "${TS_BIN}" ping -c 1 --timeout 5s "${TARGET_IP}" >/dev/null 2>&1; then
    POST_PEER_OK="1"
  else
    POST_PEER_OK="0"
  fi
fi
log "post_recovery ip=${NEW_IP:-none} peer_ok=${POST_PEER_OK}"
write_state
