#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCHDOG="${SCRIPT_DIR}/mac_tailscale_watchdog.sh"
PLIST="${HOME}/Library/LaunchAgents/com.aicoo.tailscale.watchdog.plist"

if [[ ! -f "${WATCHDOG}" ]]; then
  echo "Missing watchdog script: ${WATCHDOG}" >&2
  exit 1
fi

chmod +x "${WATCHDOG}"
mkdir -p "${HOME}/Library/LaunchAgents"
mkdir -p "${HOME}/worker/logs"

cat > "${PLIST}" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.aicoo.tailscale.watchdog</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${WATCHDOG}</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>StartInterval</key>
  <integer>120</integer>
  <key>StandardOutPath</key>
  <string>${HOME}/worker/logs/tailscale_watchdog.stdout.log</string>
  <key>StandardErrorPath</key>
  <string>${HOME}/worker/logs/tailscale_watchdog.stderr.log</string>
</dict>
</plist>
PL

# Prefer modern launchctl bootstrap flow; fallback to legacy load/unload.
if launchctl bootout "gui/$(id -u)" "${PLIST}" >/dev/null 2>&1; then
  :
fi
if launchctl bootstrap "gui/$(id -u)" "${PLIST}" >/dev/null 2>&1; then
  :
else
  launchctl unload "${PLIST}" >/dev/null 2>&1 || true
  launchctl load "${PLIST}"
fi
launchctl kickstart -k "gui/$(id -u)/com.aicoo.tailscale.watchdog" >/dev/null 2>&1 || launchctl start com.aicoo.tailscale.watchdog || true
echo "Installed watchdog: ${PLIST}"
echo "Tail logs: ${HOME}/worker/logs/tailscale_watchdog.log"
