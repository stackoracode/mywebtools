#!/usr/bin/env bash
# Mount the Windows Duix folder (C:\AI\Duix\data\face2face\temp shared over SMB) on this board.
#
# Usage: sudo deploy/mount-duix-share.sh //192.168.8.106/duix_temp <windows-user> [/mnt/duix_temp]
#
# The mount is on-demand (systemd automount) and "soft", so the board boots and the
# web UI keeps working while the Windows PC is switched off.
set -euo pipefail

SHARE="${1:-//192.168.8.106/duix_temp}"
WIN_USER="${2:?windows user name required}"
MOUNT="${3:-/mnt/duix_temp}"
LOCAL_USER="${SUDO_USER:-$USER}"
CRED=/etc/mywebtools-smb.cred

if [ "$(id -u)" != 0 ]; then echo "run with sudo" >&2; exit 1; fi
command -v mount.cifs >/dev/null || apt-get install -y cifs-utils

read -r -s -p "Password for $WIN_USER on $SHARE: " WIN_PASS; echo
umask 077
printf 'username=%s\npassword=%s\n' "$WIN_USER" "$WIN_PASS" > "$CRED"
chmod 600 "$CRED"

mkdir -p "$MOUNT"
UID_N="$(id -u "$LOCAL_USER")"; GID_N="$(id -g "$LOCAL_USER")"
OPTS="credentials=$CRED,uid=$UID_N,gid=$GID_N,iocharset=utf8,vers=3.0,file_mode=0664,dir_mode=0775,soft,nofail,_netdev,x-systemd.automount,x-systemd.idle-timeout=600,x-systemd.mount-timeout=10"
LINE="$SHARE $MOUNT cifs $OPTS 0 0"

if grep -qE "[[:space:]]$MOUNT[[:space:]]" /etc/fstab; then
  sed -i "\#[[:space:]]$MOUNT[[:space:]]#d" /etc/fstab
fi
echo "$LINE" >> /etc/fstab
systemctl daemon-reload
systemctl restart "$(systemd-escape -p --suffix=automount "$MOUNT")" 2>/dev/null || mount "$MOUNT" || true

if touch "$MOUNT/.mwt_test" 2>/dev/null && rm -f "$MOUNT/.mwt_test"; then
  echo "OK: $SHARE is writable at $MOUNT"
else
  echo "Mounted entry added, but $MOUNT is not writable yet (is the PC on and the share created?)."
fi
echo "In the web UI: Admin -> Engine -> 'Path to that folder on this server' = $MOUNT (default)."
