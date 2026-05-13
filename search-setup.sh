#!/usr/bin/env bash
# Installs (or uninstalls) the GNOME Shell search provider for the current user.
#
# Drops three files under ~/.local with the path to the locally-built binary
# substituted in. Run after `cargo build --release`.
#
# Requires a GNOME Shell session restart (logout / login on Wayland) for
# GNOME to pick up changes to the search-provider .ini.
#
# Usage:
#   ./search-setup.sh             install
#   ./search-setup.sh --uninstall remove the three files and kill any running daemon
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${REPO_ROOT}/target/release/ubuntu-desktop-help"

DATA_HOME="${XDG_DATA_HOME:-${HOME}/.local/share}"
SP_FILE="${DATA_HOME}/gnome-shell/search-providers/com.canonical.UbuntuDesktopHelp.search-provider.ini"
DBUS_FILE="${DATA_HOME}/dbus-1/services/com.canonical.UbuntuDesktopHelp.service"
DESKTOP_FILE="${DATA_HOME}/applications/com.canonical.UbuntuDesktopHelp.desktop"

uninstall() {
  # Ask any running daemon to drop its bus name so the next overview search
  # re-activates the binary we're about to remove. Best-effort.
  pkill -f "ubuntu-desktop-help.*dbus" 2>/dev/null || true

  rm -f "${SP_FILE}" "${DBUS_FILE}" "${DESKTOP_FILE}"

  if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$(dirname "${DESKTOP_FILE}")" >/dev/null 2>&1 || true
  fi

  cat <<EOF
Removed:
  ${SP_FILE}
  ${DBUS_FILE}
  ${DESKTOP_FILE}

Log out and back in to make GNOME Shell forget the search provider.
EOF
}

install() {
  if [[ ! -x "${BINARY}" ]]; then
    echo "error: ${BINARY} not found — run \`cargo build --release\` first" >&2
    exit 1
  fi

  mkdir -p "$(dirname "${SP_FILE}")" "$(dirname "${DBUS_FILE}")" "$(dirname "${DESKTOP_FILE}")"

  # Kill any previously-installed daemon so the new .service file is honoured.
  pkill -f "ubuntu-desktop-help.*dbus" 2>/dev/null || true

  # Substitute @BINARY@ with the absolute path to the built binary.
  sed "s|@BINARY@|${BINARY}|g" \
    "${REPO_ROOT}/data/com.canonical.UbuntuDesktopHelp.service.in" \
    > "${DBUS_FILE}"

  sed "s|@BINARY@|${BINARY}|g" \
    "${REPO_ROOT}/data/com.canonical.UbuntuDesktopHelp.desktop.in" \
    > "${DESKTOP_FILE}"

  # The .ini file has no substitutions; copy as-is.
  cp "${REPO_ROOT}/data/com.canonical.UbuntuDesktopHelp.search-provider.ini" "${SP_FILE}"

  if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$(dirname "${DESKTOP_FILE}")" >/dev/null 2>&1 || true
  fi

  cat <<EOF
Installed:
  ${DBUS_FILE}
  ${DESKTOP_FILE}
  ${SP_FILE}

Log out and back in so GNOME Shell picks up the new search provider, then try:
  ?? how do I change my wallpaper
EOF
}

case "${1:-}" in
  --uninstall|-u|uninstall) uninstall ;;
  ""|--install|install)     install ;;
  *) echo "usage: $0 [--uninstall]" >&2; exit 2 ;;
esac
