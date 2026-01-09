#!/bin/bash
set -e

INSTALL_DIR="$HOME/.oss-cad-suite"
# URL for the specific release found (2026-01-08) or we could fetch latest.
# Using a fixed recent one is safer than parsing HTML/JSON in shell without jq.
# Since the user is doing this *today* (2026-01-08), this link is perfect.
DOWNLOAD_URL="https://github.com/YosysHQ/oss-cad-suite-build/releases/download/2026-01-08/oss-cad-suite-linux-x64-20260108.tgz"
ARCHIVE_NAME="oss-cad-suite-linux-x64-20260108.tgz"

echo "Downloading OSS CAD Suite..."
wget -c "$DOWNLOAD_URL" -O "$ARCHIVE_NAME"

echo "Extracting to $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR"
# The tarball contains a directory named 'oss-cad-suite', so we extract to home then move or just extract inside.
# Usually it extracts to ./oss-cad-suite.
tar -xzf "$ARCHIVE_NAME" -C "$HOME"

# Clean up
rm "$ARCHIVE_NAME"

echo "Installation complete."
echo "Please add the following to your ~/.bashrc:"
echo "export PATH=\$HOME/oss-cad-suite/bin:\$PATH"
echo "Then run: source ~/.bashrc"
