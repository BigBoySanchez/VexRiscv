#!/usr/bin/env bash
# build_flash_resnet50.sh — Assemble a reproducible SPI-flash image for ResNet-50 on iCEBreaker
#
# Flash layout (iCEBreaker 16 MiB SPI flash):
#
#   Offset         Size (max)   Content
#   ──────────────────────────────────────────────────────────────────────────
#   0x000000  (0 MiB)  ~1 MiB    FPGA bitstream  (soc.bit / toplevel.bin)
#   0x100000  (1 MiB)  15 MiB    BD weight blob  (weights_bd.bin, VWB2 format)
#   ──────────────────────────────────────────────────────────────────────────
#   0x1000000(16 MiB)             end of flash
#
# Usage:
#   build_flash_resnet50.sh [OPTIONS]
#
# Options:
#   --bitstream FILE   FPGA bitstream (default: scripts/MuraxHyperRAM/iCEBreaker/bin/toplevel.bin)
#   --weights   FILE   BD weight blob (default: scripts/resnet50_artifacts/weights_bd.bin)
#   --out       FILE   Output flash image (default: scripts/resnet50_artifacts/flash_image.bin)
#   --prog             Program flash via iceprog after building
#   --prog-weights-only  Only reprogram weights region (faster iteration)
#   --dry-run          Print layout and size check; do not write flash image
#   -h, --help         Show this help
#
# The firmware is already baked into the FPGA bitstream via Spinal/BRAM init symbols
# (MuraxHyperRAM.v*_ram_symbol*.bin), so no separate firmware binary is needed.
#
# Reproducibility: with fixed inputs the output is byte-identical every run.

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BITSTREAM="${SCRIPT_DIR}/MuraxHyperRAM/iCEBreaker/bin/toplevel.bin"
WEIGHTS="${SCRIPT_DIR}/resnet50_artifacts/weights_bd.bin"
OUT="${SCRIPT_DIR}/resnet50_artifacts/flash_image.bin"
DO_PROG=0
PROG_WEIGHTS_ONLY=0
DRY_RUN=0

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bitstream)  BITSTREAM="$2";         shift 2 ;;
        --weights)    WEIGHTS="$2";           shift 2 ;;
        --out)        OUT="$2";               shift 2 ;;
        --prog)       DO_PROG=1;              shift   ;;
        --prog-weights-only) PROG_WEIGHTS_ONLY=1; shift ;;
        --dry-run)    DRY_RUN=1;              shift   ;;
        -h|--help)    grep "^#" "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Flash layout constants ─────────────────────────────────────────────────────
FLASH_SIZE_BYTES=$((16 * 1024 * 1024))   #  16 MiB total
BITSTREAM_OFFSET=0                        #   0 MiB  — bitstream
BITSTREAM_MAX=$((1 * 1024 * 1024))       #   1 MiB  — max bitstream size
WEIGHTS_OFFSET=$((1 * 1024 * 1024))      #   1 MiB  — weights (= flashOffset in MuraxHyperRAM)
WEIGHTS_REGION=$(( FLASH_SIZE_BYTES - WEIGHTS_OFFSET ))  # 15 MiB available for weights

# ── Helper ────────────────────────────────────────────────────────────────────
hr() { printf '─%.0s' {1..70}; echo; }
human() {
    local b=$1
    python3 -c "
b=$b
if b>=1048576: print(f'{b/1048576:.2f} MiB')
elif b>=1024:  print(f'{b/1024:.1f} KiB')
else:          print(f'{b} B')
"
}
pct() {
    python3 -c "print(f'{$1*100/$2:.1f}%')"
}

hr
echo "ResNet-50 Flash Image Builder"
echo "Repo: ${REPO_ROOT}"
hr

# ── Check input files ──────────────────────────────────────────────────────────
ANY_MISSING=0

file_size() {
    # Print file size in bytes, or -1 if missing
    local path="$1"
    if [[ -f "$path" ]]; then wc -c < "$path"; else echo -1; fi
}

echo "Input files:"
BITSTREAM_SIZE=$(file_size "$BITSTREAM")
WEIGHTS_SIZE=$(file_size "$WEIGHTS")

for pair in "bitstream:$BITSTREAM:$BITSTREAM_SIZE" "weights:$WEIGHTS:$WEIGHTS_SIZE"; do
    label="${pair%%:*}"; rest="${pair#*:}"; path="${rest%%:*}"; sz="${rest##*:}"
    if [[ "$sz" == "-1" ]]; then
        printf "  %-12s  %-55s  MISSING\n" "$label" "$path" >&2
        ANY_MISSING=1
    else
        printf "  %-12s  %-55s  %s\n" "$label" "$path" "$(human "$sz")"
    fi
done
echo

if [[ $ANY_MISSING -eq 1 ]]; then
    echo "ERROR: one or more input files are missing." >&2
    echo "  Build the bitstream with:  make -C scripts/MuraxHyperRAM/iCEBreaker compile" >&2
    echo "  Build the weights with:    python scripts/gen_resnet50_model.py" >&2
    exit 1
fi

# ── Size checks ───────────────────────────────────────────────────────────────
echo "Flash layout:"
hr
printf "  %-10s  %-12s  %-12s  %s\n" "Offset" "Max size" "Actual" "Content"
hr
printf "  0x%06X  %-12s  %-12s  FPGA bitstream (firmware baked in)\n" \
    "$BITSTREAM_OFFSET" "$(human $BITSTREAM_MAX)" "$(human $BITSTREAM_SIZE)"
printf "  0x%06X  %-12s  %-12s  BD weight blob (VWB2)\n" \
    "$WEIGHTS_OFFSET" "$(human $WEIGHTS_REGION)" "$(human $WEIGHTS_SIZE)"
printf "  0x%06X  (end of flash)\n" "$FLASH_SIZE_BYTES"
hr
echo

ERRORS=0

if (( BITSTREAM_SIZE > BITSTREAM_MAX )); then
    echo "ERROR: bitstream is $(human $BITSTREAM_SIZE), exceeds $(human $BITSTREAM_MAX) limit." >&2
    ERRORS=1
fi

if (( WEIGHTS_SIZE > WEIGHTS_REGION )); then
    echo "ERROR: weights are $(human $WEIGHTS_SIZE), exceeds $(human $WEIGHTS_REGION) available after 1 MiB offset." >&2
    ERRORS=1
fi

TOTAL_USED=$(( WEIGHTS_OFFSET + WEIGHTS_SIZE ))
printf "  Total flash used  : %s / %s  (%s)\n" \
    "$(human $TOTAL_USED)" "$(human $FLASH_SIZE_BYTES)" \
    "$(pct $TOTAL_USED $FLASH_SIZE_BYTES)"
echo

if [[ $ERRORS -ne 0 ]]; then
    echo "Aborting due to size errors." >&2
    exit 1
fi

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] Layout OK — no output written."
    exit 0
fi

# ── Build flash_image.bin ──────────────────────────────────────────────────────
OUT_DIR="$(dirname "$OUT")"
mkdir -p "$OUT_DIR"

echo "Building flash image: $OUT"

# Create a zeroed 16 MiB image
dd if=/dev/zero bs=1M count=16 of="$OUT" status=none

# Write bitstream at offset 0
dd if="$BITSTREAM" of="$OUT" bs=1 conv=notrunc status=none
echo "  [✓] bitstream @ 0x000000"

# Write weights at 1 MiB offset
dd if="$WEIGHTS" of="$OUT" bs=1 seek="$WEIGHTS_OFFSET" conv=notrunc status=none
echo "  [✓] weights   @ 0x100000"

FINAL_SIZE=$(wc -c < "$OUT")
echo "  Output: $(human $FINAL_SIZE)"
echo

# ── SHA-256 manifest ───────────────────────────────────────────────────────────
MANIFEST="${OUT%.bin}.manifest"
{
    echo "flash_image_manifest"
    echo "generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "flash_size_bytes: $FLASH_SIZE_BYTES"
    echo ""
    echo "bitstream_offset: $BITSTREAM_OFFSET"
    echo "bitstream_file:   $BITSTREAM"
    echo "bitstream_bytes:  $BITSTREAM_SIZE"
    echo "bitstream_sha256: $(sha256sum "$BITSTREAM" | awk '{print $1}')"
    echo ""
    echo "weights_offset:   $WEIGHTS_OFFSET"
    echo "weights_file:     $WEIGHTS"
    echo "weights_bytes:    $WEIGHTS_SIZE"
    echo "weights_sha256:   $(sha256sum "$WEIGHTS" | awk '{print $1}')"
    echo ""
    echo "flash_image_sha256: $(sha256sum "$OUT" | awk '{print $1}')"
} > "$MANIFEST"
echo "Manifest: $MANIFEST"
cat "$MANIFEST"
echo

# ── Optional programming ───────────────────────────────────────────────────────
if [[ $DO_PROG -eq 1 ]]; then
    hr
    echo "Programming full flash image (will take ~2 min)..."
    if ! command -v iceprog &>/dev/null; then
        echo "ERROR: iceprog not found in PATH." >&2
        exit 1
    fi
    iceprog "$OUT"
    echo "[✓] Programming complete."
elif [[ $PROG_WEIGHTS_ONLY -eq 1 ]]; then
    hr
    echo "Programming weights only @ 1 MiB offset (faster iteration)..."
    if ! command -v iceprog &>/dev/null; then
        echo "ERROR: iceprog not found in PATH." >&2
        exit 1
    fi
    iceprog -o 1M "$WEIGHTS"
    echo "[✓] Weights programmed."
fi

hr
echo "Done."
echo
echo "To program the full flash:          $0 --prog"
echo "To reprogram weights only:          $0 --prog-weights-only"
echo "To verify layout without building:  $0 --dry-run"
