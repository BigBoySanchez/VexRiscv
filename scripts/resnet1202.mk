# resnet1202.mk — End-to-end build + flash for ResNet-1202 on iCEBreaker
#
# Flash layout (16 MiB SPI flash):
#   0x000000  bitstream   ≤1 MiB   (firmware baked into BRAM via onChipRamHexFile)
#   0x100000  weights    ~11 MiB   (VWB2 BD4 blob, ResNet-1202 CIFAR-10)
#
# Usage (run from repo root):
#   make -f scripts/resnet1202.mk            # build firmware + bitstream + flash image
#   make -f scripts/resnet1202.mk prog       # build everything and program full flash
#   make -f scripts/resnet1202.mk prog_bitstream  # reflash bitstream only (firmware update)
#   make -f scripts/resnet1202.mk prog_weights    # reflash weights only (~60 min)
#   make -f scripts/resnet1202.mk dry_run    # size/layout check, no output written
#   make -f scripts/resnet1202.mk clean      # remove all build artefacts

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT      := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))/../
FIRMWARE_DIR   := $(REPO_ROOT)/src/main/c/murax/resnet1202_phase3_hw_decode
BITSTREAM_DIR  := $(REPO_ROOT)/scripts/MuraxHyperRAM/iCEBreaker
ARTIFACTS      := $(REPO_ROOT)/scripts/resnet1202_artifacts

FIRMWARE_HEX   := $(FIRMWARE_DIR)/build/resnet1202_phase3_hw_decode.hex
FIRMWARE_BIN   := $(FIRMWARE_DIR)/build/resnet1202_phase3_hw_decode.bin
BITSTREAM      := $(BITSTREAM_DIR)/bin/toplevel.bin
WEIGHTS        := $(ARTIFACTS)/weights_bd.bin
FLASH_IMAGE    := $(ARTIFACTS)/flash_image.bin
MANIFEST       := $(ARTIFACTS)/flash_image.manifest

FLASH_SIZE     := 16777216
WEIGHTS_OFFSET := 1048576

# ── Tools ──────────────────────────────────────────────────────────────────────
ICEPROG  ?= iceprog
SBT      ?= sbt
PYTHON3  ?= python3
RISCV_CC ?= $(HOME)/tools/xpack-riscv-none-elf-gcc-13.2.0-2/bin/riscv-none-elf-gcc

# ── Default target ─────────────────────────────────────────────────────────────
.PHONY: all
all: $(FLASH_IMAGE)

# ── 1. Firmware (.hex baked into bitstream) ────────────────────────────────────
$(FIRMWARE_HEX): $(wildcard $(FIRMWARE_DIR)/src/*.c) \
                 $(wildcard $(FIRMWARE_DIR)/src/*.S) \
                 $(wildcard $(FIRMWARE_DIR)/src/*.h)
	$(MAKE) -C $(FIRMWARE_DIR) -j$$(nproc)

.PHONY: firmware
firmware: $(FIRMWARE_HEX)

# ── 2. Weights blob ────────────────────────────────────────────────────────────
$(WEIGHTS):
	@echo "[weights] generating ResNet-1202 BD4 weight blob (tap-blocked layout)..."
	cd $(REPO_ROOT) && $(PYTHON3) scripts/gen_resnet1202_model.py --tap-blocked

.PHONY: weights
weights: $(WEIGHTS)

# ── 3. Bitstream (firmware baked in via onChipRamHexFile) ─────────────────────
# Rebuilding Verilog + bitstream bakes the current firmware .hex into BRAM.
$(BITSTREAM): $(FIRMWARE_HEX)
	@echo "[sbt] generating Verilog..."
	cd $(REPO_ROOT) && $(SBT) "runMain vexriscv.demo.MuraxHyperRAM_iCEBreaker"
	@echo "[synth] yosys + nextpnr + icepack..."
	$(MAKE) -C $(BITSTREAM_DIR) compile

.PHONY: bitstream
bitstream: $(BITSTREAM)

# ── 4. Flash image ─────────────────────────────────────────────────────────────
$(FLASH_IMAGE): $(BITSTREAM) $(WEIGHTS)
	@echo "[image] assembling flash image..."
	@$(call size_check,$(BITSTREAM),$(WEIGHTS_OFFSET))
	@WMAX=$$(( $(FLASH_SIZE) - $(WEIGHTS_OFFSET) )); \
	 WSIZ=$$(wc -c < $(WEIGHTS)); \
	 if [ $$WSIZ -gt $$WMAX ]; then \
	     echo "ERROR: weights $$($(call human,$$WSIZ)) exceed available $$($(call human,$$WMAX))"; exit 1; \
	 fi
	@mkdir -p $(dir $(FLASH_IMAGE))
	dd if=/dev/zero  bs=1M count=16 of=$(FLASH_IMAGE) status=none
	dd if=$(BITSTREAM) of=$(FLASH_IMAGE) bs=1 conv=notrunc status=none
	dd if=$(WEIGHTS)   of=$(FLASH_IMAGE) bs=1 seek=$(WEIGHTS_OFFSET) conv=notrunc status=none
	@$(call write_manifest)
	@echo "[image] done: $(FLASH_IMAGE)"
	@echo "[image] manifest: $(MANIFEST)"

# helper: check bitstream fits below weights offset
define size_check
	@BSIZ=$$(wc -c < $(1)); \
	 if [ $$BSIZ -gt $(2) ]; then \
	     echo "ERROR: bitstream $$BSIZ B exceeds max $(2) B"; exit 1; \
	 fi
endef

define write_manifest
	{ \
	  echo "flash_image_manifest"; \
	  echo "generated: $$(date -u '+%Y-%m-%dT%H:%M:%SZ')"; \
	  echo "flash_size_bytes: $(FLASH_SIZE)"; \
	  echo ""; \
	  echo "bitstream_offset: 0"; \
	  echo "bitstream_file:   $(BITSTREAM)"; \
	  echo "bitstream_bytes:  $$(wc -c < $(BITSTREAM))"; \
	  echo "bitstream_sha256: $$(sha256sum $(BITSTREAM) | awk '{print $$1}')"; \
	  echo ""; \
	  echo "weights_offset:   $(WEIGHTS_OFFSET)"; \
	  echo "weights_file:     $(WEIGHTS)"; \
	  echo "weights_bytes:    $$(wc -c < $(WEIGHTS))"; \
	  echo "weights_sha256:   $$(sha256sum $(WEIGHTS) | awk '{print $$1}')"; \
	  echo ""; \
	  echo "flash_image_sha256: $$(sha256sum $(FLASH_IMAGE) | awk '{print $$1}')"; \
	} > $(MANIFEST)
endef

# ── Programming targets ────────────────────────────────────────────────────────
.PHONY: prog
prog: $(FLASH_IMAGE)
	@echo "[prog] programming full flash (~65 min total)..."
	$(ICEPROG) $(FLASH_IMAGE)

.PHONY: prog_bitstream
prog_bitstream: $(BITSTREAM)
	@echo "[prog] flashing bitstream only (~5 s)..."
	$(ICEPROG) $(BITSTREAM)

.PHONY: prog_weights
prog_weights: $(WEIGHTS)
	@echo "[prog] flashing weights only at 1 MiB offset (~60 min)..."
	$(ICEPROG) -o 1M $(WEIGHTS)

# ── Dry run: layout + size check ──────────────────────────────────────────────
.PHONY: dry_run
dry_run:
	@echo "Flash layout check (no output written)"
	@echo "────────────────────────────────────────────────────────────────────"
	@printf "  %-10s  %-12s  %s\n" "Offset" "Size" "Content"
	@echo "────────────────────────────────────────────────────────────────────"
	@BSIZ=$$(wc -c < $(BITSTREAM) 2>/dev/null || echo MISSING); \
	 printf "  0x%06X  %-12s  bitstream\n" 0 "$$BSIZ B"
	@WSIZ=$$(wc -c < $(WEIGHTS) 2>/dev/null || echo MISSING); \
	 printf "  0x%06X  %-12s  weights (VWB2)\n" $(WEIGHTS_OFFSET) "$$WSIZ B"
	@printf "  0x%06X  (end of flash)\n" $(FLASH_SIZE)
	@echo "────────────────────────────────────────────────────────────────────"
	@if [ -f $(BITSTREAM) ] && [ -f $(WEIGHTS) ]; then \
	     BSIZ=$$(wc -c < $(BITSTREAM)); \
	     WSIZ=$$(wc -c < $(WEIGHTS)); \
	     USED=$$(( $(WEIGHTS_OFFSET) + WSIZ )); \
	     printf "  Total used: %d B / %d B (%d%%)\n" $$USED $(FLASH_SIZE) $$(( USED * 100 / $(FLASH_SIZE) )); \
	     if [ $$BSIZ -gt $(WEIGHTS_OFFSET) ]; then echo "ERROR: bitstream overflows into weights region!"; exit 1; fi; \
	     WMAX=$$(( $(FLASH_SIZE) - $(WEIGHTS_OFFSET) )); \
	     if [ $$WSIZ -gt $$WMAX ]; then echo "ERROR: weights overflow flash!"; exit 1; fi; \
	     echo "  Layout OK"; \
	 else \
	     echo "  WARNING: one or more files missing — build first"; \
	 fi

# ── Spot-check script (prints expected weight fingerprint) ────────────────────
.PHONY: weight_fingerprint
weight_fingerprint: $(WEIGHTS)
	@$(PYTHON3) -c "\
import struct; \
d = open('$(WEIGHTS)','rb').read(); \
data_off = struct.unpack_from('<I', d, 20)[0]; \
s = (sum(d[:32]) + sum(d[data_off:data_off+18])) & 0xFFFFFFFF; \
print(f'Expected spot-check u32sum: 0x{s:08X}'); \
print(f'  tensor_count={struct.unpack_from(chr(60)+\"I\", d, 8)[0]}'); \
print(f'  data_bytes  ={struct.unpack_from(chr(60)+\"I\", d, 24)[0]}'); \
"

# ── BD4-faithful quantized reference (Step 7, bd-activations-fix) ─────────────
# Regenerates scripts/resnet1202_artifacts/quantized_ref.h with BD4 raw-byte
# checksums (matching firmware print_bd4_cksum output) rather than the old
# int8 u32sum hashes written by gen_resnet1202_model.py.
# Requires: weights_bd.bin and input_32x32.raw in ARTIFACTS (run `make weights`
# first), and numpy/blockdialect_codec available in PYTHON3 environment.
QUANTIZED_REF := $(ARTIFACTS)/quantized_ref.h

.PHONY: quantized_ref
quantized_ref: $(QUANTIZED_REF)

$(QUANTIZED_REF): $(WEIGHTS) $(ARTIFACTS)/input_32x32.raw
	@echo "[ref] computing BD4-faithful reference checksums (~2-5 min)..."
	cd $(REPO_ROOT) && $(PYTHON3) scripts/quantized_reference.py \
		--model rn1202 \
		--blob $(WEIGHTS) \
		--input $(ARTIFACTS)/input_32x32.raw \
		--output $(QUANTIZED_REF)

# ── Clean ──────────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	$(MAKE) -C $(FIRMWARE_DIR) clean
	$(MAKE) -C $(BITSTREAM_DIR) clean
	rm -f $(FLASH_IMAGE) $(MANIFEST)

.PHONY: clean_image
clean_image:
	rm -f $(FLASH_IMAGE) $(MANIFEST)
