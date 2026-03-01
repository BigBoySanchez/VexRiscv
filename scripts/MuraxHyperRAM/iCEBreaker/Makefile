VERILOG = ../../../MuraxHyperRAM.v toplevel.v

WEIGHT_OFFSET = 1048576
WEIGHTS_BIN ?= weights.bin

generate :
	(cd ../../..; sbt "runMain vexriscv.demo.MuraxHyperRAM_iCEBreaker")

../../../MuraxHyperRAM.v :
	(cd ../../..; sbt "runMain vexriscv.demo.MuraxHyperRAM_iCEBreaker")

../../../MuraxHyperRAM.v*.bin:

bin/toplevel.json : ${VERILOG} ../../../MuraxHyperRAM.v*.bin
	mkdir -p bin
	rm -f MuraxHyperRAM.v*.bin
	cp ../../../MuraxHyperRAM.v*.bin . | true
	yosys -v3 -p "synth_ice40 -dsp -device u -top toplevel -json bin/toplevel.json" ${VERILOG}

bin/toplevel.asc : io.pcf bin/toplevel.json
	nextpnr-ice40 --up5k --pcf io.pcf --package sg48 --json bin/toplevel.json --asc bin/toplevel.asc

bin/toplevel.bin : bin/toplevel.asc
	icepack bin/toplevel.asc bin/toplevel.bin

compile : bin/toplevel.bin

time: bin/toplevel.bin
	icetime -tmd hx8k bin/toplevel.asc

prog : bin/toplevel.bin
	iceprog bin/toplevel.bin

# Combined flash image: bitstream at offset 0, weights at WEIGHT_OFFSET
bin/flash.img : bin/toplevel.bin $(WEIGHTS_BIN)
	@echo "Creating combined flash image..."
	@echo "  Bitstream: bin/toplevel.bin"
	@echo "  Weights:   $(WEIGHTS_BIN) at offset $(WEIGHT_OFFSET)"
	@BITSZ=$$(wc -c < bin/toplevel.bin); \
	if [ $$BITSZ -ge $(WEIGHT_OFFSET) ]; then \
		echo "ERROR: bitstream ($$BITSZ bytes) >= WEIGHT_OFFSET ($(WEIGHT_OFFSET))"; \
		exit 1; \
	fi
	@if [ -f "$(WEIGHTS_BIN)" ]; then \
		WTSZ=$$(stat -c%s "$(WEIGHTS_BIN)"); \
		TOTAL=$$(($(WEIGHT_OFFSET) + $$WTSZ)); \
		dd if=/dev/zero bs=1 count=$$TOTAL of=$@ 2>/dev/null; \
		dd if=bin/toplevel.bin of=$@ conv=notrunc 2>/dev/null; \
		dd if="$(WEIGHTS_BIN)" of=$@ conv=notrunc bs=1 seek=$(WEIGHT_OFFSET) 2>/dev/null; \
		echo "Flash image: $$TOTAL bytes (bitstream + $$WTSZ bytes weights)"; \
	else \
		echo "WARNING: $(WEIGHTS_BIN) not found, writing bitstream-only image"; \
		cp bin/toplevel.bin $@; \
	fi

prog_flash : bin/flash.img
	iceprog bin/flash.img

clean :
	rm -f ../../../MuraxHyperRAM.v
	rm -rf bin
	rm -f MuraxHyperRAM.v*.bin
