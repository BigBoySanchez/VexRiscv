VERILOG = ../../../MuraxHyperRAM.v toplevel.v

WEIGHT_OFFSET = 1048576
WEIGHTS_BIN ?= weights.bin
APP_BIN ?= app.bin

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

prog_bitstream : bin/toplevel.bin
	iceprog bin/toplevel.bin

prog_weights : $(WEIGHTS_BIN)
	iceprog -o 1M $(WEIGHTS_BIN)

prog_app : $(APP_BIN)
	iceprog -o 5M $(APP_BIN)

clean :
	rm -f ../../../MuraxHyperRAM.v
	rm -rf bin
	rm -f MuraxHyperRAM.v*.bin
