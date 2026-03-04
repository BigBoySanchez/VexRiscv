PROJ_NAME=resnet1202_phase1_int8
DEBUG=no
BENCH=no
MULDIV=yes

SRCS = 	$(wildcard src/*.c) \
		$(wildcard src/*.cpp)  \
		$(wildcard src/*.S)

OBJDIR = build

INC  = -I../resnet1202_common \
       -I../../../../../scripts/resnet1202_artifacts
LIBS =
LIBSINC = -L$(OBJDIR)
LDSCRIPT ?= ./src/linker_icebreaker.ld

SIFIVE_GCC_PACK ?= yes

ifeq ($(SIFIVE_GCC_PACK),yes)
	RISCV_NAME ?= riscv-none-elf
	RISCV_PATH ?= ~/tools/xpack-riscv-none-elf-gcc-13.2.0-2/
else
	RISCV_NAME ?= riscv32-unknown-elf
	ifeq ($(MULDIV),yes)
		RISCV_PATH ?= /opt/riscv32im/
	else
		RISCV_PATH ?= /opt/riscv32i/
	endif
endif

MABI=ilp32
ifeq ($(MULDIV),yes)
	MARCH := rv32im_zicsr
else
	MARCH := rv32i_zicsr
endif
ifeq ($(COMPRESSED),yes)
	MARCH := $(MARCH)c
endif

CFLAGS += -march=$(MARCH)  -mabi=$(MABI) -DNDEBUG
LDFLAGS += -march=$(MARCH)  -mabi=$(MABI)

ifeq ($(DEBUG),yes)
	CFLAGS += -g3 -O0
endif

ifeq ($(DEBUG),no)
	CFLAGS += -g -Os
endif

ifeq ($(BENCH),yes)
	CFLAGS += -fno-inline
endif

ifeq ($(SIFIVE_GCC_PACK),yes)
	RISCV_CLIB=$(RISCV_PATH)/$(RISCV_NAME)/lib/$(MARCH)/$(MABI)/
else
	RISCV_CLIB=$(RISCV_PATH)/$(RISCV_NAME)/lib/
endif

RISCV_OBJCOPY = $(RISCV_PATH)/bin/$(RISCV_NAME)-objcopy
RISCV_OBJDUMP = $(RISCV_PATH)/bin/$(RISCV_NAME)-objdump
RISCV_CC=$(RISCV_PATH)/bin/$(RISCV_NAME)-gcc

CFLAGS +=  -MD -fstrict-volatile-bitfields -fno-strict-aliasing
LDFLAGS +=  -nostdlib -lgcc -mcmodel=medany -nostartfiles -ffreestanding -Wl,-Bstatic,-T,$(LDSCRIPT),-Map,$(OBJDIR)/$(PROJ_NAME).map,--print-memory-usage

OBJS := $(SRCS)
OBJS := $(OBJS:.c=.o)
OBJS := $(OBJS:.cpp=.o)
OBJS := $(OBJS:.S=.o)
OBJS := $(OBJS:..=miaou)
OBJS := $(addprefix $(OBJDIR)/,$(OBJS))


all: $(OBJDIR)/$(PROJ_NAME).elf $(OBJDIR)/$(PROJ_NAME).hex $(OBJDIR)/$(PROJ_NAME).bin $(OBJDIR)/$(PROJ_NAME).asm $(OBJDIR)/$(PROJ_NAME).v

$(OBJDIR)/%.elf: $(OBJS) | $(OBJDIR)
	$(RISCV_CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBSINC) $(LIBS)

%.hex: %.elf
	$(RISCV_OBJCOPY) -O ihex $^ $@

%.bin: %.elf
	$(RISCV_OBJCOPY) -O binary $^ $@

%.v: %.elf
	$(RISCV_OBJCOPY) -O verilog $^ $@

%.asm: %.elf
	$(RISCV_OBJDUMP) -S -d $^ > $@

$(OBJDIR)/src/%.o: src/%.c | $(OBJDIR)
	$(RISCV_CC) -c $(CFLAGS) $(INC) -o $@ $^

$(OBJDIR)/src/%.o: src/%.S | $(OBJDIR)
	$(RISCV_CC) -c $(CFLAGS) $(INC) -o $@ $^

$(OBJDIR):
	mkdir -p $(OBJDIR)/src

clean:
	rm -rf $(OBJDIR)

.PHONY: all clean
