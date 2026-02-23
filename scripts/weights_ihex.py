#!/usr/bin/env python3
"""
Convert raw hex words (one 32-bit word per line) to Intel HEX format.
The output is compatible with SpinalHDL's HexTools.initRam().

Usage: python3 weights_ihex.py [input_raw_hex] [output_ihex] [base_address_hex]
Defaults: weights.hex -> weights_ihex.hex @ 0x20000000
"""

import sys
import struct
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

def raw_hex_to_intel_hex(input_file, output_file, base_address):
    """Convert raw hex word file to Intel HEX format."""
    
    # Read raw hex words
    data = bytearray()
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Each line is a 32-bit hex word (little-endian storage)
            word = int(line, 16)
            data.extend(struct.pack('<I', word))
    
    print(f"Read {len(data)} bytes from {input_file}")
    
    with open(output_file, 'w') as f:
        offset = 0
        while offset < len(data):
            # Emit Extended Linear Address record every 64KB boundary
            segment_addr = (base_address + offset) >> 16
            if offset == 0 or ((base_address + offset) & 0xFFFF) == 0:
                # Type 04: Extended Linear Address
                record_data = struct.pack('>H', segment_addr)
                checksum = (2 + 0 + 0 + 4 + (segment_addr >> 8) + (segment_addr & 0xFF)) & 0xFF
                checksum = (~checksum + 1) & 0xFF
                f.write(f":02000004{segment_addr:04X}{checksum:02X}\n")
            
            # Data records (type 00), max 16 bytes per record
            local_addr = (base_address + offset) & 0xFFFF
            chunk_size = min(16, len(data) - offset)
            
            # Don't cross a 64KB boundary within a record
            remaining_in_segment = 0x10000 - local_addr
            chunk_size = min(chunk_size, remaining_in_segment)
            
            chunk = data[offset:offset + chunk_size]
            
            # Build record: :LLAAAA00DD...CC
            record_bytes = [chunk_size, (local_addr >> 8) & 0xFF, local_addr & 0xFF, 0x00]
            record_bytes.extend(chunk)
            checksum = sum(record_bytes) & 0xFF
            checksum = (~checksum + 1) & 0xFF
            
            hex_data = ''.join(f'{b:02X}' for b in chunk)
            f.write(f":{chunk_size:02X}{local_addr:04X}00{hex_data}{checksum:02X}\n")
            
            offset += chunk_size
        
        # End-of-file record
        f.write(":00000001FF\n")
    
    print(f"Wrote Intel HEX to {output_file} (base address 0x{base_address:08X}, {len(data)} bytes)")

if __name__ == '__main__':
    input_file  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, 'weights.hex')
    output_file = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_HERE, 'weights_ihex.hex')
    base_addr   = int(sys.argv[3], 16) if len(sys.argv) > 3 else 0x20000000

    raw_hex_to_intel_hex(input_file, output_file, base_addr)
