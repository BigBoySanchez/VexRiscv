import sys
import struct

def pack(in_bin, out_bin):
    with open(in_bin, 'rb') as f:
        data = f.read()
    magic = 0xB00710AD
    length = len(data)
    entry = 0x11000000
    with open(out_bin, 'wb') as f:
        f.write(struct.pack('<III', magic, length, entry))
        f.write(data)
    print(f"Packed {in_bin} -> {out_bin} (Payload length: {length} bytes, Entry=0x{entry:08X})")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: pack_app.py <in.bin> <out.bin>")
        sys.exit(1)
    pack(sys.argv[1], sys.argv[2])
