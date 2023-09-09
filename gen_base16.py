import sys
import random

size = 64
suffix = ""

if len(sys.argv) > 1:
    size = int(sys.argv[1])
    size &= 0xFFFF80

    if size < 8:
        size = 64

if len(sys.argv) > 2:
    suffix = sys.argv[2]

def gen_hex(size):
    r = []
    for _ in range(size):
        s = random.choice("0123456789ABCDEFabcdef")
        s += random.choice("0123456789ABCDEFabcdef")

        r.append(s)

    return r


dat = gen_hex(size // 2)
dat_str = ""

for i in range(size // 64):
    dat_str += '"' + "".join(dat[i * 32:i * 32 + 32]) + '"\n'


print(f"""
// min: 64
// size: 64n
#define DATA_HEX{suffix.upper()}_SIZE {size}
#define DATA_HEX{suffix.upper()}_DECODED_SIZE {size // 2}
alignas(32) uint8_t data_hex{suffix}[DATA_HEX{suffix.upper()}_SIZE + 1] =
{dat_str}
;
alignas(32) uint8_t data_hex{suffix}_decoded[DATA_HEX{suffix.upper()}_DECODED_SIZE] = {{
{ ",".join(["0x" + i for i in dat]) }
}};
""")
