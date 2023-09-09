import sys
import random

elm_type = 16
size = 64
suffix = ""

if len(sys.argv) > 1:
    size = int(sys.argv[1])
    size &= 0xFFFFF0

    if size < 8:
        size = 16

if len(sys.argv) > 2:
    elm_type = int(sys.argv[2])

    if elm_type not in [8, 16, 32, 64]:
        elm_type = 16

if len(sys.argv) > 3:
    suffix = sys.argv[3]

type_str = f"int{elm_type}"

val_x = (1 << elm_type)
val_y = ((1 << elm_type) >> 1) if type_str.startswith("i") else 1

dat = [int((random.random() + random.random()) / 2 * val_x) - val_y for _ in range(size)]
dat_strs = list(map(str, dat))

print(f"""
// min: {min(dat)}
// max: {max(dat)}
#define DATA{elm_type}{suffix.upper()}_SIZE {size}
alignas(32) {type_str}_t data{elm_type}{suffix}[{size}] = {{
{",".join(dat_strs)}
}};
""")
