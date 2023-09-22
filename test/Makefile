ARMVFP_OPTIONS= -march=armv6z+fp -mtune=arm1176jzf-s -mfpu=vfp -mfloat-abi=hard 
SRC_DIR=../src/
RM=del
# NASM=nasm -felf 
ISPC=ispc --emit-obj --addressing=32 --arch=x86 --target-os=windows -O3
NASM=nasm.cmd -felf --prefix _
CC=gcc


generic_min_max:
	$(CC) -Ofast -ogeneric_min_max.exe test_min_max.c $(SRC_DIR)/generic/search.c
mmx_min_max:
	$(CC) -Os -mmmx -ommx_min_max.exe test_min_max.c $(SRC_DIR)/x86/search_mmx.c
	$(CC) -Os -mmmx -DUSE_ANDNOT -ommx_min_max_2.exe test_min_max.c $(SRC_DIR)/x86/search_mmx.c
sse2_min_max:
	$(CC) -Os -msse2 -osse2_min_max.exe test_min_max.c $(SRC_DIR)/x86/search_sse2.c
	$(CC) -Os -msse2 -DUSE_PIPE -osse2_min_max_2.exe test_min_max.c $(SRC_DIR)/x86/search_sse2.c
avx2_min_max:
	$(CC) -Os -mavx2 -oavx2_min_max.exe test_min_max.c $(SRC_DIR)/x86/search_avx2.c
ispc_avx2_min_max:
	$(ISPC) --target=avx2-i16x16 -o ispc_i16.o $(SRC_DIR)/ispc_i16.ispc
	$(CC) -Os -mavx2 -oispc_avx2_min_max.exe test_min_max.c ispc_i16.o
neon_min_max:
	$(CC) -Os -mfpu=neon -oneon_min_max.exe test_min_max.c $(SRC_DIR)/arm/search_neon.c
autovfp_min_max:
	$(CC) -Ofast -march=armv6z+fp -mtune=arm1176jzf-s -mfpu=vfp -mfloat-abi=hard -oautovfp_min_max.exe test_min_max.c $(SRC_DIR)/arm_autovfp.c
autoneon_min_max:
	$(CC) -Ofast -mfpu=neon -oautoneon_min_max.exe test_min_max.c $(SRC_DIR)/generic/search.c

generic_minmax:
	$(CC) -Ofast -ogeneric_minmax.exe test_minmax.c $(SRC_DIR)/generic/search.c
mmx_minmax:
	$(CC) -Os -mmmx -ommx_minmax.exe test_minmax.c $(SRC_DIR)/x86/search_mmx.c
	$(CC) -Os -mmmx -DUSE_ANDNOT -ommx_minmax_2.exe test_minmax.c $(SRC_DIR)/x86/search_mmx.c
sse2_minmax:
	$(CC) -Os -msse2 -osse2_minmax.exe test_minmax.c $(SRC_DIR)/x86/search_sse2.c
avx2_minmax:
	$(CC) -Os -mavx -mavx2 -oavx2_minmax.exe test_minmax.c $(SRC_DIR)/x86/search_avx2.c
ispc_avx2_minmax:
	$(ISPC) --target=avx2-i16x16 -o ispc_i16.o $(SRC_DIR)/ispc_i16.ispc
	$(CC) -Os -mavx2 -oispc_avx2_minmax.exe test_minmax.c ispc_i16.o
neon_minmax:
	$(CC) -Os -mfpu=neon -oneon_minmax.exe test_minmax.c $(SRC_DIR)/arm/search_neon.c
autovfp_minmax:
	$(CC) -Ofast -march=armv6z+fp -mtune=arm1176jzf-s -mfpu=vfp -mfloat-abi=hard -oautovfp_minmax.exe test_minmax.c $(SRC_DIR)/arm_autovfp.c
autoneon_minmax:
	$(CC) -Ofast -mfpu=neon -oautoneon_minmax.exe test_minmax.c $(SRC_DIR)/generic/search.c

generic_min_max8:
	$(CC) -Ofast -ogeneric_min_max8.exe test_min_max8.c $(SRC_DIR)/generic/search.c
mmx_min_max8:
	$(CC) -Os -mmmx -ommx_min_max8.exe test_min_max8.c $(SRC_DIR)/x86/search_mmx.c
sse2_min_max8:
	$(CC) -Os -msse2 -osse2_min_max8.exe test_min_max8.c $(SRC_DIR)/x86/search_sse2.c
avx2_min_max8:
	$(CC) -Os -mavx -mavx2 -oavx2_min_max8.exe test_min_max8.c $(SRC_DIR)/x86/search_avx2.c
sim_v6simd_min_max8:
	$(CC) -O2 -osim_v6simd_min_max8.exe test_min_max8.c $(SRC_DIR)/arm/search_v6simd.c $(SRC_DIR)/generic/sim_arm_v6simd_u32.c

generic_minmax8:
	$(CC) -Ofast -ogeneric_minmax8.exe test_minmax8.c $(SRC_DIR)/generic/search_
mmx_minmax8:
	$(CC) -Os -mmmx -ommx_minmax8.exe test_minmax8.c $(SRC_DIR)/x86/search_mmx.c
sse2_minmax8:
	$(CC) -Os -msse2 -osse2_minmax8.exe test_minmax8.c $(SRC_DIR)/x86/search_sse2.c
avx2_minmax8:
	$(CC) -Os -mavx -mavx2 -oavx2_minmax8.exe test_minmax8.c $(SRC_DIR)/x86/search_avx2.c

generic_count:
	$(CC) -Ofast -flto -ogeneric_count.exe test_count.c $(SRC_DIR)/generic/search.c
v6simd_count:
	$(CC) -Ofast -march=armv6z+fp -flto -ov6simd_count.exe test_count.c $(SRC_DIR)/arm/search_v6simd.c
sim_v6simd_count:
	$(CC) -O2 -osim_v6simd_count.exe test_count.c $(SRC_DIR)/arm/search_v6simd.c $(SRC_DIR)/generic/sim_arm_v6simd_u32.c
generic_count8:
	$(CC) -Ofast -flto -ogeneric_count8.exe test_count8.c $(SRC_DIR)/generic/search.c
v6simd_count8:
	$(CC) -Ofast -march=armv6z+fp -flto -ov6simd_count8.exe test_count8.c $(SRC_DIR)/arm/search_v6simd.c
sim_v6simd_count8:
	$(CC) -O2 -osim_v6simd_count8.exe test_count8.c $(SRC_DIR)/arm/search_v6simd.c $(SRC_DIR)/generic/sim_arm_v6simd_u32.c


generic_abs:
	$(CC) -Ofast -flto -ogeneric_abs.exe test_abs.c $(SRC_DIR)/generic/misc.c
mmx_abs:
	$(CC) -Ofast -flto -mmmx -ommx_abs.exe test_abs.c $(SRC_DIR)/x86/misc_mmx.c
sse2_abs:
	$(CC) -Ofast -flto -msse2 -osse2_abs.exe test_abs.c $(SRC_DIR)/x86/misc_sse2.c

generic_diff:
	$(CC) -Ofast -ogeneric_diff.exe test_diff.c $(SRC_DIR)/generic/misc.c
mmx_diff:
	$(CC) -Os -mmmx -ommx_diff.exe test_diff.c $(SRC_DIR)/x86/misc_mmx.c
sse2_diff:
	$(CC) -Os -msse2 -osse2_diff.exe test_diff.c $(SRC_DIR)/x86/misc_sse2.c
avx2_diff:
	$(CC) -Os -mavx2 -oavx2_diff.exe test_diff.c $(SRC_DIR)/x86/misc_avx2.c

generic_hw:
	$(CC) -Ofast -ogeneric_hw.exe test_hw.c $(SRC_DIR)/generic/misc.c
mmx_hw:
	$(CC) -Os -mmmx -ommx_hw.exe test_hw.c $(SRC_DIR)/x86/misc_mmx.c
sse2_hw:
	$(CC) -Os -msse2 -osse2_hw.exe test_hw.c $(SRC_DIR)/x86/misc_sse2.c
avx2_hw:
	$(CC) -Os -mavx2 -oavx2_hw.exe test_hw.c $(SRC_DIR)/x86/misc_avx2.c
	$(CC) -Os -mavx2 -DUSE_POPCNT -oavx2_hw_3.exe test_hw.c $(SRC_DIR)/x86/misc_avx2.c

generic_rev:
	$(CC) -Ofast -ogeneric_rev.exe test_rev.c $(SRC_DIR)/generic/sort.c
x86_rev:
	$(NASM) $(SRC_DIR)/x86/sort_generic.asm
	$(CC) -Ofast -ox86_rev.exe test_rev.c $(SRC_DIR)/x86_generic.o
mmx_rev:
	$(CC) -Os -mmmx -ommx_rev.exe test_rev.c $(SRC_DIR)/x86/sort_mmx.c
x86_3dnow_rev:
	$(CC) -Os -m3dnow -march=athlon-xp -mtune=athlon-xp -ox86_3dnow_rev.exe test_rev.c $(SRC_DIR)/x86/sort_mmx.c
sse2_rev:
	$(CC) -Os -msse2 -osse2_rev.exe test_rev.c $(SRC_DIR)/x86/sort_sse2.c

generic_rev8:
	$(CC) -Ofast -ogeneric_rev8.exe test_rev8.c $(SRC_DIR)/generic/sort.c
x86_rev8:
	$(NASM) $(SRC_DIR)/x86_generic.asm
	$(CC) -Ofast -ox86_rev8.exe test_rev8.c $(SRC_DIR)/x86_generic.o
mmx_rev8:
	$(CC) -Os -mmmx -ommx_rev8.exe test_rev8.c $(SRC_DIR)/x86/sort_mmx.c
x86_3dnow_rev8:
	$(CC) -Os -m3dnow -march=athlon-xp -mtune=athlon-xp -ox86_3dnow_rev8.exe test_rev8.c $(SRC_DIR)/x86/sort_mmx.c
sse2_rev8:
	$(CC) -Os -msse2 -osse2_rev8.exe test_rev8.c $(SRC_DIR)/x86/sort_sse2.c

generic_rev32:
	$(CC) -Ofast -ogeneric_rev32.exe test_rev32.c $(SRC_DIR)/generic/sort.c
x86_rev32:
	$(NASM) $(SRC_DIR)/x86_generic.asm
	$(CC) -Ofast -ox86_rev32.exe test_rev32.c $(SRC_DIR)/x86_generic.o
mmx_rev32:
	$(CC) -Os -mmmx -ommx_rev32.exe test_rev32.c $(SRC_DIR)/x86/sort_mmx.c
x86_3dnow_rev32:
	$(CC) -Os -m3dnow -march=athlon-xp -mtune=athlon-xp -ox86_3dnow_rev32.exe test_rev32.c $(SRC_DIR)/x86/sort_mmx.c
sse2_rev32:
	$(CC) -Os -msse2 -osse2_rev32.exe test_rev32.c $(SRC_DIR)/x86/sort_sse2.c
avx2_rev32:
	$(CC) -Os -mavx2 -oavx2_rev32.exe test_rev32.c $(SRC_DIR)/x86/sort_avx2.c
avx2_nocross_rev32:
	$(CC) -Os -mavx2 -DNO_CROSS_FETCH -oavx2_nocross_rev32.exe test_rev32.c $(SRC_DIR)/x86/sort_avx2.c

generic_shift:
	$(CC) -Ofast -ogeneric_shift.exe test_shift.c $(SRC_DIR)/generic/sort.c
mmx_shift:
	$(CC) -Os -mmmx -ommx_shift.exe test_shift.c $(SRC_DIR)/x86/sort_mmx.c
sse2_shift:
	$(CC) -Os -msse2 -osse2_shift.exe test_shift.c $(SRC_DIR)/x86/sort_sse2.c
avx2_shift:
	$(CC) -Os -mavx2 -oavx2_shift.exe test_shift.c $(SRC_DIR)/x86/sort_avx2.c

generic_sort:
	$(CC) -Ofast -ogeneric_sort.exe test_sort.c $(SRC_DIR)/generic/sort.c
mmx_sort:
	$(CC) -Os -mmmx -ommx_sort.exe test_sort.c $(SRC_DIR)/x86/sort_mmx.c
sse2_sort:
	$(CC) -Os -msse2 -osse2_sort.exe test_sort.c $(SRC_DIR)/x86/sort_sse2.c
avx2_sort:
	$(CC) -Os -mavx2 -oavx2_sort.exe test_sort.c $(SRC_DIR)/x86/sort_avx2.c


generic_hex:
	$(CC) -Ofast -ogeneric_hex.exe test_hex.c $(SRC_DIR)/generic/hex_u32.c
generic_hex_omp_x4:
	$(CC) -Ofast -fopenmp -ogeneric_hex_omp_x4.exe test_hex.c $(SRC_DIR)/generic/hex_u32.c
mmx_hex:
	$(CC) -Ofast -mmmx -ommx_hex.exe test_hex.c $(SRC_DIR)/x86/hex_mmx.c
neon_hex:
	$(CC) -Ofast -mfpu=neon -oneon_hex.exe test_hex.c $(SRC_DIR)/arm/hex_neon.c
avx2_hex:
	$(CC) -Ofast -mavx2 -oavx2_hex.exe test_hex.c $(SRC_DIR)/x86/hex_avx2.c
avx2_hex_omp_x4:
	$(CC) -Ofast -mavx2 -fopenmp -oavx2_hex_omp_x4.exe test_hex.c $(SRC_DIR)/x86/hex_avx2.c
dpcpp_hex_omp2_x8:
	dpcpp -Ofast -fopenmp -fvectorize -vec -odpcpp_hex_omp2x8.exe test_hex.c $(SRC_DIR)/generic/hex.c
dpcpp_xe_hex:
	dpcpp -Ofast -fopenmp -fiopenmp -fopenmp-targets=spir64_x86_64 -march=alderlake -mtune=alderlake -odpcpp_xe_hex.exe test_hex.c $(SRC_DIR)/generic/hex.c

generic_hex_enc:
	$(CC) -Ofast -ogeneric_hex_enc.exe test_hex_enc.c $(SRC_DIR)/generic/hex_u32.c
v6simd_hex_enc:
	$(CC) -Ofast -march=armv6z+fp -ov6simd_hex_enc.exe test_hex_enc.c $(SRC_DIR)/arm/hex_v6simd.c
sim_v6simd_hex_enc:
	$(CC) -O2 -osim_v6simd_hex_enc.exe test_hex_enc.c $(SRC_DIR)/arm/hex_v6simd.c $(SRC_DIR)/generic/sim_arm_v6simd_u32.c
avx2_hex_enc:
	$(CC) -Ofast -mavx2 -oavx2_hex_enc.exe test_hex_enc.c $(SRC_DIR)/x86/hex_avx2.c

generic_base64e:
	$(CC) -Ofast  -ogeneric_base64e.exe base64e.c $(SRC_DIR)/generic/base64_u32.c
generic_base64e_omp_x4:
	$(CC) -Ofast  -fopenmp  -ogeneric_base64e_omp_x4.exe base64e.c $(SRC_DIR)/generic/base64_u32.c
movbe_base64e:
	$(CC) -Ofast  -mavx2 -mmovbe -omovbe_base64e.exe base64e.c $(SRC_DIR)/generic/base64_u32.c
movbe_base64e_omp_x4:
	$(CC) -Ofast  -mavx2 -mmovbe -fopenmp  -omovbe_base64e_omp_x4.exe base64e.c $(SRC_DIR)/generic/base64_u32.c
mmx_base64e:
	$(CC) -Wall -Ofast  -mmmx  -ommx_base64e.exe base64e.c $(SRC_DIR)/x86/base64_mmx.c
avx2_base64e:
	$(CC) -Ofast  -mavx2  -oavx2_base64e.exe base64e.c $(SRC_DIR)/x86/base64_avx_2.c


clean:
	$(RM) *.exe
