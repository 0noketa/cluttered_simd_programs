#include <stdio.h>
#include <stdint.h>
#include <stdalign.h>

#include "../include/base64.h"


#define BLOCK_SIZE 0xC0000

alignas(32) uint8_t encoded[BLOCK_SIZE * 4 / 3] = {0,};
alignas(32) uint8_t decoded[BLOCK_SIZE] = {0,};


int main(int argc, char *argv[])
{
    FILE *fIn = (argc > 1) ? fopen(argv[1], "rb") : stdin;

    if (fIn == NULL) return 1;

    while (!feof(fIn))
    {
        size_t input_size = fread(decoded, 1, BLOCK_SIZE, fIn);
        if (input_size == 0) continue;

        size_t rem;
        base64_48n_encode(input_size, decoded, encoded);

        size_t output_size = input_size * 4 / 3;

        fwrite(encoded, 1, output_size, stdout);
    }

    if (fIn != stdin) fclose(fIn);

    return 0;
}
