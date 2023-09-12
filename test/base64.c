#include <stdio.h>
#include <stdint.h>
#include <stdalign.h>

#include "../include/base64.h"


#define BLOCK_SIZE 0x800000

alignas(32) uint8_t encoded[BLOCK_SIZE];
alignas(32) uint8_t decoded[BLOCK_SIZE / 4 * 3];


int main(int argc, char *argv[])
{
    if (argc < 2) return 0;

    FILE *fIn = fopen(argv[1], "rb");

    if (fIn == NULL) return 1;

    while (!feof(fIn))
    {
        size_t input_size = fread(encoded, 1, BLOCK_SIZE, fIn);
        if (input_size == 0) break;

        size_t rem;
        base64_32n_decode(input_size, encoded, decoded, &rem);

        size_t output_size = (input_size / 4 - 1) * 3 + rem;

        fwrite(decoded, 1, output_size, stdout);

        if (output_size < BLOCK_SIZE / 4 * 3)
        {
            for (size_t i = 0; i < output_size; ++i)
            {
                fprintf(stderr, "%s%02x", (i % 16 ? " " : "\n"), (int)decoded[i]);
            }

            fprintf(stderr, "\n%d -> %d  %d\n", (int)input_size, (int)output_size, (int)rem);
        }
    }

    fclose(fIn);

    return 0;
}
