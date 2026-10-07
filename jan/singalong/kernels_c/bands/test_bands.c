/* PC test harness: raw int16 mono 16 kHz from stdin -> one line per frame: voiced level_db b0 b1 ... b47
 *   gcc -O2 -o test_bands test_bands.c bands_core.c -lm
 *   ffmpeg -v error -i song.wav -ac 1 -ar 16000 -f s16le - | ./test_bands > bands.txt
 * tests/test_bands_c.py runs this and compares with singalong.frontend. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bands_core.h"

int main(void)
{
    static int16_t buf[BANDS_WIN];
    static int16_t in[BANDS_HOP];
    uint8_t out[NBANDS];
    bands_state_t st;
    size_t filled = 0;
    bands_init(&st);
    for (;;) {
        size_t n = fread(in, sizeof(int16_t), BANDS_HOP, stdin);
        if (n < BANDS_HOP) break;
        if (filled < BANDS_WIN) {
            memcpy(buf + filled, in, BANDS_HOP * sizeof(int16_t));
            filled += BANDS_HOP;
            if (filled < BANDS_WIN) continue;
        } else {
            memmove(buf, buf + BANDS_HOP, (BANDS_WIN - BANDS_HOP) * sizeof(int16_t));
            memcpy(buf + BANDS_WIN - BANDS_HOP, in, BANDS_HOP * sizeof(int16_t));
        }
        float db;
        int v = bands_frame(&st, buf, out, &db);
        printf("%d %.2f", v, db);
        for (int b = 0; b < NBANDS; b++) printf(" %d", out[b]);
        printf("\n");
    }
    return 0;
}
