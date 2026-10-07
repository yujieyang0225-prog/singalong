#ifndef BANDS_CORE_H
#define BANDS_CORE_H
#include <stdint.h>

#define NBANDS 48
#define BANDS_WIN 1024
#define BANDS_HOP 256

typedef struct {
    float floor_db;
    int have_floor;
} bands_state_t;

void bands_init(bands_state_t *st);
/* one 1024-sample frame (int16 @ 16 kHz) -> 48 bands 0..100; returns 1 if voiced. level_db may be NULL */
int bands_frame(bands_state_t *st, const int16_t *pcm, uint8_t out[NBANDS], float *level_db);
int32_t bands_log2_q4(uint32_t x);

#endif
