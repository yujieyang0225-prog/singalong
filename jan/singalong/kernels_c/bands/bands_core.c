/*
 * Semitone band front-end for the Pulsar RISC-V (D02), portable C, no malloc, no libm in the hot path.
 *
 *   int16 PCM @ 16 kHz, 1024-sample Hann window, hop 256  ->  48 bands (MIDI 40..87), 0..100 integers
 *
 * Same maths as singalong/frontend.py (log2 band energy, per-frame max-normalisation over 40 dB, voicing gate,
 * x100). The FFT here is a plain radix-2 float FFT so the kernel can be tested on a PC
 * (tests/test_bands_c.py compares it with the Python front-end on real audio). On the chip, replace bands_fft()
 * with the HW FFT (Talamo FFT step, int16, no window -> apply the window here first) and keep the rest.
 *
 * Fixed-point notes for the port: log2 of the band energy is needed at ~0.1 octave resolution only
 * (100 steps over 13.3 octaves), so a 5-bit integer log2 + 4-bit fractional table is enough; see bands_log2_q4().
 *
 * API (one frame):
 *   void bands_init(bands_state_t *st);
 *   int  bands_frame(bands_state_t *st, const int16_t *pcm1024, uint8_t out[48], float *level_db);
 *        returns 1 if the frame is voiced (bands written), 0 if silent (bands all 0)
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "bands_core.h"

#define SR 16000
#define NFFT 1024
#define NBINS (NFFT / 2 + 1)
#define LO_MIDI 40
#define RANGE_OCT 13.3f      /* 40 dB */
#define GATE_DB 15.0f
#define FLOOR_UP_DB_PER_FRAME (1.0f / 62.5f)

static float g_hann[NFFT];
#define MAXBINS 12
static float g_fb_w[NBANDS][MAXBINS]; /* triangular weights; the top band (1.25 kHz) spans ~10 bins */
static int16_t g_fb_k0[NBANDS];      /* first bin of each band */
static uint8_t g_fb_n[NBANDS];       /* number of bins */
static int g_tables_ready = 0;

static float midi_hz(float m) { return 440.0f * powf(2.0f, (m - 69.0f) / 12.0f); }

static void build_tables(void)
{
    int n, b, k;
    for (n = 0; n < NFFT; n++)                 /* numpy.hanning: symmetric, N-1 in the denominator */
        g_hann[n] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * n / (NFFT - 1));
    for (b = 0; b < NBANDS; b++) {
        float c = midi_hz(LO_MIDI + b), lo = midi_hz(LO_MIDI + b - 1), hi = midi_hz(LO_MIDI + b + 1);
        float w[NBINS], sum = 0.0f;
        int first = -1, last = -1;
        for (k = 0; k < NBINS; k++) {
            float f = (float)k * SR / NFFT;
            float a = (f - lo) / (c - lo), d = (hi - f) / (hi - c);
            float v = a < d ? a : d;
            w[k] = v > 0 ? v : 0;
            sum += w[k];
        }
        if (sum == 0.0f) {                     /* band narrower than a bin: interpolate the two nearest bins */
            float kk = c * NFFT / SR;
            int ki = (int)kk;
            w[ki] += 1.0f - (kk - ki);
            w[ki + 1] += kk - ki;
            sum = 1.0f;
        }
        for (k = 0; k < NBINS; k++) if (w[k] > 0) { if (first < 0) first = k; last = k; }
        g_fb_k0[b] = (int16_t)first;
        g_fb_n[b] = (uint8_t)(last - first + 1);
        for (k = 0; k < g_fb_n[b] && k < MAXBINS; k++) g_fb_w[b][k] = w[first + k] / sum;
    }
    g_tables_ready = 1;
}

/* in-place iterative radix-2 complex FFT, re/im arrays of length NFFT */
static void bands_fft(float *re, float *im)
{
    int i, j, k, m, len;
    for (i = 1, j = 0; i < NFFT; i++) {       /* bit reversal */
        int bit = NFFT >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { float t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (len = 2; len <= NFFT; len <<= 1) {
        float ang = -2.0f * (float)M_PI / len;
        float wr = cosf(ang), wi = sinf(ang);
        for (i = 0; i < NFFT; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (k = 0; k < len / 2; k++) {
                m = i + k;
                float ur = re[m], ui = im[m];
                float vr = re[m + len / 2] * cr - im[m + len / 2] * ci;
                float vi = re[m + len / 2] * ci + im[m + len / 2] * cr;
                re[m] = ur + vr; im[m] = ui + vi;
                re[m + len / 2] = ur - vr; im[m + len / 2] = ui - vi;
                float t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
            }
        }
    }
}

void bands_init(bands_state_t *st)
{
    if (!g_tables_ready) build_tables();
    st->floor_db = 1e9f;
    st->have_floor = 0;
}

int bands_frame(bands_state_t *st, const int16_t *pcm, uint8_t out[NBANDS], float *level_db)
{
    static float re[NFFT], im[NFFT];
    float e[NBANDS], emax = -1e30f, acc = 0.0f, db;
    int n, b, k, voiced;
    if (!g_tables_ready) build_tables();
    for (n = 0; n < NFFT; n++) {
        float x = pcm[n] / 32768.0f;
        acc += x * x;
        re[n] = x * g_hann[n];
        im[n] = 0.0f;
    }
    db = 10.0f * log10f(acc / NFFT + 1e-12f);
    if (level_db) *level_db = db;
    /* voicing gate: noise floor follows the level down at once, up slowly (frontend.OnlineGate) */
    if (!st->have_floor || db < st->floor_db) { st->floor_db = db; st->have_floor = 1; }
    else st->floor_db = fminf(st->floor_db + FLOOR_UP_DB_PER_FRAME, db);
    voiced = db >= st->floor_db + GATE_DB;

    bands_fft(re, im);
    for (b = 0; b < NBANDS; b++) {
        float s = 0.0f;
        for (k = 0; k < g_fb_n[b] && k < MAXBINS; k++) {
            int bin = g_fb_k0[b] + k;
            float p = re[bin] * re[bin] + im[bin] * im[bin];
            s += g_fb_w[b][k] * p;
        }
        e[b] = log2f(s + 1e-10f);
        if (e[b] > emax) emax = e[b];
    }
    for (b = 0; b < NBANDS; b++) {
        float v = (e[b] - emax);
        if (v < -RANGE_OCT) v = -RANGE_OCT;
        v = v / RANGE_OCT + 1.0f;                 /* 0..1 */
        out[b] = voiced ? (uint8_t)(v * 100.0f + 0.5f) : 0;
    }
    return voiced;
}

/* Integer log2 in Q4 (16 steps per octave) for the fixed-point port: log2(x) ≈ msb + table[next 4 bits]. */
int32_t bands_log2_q4(uint32_t x)
{
    static const uint8_t frac[16] = {0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15};
    int msb;
    if (x == 0) return -16 * 32;
    msb = 31 - __builtin_clz(x);
    return msb * 16 + frac[(msb >= 4 ? (x >> (msb - 4)) : (x << (4 - msb))) & 15];
}
