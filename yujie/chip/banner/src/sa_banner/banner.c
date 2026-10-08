/* Pass-through kernel: copies each input frame unchanged and draws one fixed text on the
 * EVK display with the first frame after boot. Nothing is drawn again, so a window costs
 * one 48-value copy per frame. Pin setup as in the SDK's display kernel. */
#include "kernel_sa_banner.h"
#include "bsp/display/ea1026.h"
#include <string.h>

#define GPIO_DISP_SCLK  GPIO_PIN_0
#define GPIO_DISP_MOSI  GPIO_PIN_1
#define GPIO_DISP_DC    GPIO_PIN_2
#define GPIO_DISP_CS    GPIO_PIN_3

static QUEUE(sa_banner);
static bool drawn;
static void flushed(void) {}

kernel_handle_t sa_banner_init(const sa_banner_kernel_param_t *p) {
    gpio_set_mux_mode(GPIO_DISP_SCLK, GPIO_MODE_2);
    gpio_set_mux_mode(GPIO_DISP_MOSI, GPIO_MODE_2);
    gpio_set_mux_mode(GPIO_DISP_CS, GPIO_MODE_0);
    gpio_set_inversion(GPIO_DISP_CS, GPIO_IO_INVERTED);
    gpio_set_direction(GPIO_DISP_CS, GPIO_DIR_OUT);
    gpio_set_mux_mode(GPIO_DISP_DC, GPIO_MODE_0);
    gpio_set_direction(GPIO_DISP_DC, GPIO_DIR_OUT);
    spim_handle_t spi;
    spim_init(0, &spi);
    spim_set_cs_gpio(spi, GPIO_DISP_CS);
    spim_set_clock(spi, 10000000);
    ea1026_init(spi, GPIO_NB, GPIO_DISP_DC);
    ea1026_register_callback(flushed);
    return (kernel_handle_t)p;
}
bool sa_banner_core_is_free(void) { return true; }
void sa_banner_core_queue_op(core_queue_callback_t cb) { queue_tail_put(sa_banner, cb); }
void sa_banner_do_core_work(kernel_handle_t h, const int32_t *in, int32_t *out, core_done_callback_t cb) {
    const sa_banner_kernel_param_t *p = (const sa_banner_kernel_param_t *)h;
    memcpy(out, in, p->count * sizeof(int32_t));
    if (!drawn) {
        drawn = true;
        ea1026_clear();
        ea1026_draw_text(0, 0, edm_set | edm_auto_wrap, (const char *)p->text);
        ea1026_flush();
    }
    cb(true, true);
    core_queue_callback_t next = queue_head_get(sa_banner);
    if (next) next();
}
