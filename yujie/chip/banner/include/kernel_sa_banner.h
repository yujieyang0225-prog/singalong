#ifndef KERNEL_SA_BANNER_H
#define KERNEL_SA_BANNER_H
#include "kernel_common.h"
typedef struct { uint32_t count; uint8_t *text; int32_t num_text; } sa_banner_kernel_param_t;
kernel_handle_t sa_banner_init(const sa_banner_kernel_param_t *p);
void sa_banner_do_core_work(kernel_handle_t h, const int32_t *in, int32_t *out, core_done_callback_t cb);
bool sa_banner_core_is_free(void);
void sa_banner_core_queue_op(core_queue_callback_t cb);
#endif
