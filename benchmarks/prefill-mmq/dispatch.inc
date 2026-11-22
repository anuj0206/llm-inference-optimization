#include "shape.h"

static int prefill_q4_tile_override() {
    static const int width = [] {
        const char * value = std::getenv("LLM_OPT_Q4_PREFILL_TILE");
        if (!value) {
            return 0;
        }
        if (std::strcmp(value, "32") == 0) {
            return 32;
        }
        if (std::strcmp(value, "64") == 0) {
            return 64;
        }
        if (std::strcmp(value, "128") == 0) {
            return 128;
        }
        GGML_ABORT("LLM_OPT_Q4_PREFILL_TILE must be 32, 64, or 128");
    }();
    return width;
}
