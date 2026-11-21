// Included in speculative.cpp by build-adaptive-ngram.py.
#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

const std::array<double, 9> * common_ngram_budget_costs();
llama_tokens common_ngram_budget_draft(const llama_tokens & history, llama_token sampled, std::size_t cap);

namespace {

struct ngram_acceptance_budget {
    static constexpr std::size_t max_draft = 8;
    static constexpr std::size_t review_passes = 4;
    static constexpr std::size_t initial_cooldown = 8;
    static constexpr std::size_t max_cooldown = 32;

    std::size_t cap = max_draft;
    std::size_t pending = 0;
    std::size_t cooldown = initial_cooldown;
    std::size_t until_probe = 0;
    std::size_t passes = 0;
    std::size_t accepted = 0;
    double measured_cost = 0.0;
    bool filled_cap = false;

    std::size_t next_cap() {
        if (cap > 0) {
            return cap;
        }
        if (until_probe > 0) {
            --until_probe;
            return 0;
        }
        return 1;
    }

    void observe(std::size_t accepted_now, const std::array<double, 9> & step_us) {
        const auto offered = std::exchange(pending, 0);
        if (offered == 0) {
            return;
        }
        if (cap == 0) {
            if (accepted_now > 0) {
                cap = 2;
                cooldown = initial_cooldown;
            } else {
                cooldown = std::min(cooldown * 2, max_cooldown);
                until_probe = cooldown;
            }
            return;
        }

        ++passes;
        accepted += accepted_now;
        measured_cost += step_us[offered];
        filled_cap |= accepted_now == cap;
        if (passes < review_passes) {
            return;
        }

        // Compare this window's emitted tokens per estimated millisecond with serial decode.
        double baseline_cost = (passes + accepted) * step_us[0];
        if (baseline_cost < 1.015 * measured_cost) {
            cap /= accepted == 0 ? 4 : 2;
            if (cap == 0) {
                until_probe = cooldown;
            }
        } else if (filled_cap) {
            cap = std::min(cap * 2, max_draft);
        }
        passes = 0;
        accepted = 0;
        measured_cost = 0.0;
        filled_cap = false;
    }
};

}
