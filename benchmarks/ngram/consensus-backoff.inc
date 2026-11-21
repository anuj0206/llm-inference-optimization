// Included in the pinned llama.cpp ngram-map.cpp by build-adaptive-ngram.py.
#include <algorithm>
#include <array>
#include <cstdlib>

static llama_tokens ngram_consensus_backoff(const common_ngram_simple_config & config,
                                            const llama_tokens & history, llama_token sampled) {
    static const bool enabled = [] {
        const char * value = std::getenv("LLM_OPT_NGRAM_BACKOFF");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    if (!enabled || config.size_ngram != 6 || config.size_mgram != 8) {
        return {};
    }

    constexpr std::size_t max_matches = 16;
    for (const std::size_t length : {4, 3, 2}) {
        if (history.size() <= length + 1) {
            continue;
        }
        std::array<std::size_t, max_matches> matches{};
        std::size_t count = 0;
        const auto suffix = history.end() - (length - 1);
        for (std::size_t start = history.size() - length - 1; start > 0 && count < max_matches; --start) {
            if (history[start + length - 1] == sampled &&
                std::equal(suffix, history.end(), history.begin() + start)) {
                matches[count++] = start + length;
            }
        }

        llama_tokens draft;
        for (std::size_t offset = 0; offset < 3 && count >= 3; ++offset) {
            llama_token winner = 0;
            std::size_t votes = 0;
            for (std::size_t i = 0; i < count; ++i) {
                if (matches[i] + offset >= history.size()) {
                    continue;
                }
                const auto candidate = history[matches[i] + offset];
                std::size_t support = 0;
                for (std::size_t j = 0; j < count; ++j) {
                    support += matches[j] + offset < history.size()
                        && history[matches[j] + offset] == candidate;
                }
                if (support > votes) {
                    winner = candidate;
                    votes = support;
                }
            }
            if (votes < 3 || votes * 4 < count * 3) {
                break;
            }
            draft.push_back(winner);
            count = std::remove_if(matches.begin(), matches.begin() + count, [&](std::size_t end) {
                return end + offset >= history.size() || history[end + offset] != winner;
            }) - matches.begin();
        }
        if (!draft.empty()) {
            return draft;
        }
    }
    return {};
}
