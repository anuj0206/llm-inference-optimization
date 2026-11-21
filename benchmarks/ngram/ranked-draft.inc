// Included in the pinned llama.cpp ngram-map.cpp by build-adaptive-ngram.py.
#include <array>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace {

constexpr std::size_t max_draft = 8;
constexpr std::size_t max_matches = 16;
constexpr std::size_t max_suffix = 6;
using suffix_tokens = std::array<llama_token, max_suffix>;

struct ngram_experiment {
    int probe_length = 0;
    bool ranked = false;
    bool rolling = false;
    std::array<double, max_draft + 1> step_us{};

    ngram_experiment() {
        if (const char * value = std::getenv("LLM_OPT_NGRAM_PROBE")) {
            if (value[0] < '1' || value[0] > '8' || value[1] != '\0') {
                throw std::runtime_error("LLM_OPT_NGRAM_PROBE must be a draft length from 1 to 8");
            }
            probe_length = value[0] - '0';
        }
        if (const char * value = std::getenv("LLM_OPT_NGRAM_POLICY")) {
            const std::string_view policy(value);
            rolling = policy == "rolling";
            ranked = rolling || policy == "ranked";
        }
        if (ranked) {
            const char * costs = std::getenv("LLM_OPT_NGRAM_STEP_US");
            std::istringstream input(costs ? costs : "");
            for (auto & cost : step_us) {
                if (!(input >> cost) || cost <= 0.0) {
                    throw std::runtime_error("LLM_OPT_NGRAM_STEP_US requires nine positive costs in microseconds");
                }
            }
        }
    }
};

struct ngram_matches {
    std::array<std::size_t, max_matches> ends{};
    std::size_t count = 0;
};

using ngram_match_table = std::array<ngram_matches, max_suffix + 1>;

ngram_match_table find_ngram_matches(const llama_tokens & history, const suffix_tokens & suffix) {
    ngram_match_table matches{};
    // Each end points to a confirmed token immediately after a matching suffix.
    for (std::size_t end = history.size() - 1; end >= 2; --end) {
        if (history[end - 1] != suffix.back()) {
            continue;
        }
        for (std::size_t length = 2; length <= max_suffix && length <= end; ++length) {
            if (history[end - length] != suffix[max_suffix - length]) {
                break;
            }
            auto & match = matches[length];
            if (match.count < max_matches) {
                match.ends[match.count++] = end;
            }
        }
        if (matches[max_suffix].count == max_matches) {
            break;
        }
    }
    return matches;
}

struct ngram_vote {
    llama_token token = 0;
    std::size_t count = 0;
};

ngram_vote vote_on_token(const llama_tokens & history, const ngram_matches & matches, std::size_t offset) {
    ngram_vote best;
    for (std::size_t i = 0; i < matches.count; ++i) {
        const std::size_t position = matches.ends[i] + offset;
        if (position >= history.size()) {
            continue;
        }
        const auto candidate = history[position];
        std::size_t votes = 0;
        for (std::size_t j = 0; j < matches.count; ++j) {
            const auto other = matches.ends[j] + offset;
            votes += other < history.size() && history[other] == candidate;
        }
        if (votes > best.count) {
            best = {candidate, votes};
        }
    }
    return best;
}

double ngram_prefix_probability(std::size_t votes, std::size_t total, std::size_t length) {
    // Discount sparse evidence, especially for shorter suffixes.
    constexpr std::array<double, max_suffix + 1> prior_misses{0.0, 0.0, 3.0, 1.5, 1.0, 0.75, 0.5};
    return votes / (total + prior_misses[length]);
}

llama_tokens ngram_ranked_draft(const ngram_experiment & experiment, const llama_tokens & history,
                               llama_token sampled) {
    if (history.size() < max_suffix - 1) {
        return {};
    }
    suffix_tokens suffix;
    std::copy_n(history.end() - (max_suffix - 1), max_suffix - 1, suffix.begin());
    suffix.back() = sampled;
    const auto matches = find_ngram_matches(history, suffix);
    llama_tokens best;
    double best_rate = 1.015 / experiment.step_us[0];
    const auto consider = [&](const llama_tokens & draft, double expected) {
        const double rate = (1.0 + expected) / experiment.step_us[draft.size()];
        if (rate > best_rate) {
            best_rate = rate;
            best = draft;
        }
    };
    for (std::size_t length = max_suffix; length >= 2; --length) {
        auto remaining = matches[length];
        llama_tokens draft;
        double expected = 0.0;
        for (std::size_t offset = 0; offset < max_draft; ++offset) {
            const auto vote = vote_on_token(history, remaining, offset);
            if (vote.count == 0) {
                break;
            }
            draft.push_back(vote.token);
            expected += ngram_prefix_probability(vote.count, matches[length].count, length);
            consider(draft, expected);
            const auto end = std::remove_if(
                remaining.ends.begin(), remaining.ends.begin() + remaining.count, [&](std::size_t position) {
                    return position + offset >= history.size() || history[position + offset] != vote.token;
                });
            remaining.count = end - remaining.ends.begin();
        }
    }
    if (experiment.rolling && best.size() < 4) {
        llama_tokens draft;
        double prefix_probability = 1.0;
        double expected = 0.0;
        auto next_matches = matches;
        for (std::size_t offset = 0; offset < 3; ++offset) {
            llama_token winner = 0;
            double probability = 0.0;
            for (std::size_t length = max_suffix; length >= 2; --length) {
                const auto & match = next_matches[length];
                const auto vote = vote_on_token(history, match, 0);
                const double score = ngram_prefix_probability(vote.count, match.count, length);
                if (score > probability) {
                    winner = vote.token;
                    probability = score;
                }
            }
            if (probability == 0.0) {
                break;
            }
            draft.push_back(winner);
            prefix_probability *= probability;
            expected += prefix_probability;
            consider(draft, expected);
            if (offset + 1 < 3) {
                std::move(suffix.begin() + 1, suffix.end(), suffix.begin());
                suffix.back() = winner;
                next_matches = find_ngram_matches(history, suffix);
            }
        }
    }
    return best;
}

}
