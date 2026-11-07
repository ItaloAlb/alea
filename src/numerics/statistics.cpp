// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "numerics/statistics.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace alea::numerics {

MeanError block_mean_error(std::span<const double> block_means) {
    const std::size_t m = block_means.size();
    MeanError result;
    if (m == 0) return result;

    double sum = 0.0;
    for (double x : block_means) sum += x;
    result.mean = sum / static_cast<double>(m);

    if (m < 2) return result;  // error not estimable from a single block

    double sq = 0.0;
    for (double x : block_means) {
        const double d = x - result.mean;
        sq += d * d;
    }
    const double sample_variance = sq / static_cast<double>(m - 1);
    result.error = std::sqrt(sample_variance / static_cast<double>(m));
    return result;
}

namespace {

// One row of the reblocking transformation.
struct Level {
    int    level     = 0;   // transformation number (block size 2^level)
    int    n_blocks  = 0;   // blocks at this level
    double sem       = 0.0; // standard error of the mean at this level
    double sem_err   = 0.0; // statistical uncertainty of the SEM
};

// Minimum blocks for the SEM (and its error-on-error) to be meaningful;
// below this the plateau search falls back progressively (tools/reblock.py).
constexpr int kMinBlocks = 16;

// Full reblocking transformation: repeatedly average adjacent pairs, dropping
// a trailing odd sample, until fewer than 2 blocks remain.
std::vector<Level> reblock_table(std::span<const double> x) {
    std::vector<Level> table;
    std::vector<double> cur(x.begin(), x.end());
    int level = 0;
    while (cur.size() >= 2) {
        const int n_blocks = static_cast<int>(cur.size());

        double sum = 0.0;
        for (double v : cur) sum += v;
        const double mean = sum / static_cast<double>(n_blocks);

        double sq = 0.0;
        for (double v : cur) {
            const double d = v - mean;
            sq += d * d;
        }
        const double var = sq / static_cast<double>(n_blocks - 1);

        Level L;
        L.level    = level;
        L.n_blocks = n_blocks;
        L.sem      = std::sqrt(var / static_cast<double>(n_blocks));
        L.sem_err  = L.sem / std::sqrt(2.0 * static_cast<double>(n_blocks - 1));
        table.push_back(L);

        if (cur.size() % 2 != 0) cur.pop_back();  // drop trailing odd sample
        for (std::size_t i = 0; i + 1 < cur.size(); i += 2)
            cur[i / 2] = 0.5 * (cur[i] + cur[i + 1]);
        cur.resize(cur.size() / 2);
        ++level;
    }
    return table;
}

// Smallest level whose increment to the next level is within the combined
// error-on-error; false when no candidate level pair satisfied it.
bool select_plateau(const std::vector<Level>& table, int& onset) {
    auto eoe = [](const Level& L) { return L.sem_err; };

    std::vector<const Level*> cand;
    for (const Level& L : table)
        if (L.n_blocks >= kMinBlocks) cand.push_back(&L);
    if (cand.size() < 4) {
        cand.clear();
        for (const Level& L : table)
            if (L.n_blocks >= 4) cand.push_back(&L);
    }
    if (cand.empty()) {
        onset = table.back().level;
        return false;
    }

    for (std::size_t i = 0; i + 1 < cand.size(); ++i) {
        const Level& a = *cand[i];
        const Level& b = *cand[i + 1];
        if (b.sem - a.sem <= eoe(a) + eoe(b)) {
            onset = a.level;
            return true;  // growth has stopped
        }
    }
    onset = cand.back()->level;
    return false;  // never stabilized: noise-dominated tail
}

// Largest SEM on the plateau (from the onset onward, ignoring the few-block
// noisy tail).  Taking the max keeps the error bar honest — never optimistic.
double plateau_error(const std::vector<Level>& table, int onset, const Level*& selected) {
    double best = 0.0;
    bool   have = false;
    for (const Level& L : table) {
        if (L.level < onset || L.n_blocks < kMinBlocks) continue;
        if (!have || L.sem > best) selected = &L;
        best = have ? std::max(best, L.sem) : L.sem;
        have = true;
    }
    if (!have) {
        for (const Level& L : table) {
            if (L.level < onset || L.n_blocks < 4) continue;
            if (!have || L.sem > best) selected = &L;
            best = have ? std::max(best, L.sem) : L.sem;
            have = true;
        }
    }
    if (!have) {
        // Fall back to the onset row itself.
        for (const Level& L : table) {
            if (L.level == onset) {
                selected = &L;
                return L.sem;
            }
        }
        if (!table.empty()) selected = &table.front();
        return table.empty() ? 0.0 : table.front().sem;
    }
    return best;
}

// Plateau, error and diagnostics of a reblocking table (single or pooled).
void assemble(ReblockResult& r, const std::vector<Level>& table) {
    r.naive_error = table.front().sem;

    int onset = table.front().level;
    r.plateau_found = select_plateau(table, onset);
    r.level = onset;
    const Level* selected = nullptr;
    r.error = plateau_error(table, onset, selected);
    if (selected != nullptr) {
        r.selected_level = selected->level;
        r.selected_n_blocks = static_cast<std::size_t>(selected->n_blocks);
        r.error_on_error = selected->sem_err;
    }

    if (r.naive_error > 0.0 && r.error > 0.0) {
        const double inefficiency = (r.error / r.naive_error);
        const double s = inefficiency * inefficiency;
        r.tau_int = 0.5 * (s - 1.0);
        if (r.tau_int < 0.0) r.tau_int = 0.0;
        r.block_length = std::max(1, static_cast<int>(std::ceil(s)));
    }
}

double mean_of(std::span<const double> series) {
    double sum = 0.0;
    for (double v : series) sum += v;
    return sum / static_cast<double>(series.size());
}

}  // namespace

ReblockResult reblock(std::span<const double> series) {
    ReblockResult r;
    if (series.empty()) return r;
    r.mean = mean_of(series);

    r.selected_n_blocks = series.size();
    if (series.size() < 2) return r;

    const std::vector<Level> table = reblock_table(series);
    if (table.empty()) return r;
    assemble(r, table);
    return r;
}

ReblockResult reblock_chains(std::span<const double> chains, std::size_t n_chains) {
    if (n_chains == 0 || chains.size() % n_chains != 0)
        throw std::invalid_argument(
            "reblock_chains: the series does not split into equal-length chains");
    if (n_chains == 1) return reblock(chains);

    ReblockResult r;
    if (chains.empty()) return r;
    r.mean = mean_of(chains);
    r.selected_n_blocks = chains.size();

    const std::size_t length = chains.size() / n_chains;
    if (length < 2) return r;  // no chain has a variance

    std::vector<std::vector<Level>> per_chain;  // finalize-time scratch
    per_chain.reserve(n_chains);
    for (std::size_t i = 0; i < n_chains; ++i)
        per_chain.push_back(reblock_table(chains.subspan(i * length, length)));

    // Equal lengths give every chain the same number of levels.
    const std::size_t n_levels = per_chain.front().size();
    const double k = static_cast<double>(n_chains);
    std::vector<Level> pooled(n_levels);
    for (std::size_t t = 0; t < n_levels; ++t) {
        double sem2 = 0.0;
        int n_blocks = 0, dof = 0;
        for (const auto& table : per_chain) {
            sem2 += table[t].sem * table[t].sem;
            n_blocks += table[t].n_blocks;
            dof += table[t].n_blocks - 1;
        }
        Level& L = pooled[t];
        L.level    = static_cast<int>(t);
        L.n_blocks = n_blocks;
        L.sem      = std::sqrt(sem2) / k;
        L.sem_err  = L.sem / std::sqrt(2.0 * static_cast<double>(dof));
    }
    assemble(r, pooled);
    return r;
}

RatioResult ratio_of_means(std::span<const double> num,
                           std::span<const double> den) {
    if (num.size() != den.size())
        throw std::invalid_argument("ratio_of_means: series lengths differ");
    RatioResult r;
    const std::size_t n = num.size();
    if (n == 0) return r;
    r.numerator = reblock(num);
    r.denominator = reblock(den);
    double sum_num = 0.0, sum_den = 0.0;
    for (std::size_t b = 0; b < n; ++b) {
        sum_num += num[b];
        sum_den += den[b];
    }
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    if (sum_den == 0.0) {
        r.mean = r.error = r.jackknife_bias = nan;
        return r;
    }
    r.mean = sum_num / sum_den;

    // Coarse blocks of 2^level inputs, as the reblocking pairs them (the
    // trailing remainder is left out of the jackknife, not of the mean).
    int level = std::max(r.numerator.selected_level, r.denominator.selected_level);
    while (level > 0 && (n >> level) < 2) --level;
    const std::size_t width = std::size_t{1} << level;
    const std::size_t n_coarse = n / width;
    r.level = level;
    r.n_coarse_blocks = n_coarse;
    if (n_coarse < 2) return r;

    double total_num = 0.0, total_den = 0.0;
    for (std::size_t b = 0; b < n_coarse * width; ++b) {
        total_num += num[b];
        total_den += den[b];
    }
    std::vector<double> leave_one_out(n_coarse);  // finalize-time scratch
    double mean_loo = 0.0;
    for (std::size_t j = 0; j < n_coarse; ++j) {
        double block_num = 0.0, block_den = 0.0;
        for (std::size_t b = j * width; b < (j + 1) * width; ++b) {
            block_num += num[b];
            block_den += den[b];
        }
        const double rest_den = total_den - block_den;
        leave_one_out[j] = rest_den == 0.0 ? nan : (total_num - block_num) / rest_den;
        mean_loo += leave_one_out[j];
    }
    mean_loo /= static_cast<double>(n_coarse);
    double spread = 0.0;
    for (double value : leave_one_out) spread += (value - mean_loo) * (value - mean_loo);
    const double nc = static_cast<double>(n_coarse);
    r.error = std::sqrt((nc - 1.0) / nc * spread);
    r.jackknife_bias = (nc - 1.0) * (mean_loo - total_num / total_den);
    r.reliable = r.numerator.plateau_found && r.denominator.plateau_found &&
                 n_coarse >= 16 && std::isfinite(r.error);
    return r;
}

namespace {
// Median of v (reordered in place); v non-empty.
double median_in_place(std::vector<double>& v) {
    const std::size_t n = v.size(), mid = n / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
    const double upper = v[mid];
    if (n % 2 == 1) return upper;
    // nth_element left every value of [0, mid) at most v[mid]: the lower
    // middle value is their maximum.
    const double lower = *std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid));
    return 0.5 * (lower + upper);
}
}  // namespace

double robust_spread(std::span<const double> x, std::vector<double>& scratch) {
    if (x.empty()) return 0.0;
    scratch.assign(x.begin(), x.end());
    const double centre = median_in_place(scratch);
    for (std::size_t i = 0; i < x.size(); ++i) scratch[i] = std::abs(x[i] - centre);
    return 1.4826 * median_in_place(scratch);
}

AutocorrelationResult integrated_autocorrelation(std::span<const double> series,
                                                 std::size_t n_chains,
                                                 double window_factor) {
    if (n_chains == 0)
        throw std::invalid_argument("integrated_autocorrelation: no chains");
    if (series.size() % n_chains != 0)
        throw std::invalid_argument(
            "integrated_autocorrelation: length is not a multiple of the chain count");
    if (!(window_factor > 0.0))
        throw std::invalid_argument(
            "integrated_autocorrelation: window factor must be positive");

    AutocorrelationResult r;
    const std::size_t K = n_chains;
    const std::size_t n = series.size() / K;
    r.n_steps = n;
    if (n < 2) return r;

    double mean = 0.0;
    for (double x : series) mean += x;
    mean /= static_cast<double>(series.size());

    // C(k) for k = 0..n_steps/4, each averaged over the K·(n − k) pairs.
    auto covariance = [&](std::size_t k) {
        double s = 0.0;
        for (std::size_t t = 0; t + k < n; ++t) {
            const double* a = series.data() + t * K;
            const double* b = series.data() + (t + k) * K;
            for (std::size_t c = 0; c < K; ++c) s += (a[c] - mean) * (b[c] - mean);
        }
        return s / static_cast<double>(K * (n - k));
    };
    const double c0 = covariance(0);
    if (!(c0 > 0.0)) return r;

    const std::size_t max_lag = n / 4;
    double sum = 0.0;   // Σ_{k=1..W} ρ(k)
    for (std::size_t w = 1; w <= max_lag; ++w) {
        sum += covariance(w) / c0;
        r.window  = w;
        r.tau_int = sum;
        if (static_cast<double>(w) >= window_factor * (0.5 + sum)) {
            r.window_found = true;
            break;
        }
    }
    return r;
}

}  // namespace alea::numerics