// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>

#include "numerics/random.h"

namespace alea::numerics {

// xoshiro256++ 1.0 (D. Blackman and S. Vigna, "Scrambled linear pseudorandom
// number generators", ACM Trans. Math. Softw. 47(4), 36 (2021); the reference
// xoshiro256plusplus.c).  Period 2²⁵⁶ − 1, 32 bytes of state.  Unlike
// xoshiro256+, its low bits have no linear artifacts, so XoshiroRandom may take
// the ziggurat region from them.
class Xoshiro256pp {
public:
    // State from four consecutive splitmix64 outputs, the seeding the authors
    // recommend.  splitmix64 is a bijection, so the words are never all zero.
    explicit Xoshiro256pp(std::uint64_t seed) noexcept {
        for (std::uint64_t i = 0; i < state_.size(); ++i)
            state_[i] = splitmix64(seed + i * kSplitmixGamma);
    }
    // A given state (not all zero), for comparison with the reference code.
    explicit Xoshiro256pp(const std::array<std::uint64_t, 4>& state) noexcept : state_(state) {}

    std::uint64_t operator()() noexcept {
        const std::uint64_t result = std::rotl(state_[0] + state_[3], 23) + state_[0];
        const std::uint64_t t      = state_[1] << 17;
        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= t;
        state_[3] = std::rotl(state_[3], 45);
        return result;
    }

private:
    static constexpr std::uint64_t kSplitmixGamma = 0x9e3779b97f4a7c15ULL;   // splitmix64's increment
    std::array<std::uint64_t, 4> state_{};
};

// Random numbers for the VMC and DMC sweeps and the DMC branching: xoshiro256++
// bits, uniforms and ziggurat Gaussians.  PIMC keeps numerics::Random.
//
// Uniforms: the top 53 bits times 2⁻⁵³, i.e. k·2⁻⁵³ with k uniform on
// [0, 2⁵³), the lattice std::generate_canonical gives.
//
// Gaussians: the ziggurat of G. Marsaglia and W. W. Tsang, "The ziggurat
// method for generating random variables", J. Stat. Softw. 5(8), 1 (2000),
// with 256 regions of equal area v under f(x) = exp(−x²/2) on x ≥ 0: the base
// strip [0, v/f(r)) × [0, f(r)), whose part beyond r stands for the tail, and
// 255 layers i = 1..255 (1 at the top), [0, xᵢ) × [f(xᵢ), f(xᵢ₋₁)), with
// x₂₅₅ = r and f(x₀) = 1 above the top layer.  Pick a region uniformly and a
// point x uniformly along it:
//   - x < xᵢ₋₁ (inside the column under the layer above; x < r in the base):
//     the whole column lies under f, return x (~99 % of draws);
//   - base strip, x ≥ r: return an exact tail sample r + ξ (G. Marsaglia,
//     "Generating a variable from the tail of the normal distribution",
//     Technometrics 6(1), 101 (1964)): ξ = −ln U₁ / r, accepted when
//     −2 ln U₂ > ξ²;
//   - otherwise accept x when a uniform height in the layer falls under f(x),
//     else start again.
// Every accepted point is uniform under f, so the result is exactly
// half-normal; an independent bit gives the sign.  The region index, the sign
// and the 52-bit position come from disjoint bits of one 64-bit output, which
// avoids the index-value correlation of the original 32-bit version
// (J. A. Doornik, "An improved ziggurat method to generate normal random
// samples", Nuffield College, Oxford, 2005); numpy's random_standard_normal
// uses the same split.
//
// Cost on the Xeon E5-2680 v4, in the sweep pattern (dim Gaussians and one
// uniform per particle and walker): 2.5 ns per number, versus 15.6 ns for
// std::mt19937_64 with std::normal_distribution (Marsaglia's polar method,
// whose rejection branch is unpredictable).
class XoshiroRandom {
public:
    explicit XoshiroRandom(std::uint64_t seed) noexcept
        : engine_(seed), ziggurat_(&normal_ziggurat()) {}

    // Uniform sample in [0, 1).
    double uniform() noexcept {
        return static_cast<double>(static_cast<std::int64_t>(engine_() >> 11)) * 0x1.0p-53;
    }

    // Standard normal sample (mean 0, standard deviation 1).
    double gaussian() noexcept {
        const Ziggurat& z = *ziggurat_;
        for (;;) {
            const std::uint64_t bits     = engine_();
            const std::size_t   i        = bits & 0xff;                  // region
            const std::uint64_t sign     = (bits & 0x100) << 55;         // bit 8 → bit 63
            const std::uint64_t position = (bits >> 9) & kPositionMask;  // 52 bits
            const double x = static_cast<double>(static_cast<std::int64_t>(position)) * z.w[i];
            if (position < z.k[i]) return with_sign(x, sign);
            if (i == 0) return with_sign(tail(z.r), sign);
            if (z.f[i] + uniform() * (z.f[i - 1] - z.f[i]) < std::exp(-0.5 * x * x))
                return with_sign(x, sign);
        }
    }

private:
    static constexpr std::uint64_t kPositionMask = (std::uint64_t{1} << 52) - 1;

    // Ziggurat tables, 2⁵² fixed point: a region-i position p < 2⁵² maps to
    // x = p·w[i]; the fast path is p < k[i]; f[i] = f(xᵢ) bounds the layer.
    //   k[0] = 2⁵²·r/x̃₀, w[0] = x̃₀/2⁵², f[0] = 1, with x̃₀ = v/f(r);
    //   k[1] = 0 (no column under the top layer);
    //   k[i] = 2⁵²·xᵢ₋₁/xᵢ, w[i] = xᵢ/2⁵², f[i] = f(xᵢ) for i ≥ 1.
    struct Ziggurat {
        std::array<std::uint64_t, 256> k{};
        std::array<double, 256>        w{};
        std::array<double, 256>        f{};
        double                         r = 0.0;   // start of the tail
    };

    // Built once.  The layers follow from r and v(r) = r f(r) + ∫ᵣ^∞ f by
    // xᵢ₋₁ = f⁻¹(f(xᵢ) + v/xᵢ), and r is the root of "the top layer closes at
    // f = 1", f(x₁) + v/x₁ = 1, found by bisection: a larger r makes v and
    // every layer thinner.  The root, 3.6541528853610092, matches Marsaglia
    // and Tsang's 3.6541528853610088 to 1e-16, so the regions are equal to
    // double precision.
    static Ziggurat make_normal_ziggurat() {
        constexpr int n = 256;
        auto f   = [](double x) { return std::exp(-0.5 * x * x); };
        auto v_r = [&](double r) {
            return r * f(r) + std::sqrt(0.5 * std::numbers::pi) * std::erfc(r / std::numbers::sqrt2);
        };
        std::array<double, n> x{};   // layer edges x[1..255]
        // f(x₁) + v/x₁ − 1, or +1 when the layers reach f = 1 before the top.
        auto top_mismatch = [&](double r) {
            const double v = v_r(r);
            x[n - 1] = r;
            for (int i = n - 1; i >= 2; --i) {
                const double y = f(x[i]) + v / x[i];
                if (y >= 1.0) return 1.0;
                x[i - 1] = std::sqrt(-2.0 * std::log(y));
            }
            return f(x[1]) + v / x[1] - 1.0;
        };
        double lo = 3.0, hi = 4.5;   // mismatch > 0 at lo, < 0 at hi
        for (;;) {
            const double mid = 0.5 * (lo + hi);
            if (mid == lo || mid == hi) break;
            (top_mismatch(mid) > 0.0 ? lo : hi) = mid;
        }
        Ziggurat z;
        z.r = hi;
        (void)top_mismatch(z.r);   // leaves the edges of r in x
        const double two52 = 0x1.0p52;
        const double base  = v_r(z.r) / f(z.r);   // x̃₀
        z.k[0] = static_cast<std::uint64_t>(two52 * (z.r / base));
        z.w[0] = base / two52;
        z.f[0] = 1.0;
        for (int i = 1; i < n; ++i) {
            z.k[i] = i == 1 ? 0 : static_cast<std::uint64_t>(two52 * (x[i - 1] / x[i]));
            z.w[i] = x[i] / two52;
            z.f[i] = f(x[i]);
        }
        return z;
    }

    static const Ziggurat& normal_ziggurat() {
        static const Ziggurat z = make_normal_ziggurat();
        return z;
    }

    // Sign by bit flip: a random sign bit as a branch mispredicts half the
    // time (5.4 instead of 3.2 ns per Gaussian on the Xeon).
    static double with_sign(double x, std::uint64_t sign) noexcept {
        return std::bit_cast<double>(std::bit_cast<std::uint64_t>(x) ^ sign);
    }

    // Marsaglia's tail beyond r; 1 − U ∈ (0, 1] keeps the logarithms finite.
    double tail(double r) noexcept {
        for (;;) {
            const double xi = -std::log1p(-uniform()) / r;
            const double y  = -std::log1p(-uniform());
            if (y + y > xi * xi) return r + xi;
        }
    }

    Xoshiro256pp    engine_;
    const Ziggurat* ziggurat_;   // the shared tables, looked up once
};

}  // namespace alea::numerics
