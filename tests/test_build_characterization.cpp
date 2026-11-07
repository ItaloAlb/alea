// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Characterization net for the config → (ψ, H) construction path.
//
// Every YAML config shipped with the repository is built through the real
// App::build_drivers, and the resulting trial wavefunction and Hamiltonian are
// evaluated on the first few initial walkers (seed forced, so the positions are
// deterministic).  What is pinned:
//
//   * the multiset of components — name plus every parameter (name, value,
//     fixed flag) — independent of the order they were added in;
//   * log|ψ| up to a walker-independent constant, and E_L (real and imaginary
//     part), per walker; both are sums over components and therefore also
//     order-independent;
//   * or, when building throws, the error message.
//
// It exists to prove that refactoring the builders changes nothing physical.
// Regenerate the golden file ONLY when a change is meant to alter the physics:
//
//   ALEA_UPDATE_GOLDEN=1 ./test_build_characterization
//
// A config that is on disk but absent from the golden file fails (regenerate
// deliberately); a golden entry whose config was deleted is ignored.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "io/parser.h"
#include "population/walker_set.h"
#include "qmc/app.h"

namespace fs = std::filesystem;

namespace {

constexpr std::size_t kWalkers = 4;
constexpr double      kTol     = 1e-10;   // relative, with an absolute floor

// The record must hold on any machine, so it pins only digits the build
// determines.  In a periodic cell the Ewald sum and the image-summed Padé
// Jastrow go through the long-range breakup, a least-squares fit with
// cond ~1e5-1e6: round-off differences between builds (for example, sin, cos
// and exp differ in the last bit between math libraries) come out amplified.
// A GCC 11.5/glibc build differs from the GCC 15.2/UCRT one that wrote the
// record by
//
//   * a walker-independent 3e-10 to 5e-9 in log|ψ|, while the differences
//     between walkers agree to 3e-14.  A constant in log|ψ| is ψ's
//     normalization, which no ratio, drift, local energy or optimizer step
//     sees, so log|ψ| is compared relative to walker 0 (lossless);
//   * 1.2e-9 Ha in E_L (heg2d_twisted).  Periodic E_L is compared at
//     kBreakupTol, the accuracy the breakup is held to anyway (the Madelung
//     test; the Jastrow fit's noise floor); digits below it carry no physics.
constexpr double kBreakupTol = 1e-8;   // relative, with an absolute floor

std::string fmt(double x) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", x);
    return buf;
}

// One config's fingerprint, as text lines.  Line 0 is either "error: ..." or
// "components" followed by sorted component lines, then one line per walker,
// "  w<i> log|ψ| E_L Im E_L".
struct Fingerprint {
    std::vector<std::string> lines;
    bool periodic = false;   // the cell is periodic: E_L is compared at kBreakupTol
};

Fingerprint fingerprint(const fs::path& path) {
    Fingerprint fp;
    std::vector<std::string>& out = fp.lines;
    try {
        alea::io::Config cfg = alea::io::parse_config_file(path.string());
        cfg.seed   = 12345;   // deterministic initial positions
        cfg.log    = false;
        cfg.output = false;
        alea::App app(cfg);
        alea::DriverPack pack = app.build_drivers();
        if (!pack.psi || !pack.hamiltonian || pack.ensembles.empty()) {
            out.push_back("no-drivers");
            return fp;
        }

        std::vector<std::string> comps;
        for (const auto& c : pack.psi->components()) {
            std::string line = "  " + c->name() + " :";
            for (const alea::Parameter& p : c->parameters())
                line += " " + p.name + "=" + fmt(p.value) + (p.is_fixed ? "f" : "v");
            comps.push_back(line);
        }
        std::sort(comps.begin(), comps.end());
        out.push_back("components " + std::to_string(comps.size()));
        out.insert(out.end(), comps.begin(), comps.end());

        alea::WalkerEnsemble& ens = *pack.ensembles.front();
        fp.periodic = ens.cell().is_periodic();
        const std::size_t nw = std::min(kWalkers, ens.num_walkers());
        alea::WalkerSet ws(nw, ens.n_particles(), ens.dim(), ens.cell());
        for (std::size_t w = 0; w < nw; ++w) ws.add_walker(ens.walkers()[w]);
        ws.load_all();
        alea::TrialWavefunction psi = pack.psi->clone();
        alea::Hamiltonian       ham = pack.hamiltonian->clone();
        psi.evaluate_batch(ws);
        ham.evaluate_batch(ws);
        for (std::size_t w = 0; w < nw; ++w) {
            const double im = ws.has_phase() ? ws.local_energies_imag()[w] : 0.0;
            out.push_back("  w" + std::to_string(w) + " " + fmt(ws.log_psi()[w]) +
                          " " + fmt(ws.local_energies()[w]) + " " + fmt(im));
        }
    } catch (const std::exception& e) {
        std::string msg = e.what();
        std::replace(msg.begin(), msg.end(), '\n', ' ');
        out.assign(1, "error: " + msg);
    }
    return fp;
}

std::vector<fs::path> all_configs() {
    const fs::path root = ALEA_REPO_DIR;
    std::vector<fs::path> files;
    for (const char* dir : {"configs", "configs/tests", "tests/configs"})
        for (const auto& e : fs::directory_iterator(root / dir))
            if (e.is_regular_file() && e.path().extension() == ".yaml")
                files.push_back(fs::relative(e.path(), root));
    std::sort(files.begin(), files.end());
    return files;
}

fs::path golden_path() {
    return fs::path(ALEA_REPO_DIR) / "tests" / "golden" / "build_characterization.txt";
}

// "config <path>" headers followed by that config's lines.
std::map<std::string, std::vector<std::string>> read_golden() {
    std::map<std::string, std::vector<std::string>> g;
    std::ifstream in(golden_path());
    std::string line, current;
    while (std::getline(in, line)) {
        if (line.rfind("config ", 0) == 0) {
            current = line.substr(7);
            g[current];
        } else if (!current.empty()) {
            g[current].push_back(line);
        }
    }
    return g;
}

bool close(double x, double y, double tol, double scale = 0.0) {
    return std::abs(x - y) <= tol * std::max({1.0, scale, std::abs(x), std::abs(y)});
}

bool numbers_match(const std::string& a, const std::string& b) {
    std::istringstream sa(a), sb(b);
    std::string ta, tb;
    while (sa >> ta) {
        if (!(sb >> tb)) return false;
        if (ta == tb) continue;
        char* ea = nullptr;
        char* eb = nullptr;
        const double x = std::strtod(ta.c_str(), &ea);
        const double y = std::strtod(tb.c_str(), &eb);
        if (*ea != '\0' || *eb != '\0') return false;   // non-numeric token differs
        if (!close(x, y, kTol)) return false;
    }
    return !(sb >> tb);
}

// "  w<i> log|ψ| E_L Im E_L" → label and the three numbers; false otherwise.
bool parse_walker(const std::string& s, std::string& label, double (&v)[3]) {
    std::istringstream ss(s);
    std::string extra;
    return ss >> label >> v[0] >> v[1] >> v[2] && !(ss >> extra);
}

// log|ψ| is compared as its difference from walker 0, at the tolerance of the
// absolute values (their round-off sets the difference's); E_L at kBreakupTol
// in a periodic cell.
bool walker_lines_match(const std::string& a, const std::string& b, const std::string& a0,
                        const std::string& b0, bool periodic) {
    std::string la, lb, la0, lb0;
    double va[3], vb[3], va0[3], vb0[3];
    if (!parse_walker(a, la, va) || !parse_walker(b, lb, vb) || !parse_walker(a0, la0, va0) ||
        !parse_walker(b0, lb0, vb0) || la != lb)
        return false;
    const double scale = std::max({std::abs(va[0]), std::abs(vb[0]), std::abs(va0[0]),
                                   std::abs(vb0[0])});
    const double e_tol = periodic ? kBreakupTol : kTol;
    return close(va[0] - va0[0], vb[0] - vb0[0], kTol, scale) && close(va[1], vb[1], e_tol) &&
           close(va[2], vb[2], e_tol);
}

// Component lines carry "name=value[f|v]" tokens; compare the value numerically.
bool component_lines_match(const std::string& a, const std::string& b) {
    auto split = [](const std::string& s) {
        std::vector<std::string> t;
        std::istringstream ss(s);
        std::string tok;
        while (ss >> tok) t.push_back(tok);
        return t;
    };
    const auto ta = split(a), tb = split(b);
    if (ta.size() != tb.size()) return false;
    for (std::size_t i = 0; i < ta.size(); ++i) {
        if (ta[i] == tb[i]) continue;
        const auto ea = ta[i].find('='), eb = tb[i].find('=');
        if (ea == std::string::npos || eb == std::string::npos) return false;
        if (ta[i].substr(0, ea) != tb[i].substr(0, eb)) return false;
        if (ta[i].back() != tb[i].back()) return false;   // fixed flag
        const std::string va = ta[i].substr(ea + 1, ta[i].size() - ea - 2);
        const std::string vb = tb[i].substr(eb + 1, tb[i].size() - eb - 2);
        if (!numbers_match(va, vb)) return false;
    }
    return true;
}

}  // namespace

TEST_CASE("every shipped config builds the same psi and H as the golden record",
          "[characterization]") {
    const std::vector<fs::path> configs = all_configs();
    REQUIRE(!configs.empty());

    if (std::getenv("ALEA_UPDATE_GOLDEN")) {
        fs::create_directories(golden_path().parent_path());
        std::ofstream out(golden_path());
        for (const fs::path& p : configs) {
            out << "config " << p.generic_string() << '\n';
            for (const std::string& l : fingerprint(fs::path(ALEA_REPO_DIR) / p).lines)
                out << l << '\n';
        }
        WARN("golden file regenerated: " << golden_path().string());
        return;
    }

    const auto golden = read_golden();
    REQUIRE(!golden.empty());
    for (const fs::path& p : configs) {
        const std::string key = p.generic_string();
        INFO("config " << key);
        const auto it = golden.find(key);
        REQUIRE(it != golden.end());   // new config: regenerate deliberately
        const Fingerprint fp = fingerprint(fs::path(ALEA_REPO_DIR) / p);
        const std::vector<std::string>& now = fp.lines;
        const std::vector<std::string>& was = it->second;
        REQUIRE(now.size() == was.size());
        std::size_t w0 = now.size();   // walker 0's line
        for (std::size_t i = 0; i < now.size(); ++i) {
            INFO("got:      " << now[i]);
            INFO("expected: " << was[i]);
            const bool is_walker = now[i].rfind("  w", 0) == 0;
            if (is_walker && w0 == now.size()) w0 = i;
            if (is_walker)
                CHECK(walker_lines_match(now[i], was[i], now[w0], was[w0], fp.periodic));
            else if (i > 0)
                CHECK(component_lines_match(now[i], was[i]));
            else
                CHECK(numbers_match(now[i], was[i]));
        }
    }
}
