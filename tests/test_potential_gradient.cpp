// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Independent contracts for G=grad V, not grad(log psi): analytic limits and
// central differences of the actual sampled potential, with unchanged state.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "hamiltonian/hamiltonian.h"
#include "hamiltonian/coulomb.h"
#include "hamiltonian/harmonic_trap.h"
#include "hamiltonian/kinetic.h"
#include "hamiltonian/moire.h"
#include "hamiltonian/rytova_keldysh.h"
#include "hamiltonian/ewald_term.h"
#include "hamiltonian/screened_coulomb.h"
#include "pimc/path_set.h"

namespace {
thread_local bool track_allocations = false;
thread_local std::size_t allocations = 0;
}
#if defined(__GNUC__)
#define ALEA_NOINLINE __attribute__((noinline))
#else
#define ALEA_NOINLINE
#endif
ALEA_NOINLINE void* operator new(std::size_t size) {
    if (track_allocations) ++allocations;
    if (void* ptr = std::malloc(size ? size : 1)) return ptr;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
ALEA_NOINLINE void operator delete(void* ptr) noexcept { std::free(ptr); }
ALEA_NOINLINE void operator delete[](void* ptr) noexcept { std::free(ptr); }
ALEA_NOINLINE void operator delete(void* ptr, std::size_t) noexcept { std::free(ptr); }
ALEA_NOINLINE void operator delete[](void* ptr, std::size_t) noexcept { std::free(ptr); }

namespace {
using namespace alea;
using Catch::Approx;

class UnsupportedGradient final : public HamiltonianComponent {
public:
    std::string name() const override { return "unsupported_test_potential"; }
    std::unique_ptr<HamiltonianComponent> clone() const override {
        return std::make_unique<UnsupportedGradient>(*this);
    }
};

ParticleGroup group(std::size_t first, std::size_t last, double charge,
                    std::size_t dim) {
    ParticleGroup g{};
    g.first=first; g.last=last; g.count=last-first; g.charge=charge;
    g.mass.resize(dim); g.inv_mass.resize(dim);
    for (std::size_t d=0; d<dim; ++d) {
        g.mass[d]=1.0+d; g.inv_mass[d]=1.0/g.mass[d];
    }
    return g;
}

void initialize(ParticleBatch& batch) {
    const auto nw=batch.num_walkers(), dim=batch.dim();
    for (std::size_t p=0; p<batch.n_particles(); ++p)
        for (std::size_t w=0; w<nw; ++w) {
            double position[SimulationCell::kMaxDim]; // [dim]
            for (std::size_t d=0; d<dim; ++d)
                position[d]=-0.8+0.83*p+0.16*w+0.13*d+0.07*p*d;
            batch.cell().wrap(position);
            for (std::size_t d=0; d<dim; ++d)
                batch.positions()[(p*dim+d)*nw+w]=position[d];
        }
    batch.dist_tables().build_all(batch.positions());
}

// Differentiate the scalar V, independently of every analytic force formula.
// Three shrinking steps expose truncation/roundoff rather than trusting one h.
void check_finite_differences(const Hamiltonian& ham, WalkerSet& batch) {
    auto workspace=ham.make_potential_gradient_workspace(batch);
    std::vector<double> gradient(batch.positions().size()); // [np*dim*nw]
    ham.evaluate_potential_gradient(batch,gradient,workspace);
    std::vector<double> values(batch.num_walkers()); // [nw]
    const auto nw=batch.num_walkers(), dim=batch.dim();
    for (double step : {1e-4,5e-5,2e-5})
        for (std::size_t p=0; p<batch.n_particles(); ++p)
            for (std::size_t d=0; d<dim; ++d)
                for (std::size_t w=0; w<nw; ++w) {
                    const std::size_t index=(p*dim+d)*nw+w;
                    const double saved=batch.positions()[index];
                    batch.positions()[index]=saved+step;
                    batch.refresh_particle_tables(p,w,w+1);
                    ham.evaluate_potential(batch,values,w,w+1);
                    const double plus=values[w];
                    batch.positions()[index]=saved-step;
                    batch.refresh_particle_tables(p,w,w+1);
                    ham.evaluate_potential(batch,values,w,w+1);
                    const double minus=values[w];
                    batch.positions()[index]=saved;
                    batch.refresh_particle_tables(p,w,w+1);
                    CAPTURE(p,d,w,step);
                    CHECK(gradient[index]==Approx((plus-minus)/(2*step)).margin(2e-7));
                }
}

Hamiltonian composite(std::size_t dim) {
    const auto a=group(0,2,-1,dim), b=group(2,3,2,dim);
    Hamiltonian ham;
    ham.add_component(std::make_unique<Kinetic>(std::vector{a,b}));
    ham.add_component(std::make_unique<CoulombTerm>(a,a,2.5));
    ham.add_component(std::make_unique<CoulombTerm>(a,b,3.0,0.7));
    ham.add_component(std::make_unique<HarmonicTrap>(std::vector{a,b},0.8,
                                                    std::vector<double>(dim,0.3)));
    ham.add_component(std::make_unique<MoireTerm>(dim,1.2,0.17,0.4,
        moire_reciprocal_vectors(5.0,0.23,dim),std::vector<std::pair<std::size_t,std::size_t>>{{0,1},{2,3}}));
    return ham;
}
}

TEST_CASE("Analytic potential gradients differentiate the sampled composite in 1D 2D and 3D", "[potential-gradient]") {
    for (std::size_t dim : {1,2,3}) {
        DYNAMIC_SECTION("dimension " << dim) {
            WalkerSet batch(5,3,dim);
            initialize(batch);
            const auto ham=composite(dim);
            CHECK(ham.supports_potential_gradient());
            check_finite_differences(ham,batch);
        }
    }
}

TEST_CASE("Trap gradients preserve anisotropic mass signed curvature and selected particles", "[potential-gradient]") {
    WalkerSet batch(3,3,3);
    initialize(batch);
    const auto a=group(0,1,0,3), b=group(2,3,0,3);
    HarmonicTrap trap({a,b},-2.0,{0.2,-0.4,0.6});
    auto workspace=trap.make_potential_gradient_workspace(batch);
    std::vector<double> gradient(batch.positions().size(),7.0); // [np*dim*nw]
    trap.evaluate_potential_gradient(batch,gradient,*workspace);
    for (std::size_t p=0; p<3; ++p)
        for (std::size_t d=0; d<3; ++d)
            for (std::size_t w=0; w<3; ++w) {
                const auto index=(p*3+d)*3+w;
                const double center=std::array{0.2,-0.4,0.6}[d];
                const double expected=p==1 ? 7.0 : 7.0-2.0*(1+d)*(batch.positions()[index]-center);
                CHECK(gradient[index]==Approx(expected));
            }
}

TEST_CASE("Coulomb gradients use canonical periodic directions and equal opposite pair forces", "[potential-gradient]") {
    for (const auto& vectors : {std::vector<double>{4.0},std::vector<double>{4,0,0,5},
                                std::vector<double>{4,0,2,3.4641016151377544}}) {
        const std::size_t dim=vectors.size()==1 ? 1 : 2;
        DYNAMIC_SECTION("periodic cell " << vectors.size() << " last " << vectors.back()) {
            SimulationCell cell(dim,vectors);
            WalkerSet batch(3,3,dim,cell);
            initialize(batch);
            // Cross the coordinate boundary without approaching an image tie.
            batch.positions()[0]=-1.9;
            batch.positions()[dim*3]=1.7;
            batch.dist_tables().build_all(batch.positions());
            const auto a=group(0,2,-1,dim), b=group(2,3,1,dim);
            Hamiltonian ham;
            ham.add_component(std::make_unique<CoulombTerm>(a,a,2.0));
            ham.add_component(std::make_unique<CoulombTerm>(a,b,3.0,0.6));
            check_finite_differences(ham,batch);
            auto workspace=ham.make_potential_gradient_workspace(batch);
            std::vector<double> gradient(batch.positions().size()); // [np*dim*nw]
            ham.evaluate_potential_gradient(batch,gradient,workspace);
            for (std::size_t d=0; d<dim; ++d)
                for (std::size_t w=0; w<3; ++w)
                    CHECK(std::abs(gradient[d*3+w]+gradient[(dim+d)*3+w]+gradient[(2*dim+d)*3+w])<1e-13);
        }
    }
}

TEST_CASE("Bounded Coulomb coincidence has zero gradient while a true pole is rejected", "[potential-gradient]") {
    WalkerSet batch(2,2,2);
    const auto a=group(0,1,-1,2),b=group(1,2,1,2);
    for (double separation : {0.7,1e-100}) {
        CoulombTerm term(a,b,2.0,separation);
        auto workspace=term.make_potential_gradient_workspace(batch);
        std::vector<double> gradient(8,0.0);
        term.evaluate_potential_gradient(batch,gradient,*workspace);
        CHECK(gradient==std::vector<double>(8,0.0));
    }
    CoulombTerm singular(a,b,2.0);
    auto workspace=singular.make_potential_gradient_workspace(batch);
    std::vector<double> gradient(8,0.0);
    CHECK_THROWS_AS(singular.evaluate_potential_gradient(batch,gradient,*workspace),std::domain_error);
    CHECK(std::vector<double>(batch.positions().begin(),batch.positions().end())==std::vector<double>(8,0.0));
}

TEST_CASE("Moire gradients exclude constant offsets and agree across periodic images", "[potential-gradient]") {
    SimulationCell cell(2,std::array{4.0,0.0,0.0,5.0});
    WalkerSet batch(3,3,2,cell);
    initialize(batch);
    const double pi=std::acos(-1.0);
    const std::vector<std::vector<double>> vectors{{pi/2,0},{0,2*pi/5},{pi/2,2*pi/5}};
    Hamiltonian ham;
    ham.add_component(std::make_unique<MoireTerm>(2,8.0,0.3,0.6,vectors,
        std::vector<std::pair<std::size_t,std::size_t>>{{0,1},{2,3}}));
    check_finite_differences(ham,batch);
    auto workspace=ham.make_potential_gradient_workspace(batch);
    std::vector<double> before(18),after(18); // [np*dim*nw]
    ham.evaluate_potential_gradient(batch,before,workspace);
    for (std::size_t w=0; w<3; ++w) batch.positions()[w]+=4.0;
    ham.evaluate_potential_gradient(batch,after,workspace);
    for (std::size_t i=0; i<18; ++i) CHECK(after[i]==Approx(before[i]).margin(1e-14));
    MoireTerm constant(2,8.0,0.0,0.6,vectors,{{0,3}});
    auto scratch=constant.make_potential_gradient_workspace(batch);
    std::fill(after.begin(),after.end(),0.0);
    constant.evaluate_potential_gradient(batch,after,*scratch);
    CHECK(after==std::vector<double>(18,0.0));
}

TEST_CASE("Coulomb gradients retain representable inverse-square far fields", "[potential-gradient]") {
    WalkerSet batch(1,2,1);
    batch.positions()[1]=1e150;
    CoulombTerm pair(group(0,1,1,1),group(1,2,1,1),1.0);
    auto workspace=pair.make_potential_gradient_workspace(batch);
    std::vector<double> gradient(2,0.0);
    pair.evaluate_potential_gradient(batch,gradient,*workspace);
    CHECK(gradient[0]>0.0);
    CHECK(gradient[0]/1e-300==Approx(1.0).epsilon(1e-14));
    CHECK(gradient[1]==-gradient[0]);
}

TEST_CASE("Gradient capabilities reject unsupported potentials and skip wavefunction kinetic terms", "[potential-gradient]") {
    WalkerSet batch(3,3,2);
    initialize(batch);
    Hamiltonian ham;
    CHECK(ham.supports_potential_gradient());
    ham.add_component(std::make_unique<Kinetic>(std::vector{group(0,3,-1,2)}));
    CHECK(ham.supports_potential_gradient());
    auto workspace=ham.make_potential_gradient_workspace(batch);
    std::vector<double> gradient(18,9.0);
    ham.evaluate_potential_gradient(batch,gradient,workspace);
    CHECK(gradient==std::vector<double>(18,0.0));
    ham.add_component(std::make_unique<UnsupportedGradient>());
    CHECK_FALSE(ham.supports_potential_gradient());
    CHECK_THROWS_WITH(ham.make_potential_gradient_workspace(batch),
        Catch::Matchers::ContainsSubstring("potential gradient is not supported"));
    CHECK_THROWS_AS(ham.evaluate_potential_gradient(batch,gradient,workspace),std::invalid_argument);
    const auto& unsupported=*ham.components().back();
    HarmonicTrap trap({group(0,3,0,2)},1.0);
    auto scratch=trap.make_potential_gradient_workspace(batch);
    CHECK_THROWS_AS(unsupported.evaluate_potential_gradient(batch,gradient,*scratch),std::logic_error);
}

TEST_CASE("Gradient evaluation preserves coordinates tables proposals and wavefunction arrays", "[potential-gradient]") {
    WalkerSet batch(5,3,2);
    initialize(batch);
    std::fill(batch.gradients().begin(),batch.gradients().end(),4.0);
    std::fill(batch.laplacians().begin(),batch.laplacians().end(),-3.0);
    std::fill(batch.local_energies().begin(),batch.local_energies().end(),12.0);
    std::vector<double> candidate(10,0.4);
    batch.propose_move_batch(0,candidate);
    const std::vector<double> positions(batch.positions().begin(),batch.positions().end());
    const std::vector<double> distances(batch.dist_tables().data().begin(),batch.dist_tables().data().end());
    const std::vector<double> pending(batch.pending_dist_rows().begin(),batch.pending_dist_rows().end());
    const auto ham=composite(2);
    auto workspace=ham.make_potential_gradient_workspace(batch);
    std::vector<double> gradient(30);
    ham.evaluate_potential_gradient(batch,gradient,workspace);
    CHECK(std::vector<double>(batch.positions().begin(),batch.positions().end())==positions);
    CHECK(std::vector<double>(batch.dist_tables().data().begin(),batch.dist_tables().data().end())==distances);
    CHECK(std::vector<double>(batch.pending_dist_rows().begin(),batch.pending_dist_rows().end())==pending);
    CHECK(std::vector<double>(batch.gradients().begin(),batch.gradients().end())==std::vector<double>(30,4.0));
    CHECK(std::vector<double>(batch.laplacians().begin(),batch.laplacians().end())==std::vector<double>(30,-3.0));
    CHECK(std::vector<double>(batch.local_energies().begin(),batch.local_energies().end())==std::vector<double>(5,12.0));
}

TEST_CASE("Gradient crowds clones and paths reproduce the full batch with untouched columns", "[potential-gradient]") {
    WalkerSet batch(5,3,2);
    initialize(batch);
    const auto ham=composite(2);
    auto full_workspace=ham.make_potential_gradient_workspace(batch);
    std::vector<double> full(30),split(30,9.0),cloned(30),path_output(30);
    ham.evaluate_potential_gradient(batch,full,full_workspace);
    ham.evaluate_potential_gradient(batch,split,full_workspace,1,4);
    for (std::size_t i=0; i<30; ++i) CHECK(split[i]==(i%5==0 || i%5==4 ? 9.0 : full[i]));
    ham.evaluate_potential_gradient(batch,split,full_workspace,0,1);
    ham.evaluate_potential_gradient(batch,split,full_workspace,4,5);
    CHECK(split==full);
    const auto clone=ham.clone();
    auto clone_workspace=clone.make_potential_gradient_workspace(batch);
    clone.evaluate_potential_gradient(batch,cloned,clone_workspace);
    CHECK(cloned==full);
    PathSet path(5,3,2,0.1,SimulationCell(2));
    std::copy(batch.positions().begin(),batch.positions().end(),path.positions().begin());
    path.rebuild_tables();
    auto path_workspace=ham.make_potential_gradient_workspace(path);
    ham.evaluate_potential_gradient(path,path_output,path_workspace);
    CHECK(path_output==full);
    auto first=ham.make_potential_gradient_workspace(batch),second=ham.make_potential_gradient_workspace(batch);
    std::fill(split.begin(),split.end(),9.0);
    // A shared const Hamiltonian, separate prepared scratch, disjoint crowds.
    #pragma omp parallel for num_threads(2)
    for (int crowd=0; crowd<2; ++crowd)
        ham.evaluate_potential_gradient(batch,split,crowd==0 ? first : second,
                                        crowd==0 ? 0 : 2,crowd==0 ? 2 : 5);
    CHECK(split==full);
}

TEST_CASE("Gradient workspace misuse is rejected before output or coordinates change", "[potential-gradient]") {
    WalkerSet batch(5,3,2),other(3,3,2);
    initialize(batch); initialize(other);
    auto ham=composite(2);
    auto workspace=ham.make_potential_gradient_workspace(batch);
    std::vector<double> out(30,7.0),small(18,7.0);
    const std::vector<double> positions(batch.positions().begin(),batch.positions().end());
    CHECK_THROWS_AS(ham.evaluate_potential_gradient(batch,small,workspace),std::invalid_argument);
    CHECK_THROWS_AS(ham.evaluate_potential_gradient(other,small,workspace),std::invalid_argument);
    CHECK_THROWS_AS(ham.evaluate_potential_gradient(batch,out,workspace,4,2),std::invalid_argument);
    CHECK_THROWS_AS(ham.evaluate_potential_gradient(batch,batch.positions(),workspace),std::invalid_argument);
    CHECK(std::vector<double>(batch.positions().begin(),batch.positions().end())==positions);
    const auto clone=ham.clone();
    CHECK_THROWS_AS(clone.evaluate_potential_gradient(batch,out,workspace),std::invalid_argument);
    ham.add_component(std::make_unique<HarmonicTrap>(std::vector{group(0,3,0,2)},1.0));
    CHECK_THROWS_AS(ham.evaluate_potential_gradient(batch,out,workspace),std::invalid_argument);
    CHECK(out==std::vector<double>(30,7.0));
    CHECK(small==std::vector<double>(18,7.0));
}

TEST_CASE("Gradient preparation refuses incompatible component geometry and nonfinite parameters", "[potential-gradient]") {
    WalkerSet batch(2,3,2);
    initialize(batch);
    HarmonicTrap trap({group(0,3,0,3)},1.0);
    CHECK_THROWS_AS(trap.make_potential_gradient_workspace(batch),std::invalid_argument);
    CoulombTerm bad_pair(group(0,1,-1,2),group(2,4,1,2),1.0);
    CHECK_THROWS_AS(bad_pair.make_potential_gradient_workspace(batch),std::invalid_argument);
    MoireTerm bad_vector(2,0,1,0,{{1.0}},{{0,3}});
    CHECK_THROWS_AS(bad_vector.make_potential_gradient_workspace(batch),std::invalid_argument);
    MoireTerm bad_range(2,0,1,0,{{1.0,0}},{{0,4}});
    CHECK_THROWS_AS(bad_range.make_potential_gradient_workspace(batch),std::invalid_argument);
    MoireTerm bad_parameter(2,0,1,std::numeric_limits<double>::infinity(),{{1.0,0}},{{0,3}});
    CHECK_THROWS_AS(bad_parameter.make_potential_gradient_workspace(batch),std::invalid_argument);
    batch.positions()[0]=std::numeric_limits<double>::quiet_NaN();
    CoulombTerm pair(group(0,1,-1,2),group(1,3,1,2),1.0);
    CHECK_THROWS_AS(pair.make_potential_gradient_workspace(batch),std::invalid_argument);
}

TEST_CASE("Prepared analytic gradients make no C++ allocations during repeated measurement", "[potential-gradient]") {
    WalkerSet batch(5,3,2);
    initialize(batch);
    const auto ham=composite(2);
    auto workspace=ham.make_potential_gradient_workspace(batch);
    std::vector<double> gradient(30);
    ham.evaluate_potential_gradient(batch,gradient,workspace);
    allocations=0; track_allocations=true;
    try {
        for (int repeat=0; repeat<100; ++repeat)
            ham.evaluate_potential_gradient(batch,gradient,workspace);
    } catch (...) { track_allocations=false; throw; }
    track_allocations=false;
    CHECK(allocations==0);
}

TEST_CASE("Prepared screened gradients make no C++ allocations including their first evaluation", "[potential-gradient][screened]") {
    WalkerSet batch(5,3,2);
    initialize(batch);
    Hamiltonian ham;
    ham.add_component(std::make_unique<ScreenedCoulombTerm>(group(0,1,-1,2),group(1,3,1,2),
        std::vector<double>{1.0,4.0,2.0,1.5},std::vector<double>{-1.0,1.0,3.0},2,3,2.0,0.1));
    auto workspace=ham.make_potential_gradient_workspace(batch);
    std::vector<double> gradient(30); // [np*dim*nw]
    allocations=0; track_allocations=true;
    try {
        for (int repeat=0; repeat<100; ++repeat)
            ham.evaluate_potential_gradient(batch,gradient,workspace);
    } catch (...) { track_allocations=false; throw; }
    track_allocations=false;
    CHECK(allocations == 0);
}

TEST_CASE("Accumulated gradient overflow is rejected instead of returning a nonfinite measurement", "[potential-gradient]") {
    WalkerSet batch(1,1,1);
    batch.positions()[0]=4.0;
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(std::vector{group(0,1,0,1)},
                                                   std::numeric_limits<double>::max()));
    auto workspace=ham.make_potential_gradient_workspace(batch);
    std::vector<double> gradient(1);
    CHECK_THROWS_AS(ham.evaluate_potential_gradient(batch,gradient,workspace),std::domain_error);
}

TEST_CASE("Prepared Ewald gradients allocate no C++ scratch even on their first evaluation", "[potential-gradient][ewald]") {
    const SimulationCell cell(2,std::array{6.0,0.0,0.0,6.0});
    WalkerSet batch(5,3,2,cell); initialize(batch);
    Hamiltonian ham;
    ham.add_component(std::make_unique<EwaldTerm>(std::vector{group(0,3,-1,2)},1.5,cell,1e-8));
    auto workspace=ham.make_potential_gradient_workspace(batch);
    std::vector<double> output(30);
    allocations=0; track_allocations=true;
    try {
        for (int repeat=0; repeat<100; ++repeat)
            ham.evaluate_potential_gradient(batch,output,workspace);
    } catch (...) { track_allocations=false; throw; }
    track_allocations=false;
    CHECK(allocations==0);
}
