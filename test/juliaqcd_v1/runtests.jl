# Tests of the JuliaQCD extension on the v1 path (Gaugefields 1.1.x, LatticeMatrices.jl
# MPILattice fields, one process). Run from the repository root with
#
#   julia --project=test/juliaqcd_v1 -e 'using Pkg; Pkg.instantiate(); include("test/juliaqcd_v1/runtests.jl")'
#
# The propagators of LatticeDiracOperators.jl are compared with the reference
# implementation of HALQCD.jl (wilson_clover_point_propagator) on a small hot SU(3)
# field with 2HEX links, the setup of the Kurth et al. reproduction
# (scripts/kurth_pipi_production.jl). The differences were 2e-13 to 5e-13 when this
# test was written.
import JACC
JACC.@init_backend
using Gaugefields
using LatticeDiracOperators
using HALQCD
using LinearAlgebra
using Test

const SHAPE = (4, 4, 4, 8)
const KAPPA = 0.12
const CSW = 1.0
const GAMMAS = Tuple(LatticeDiracOperators.Dirac_operators.mk_gamma(1)[1][:, :, μ] for μ in 1:4)
const HEX = HEXSmearing(; alpha_outer=0.95, alpha_middle=0.76, alpha_inner=0.38, iterations=2)

relative_difference(a, b) = maximum(abs, a .- b) / maximum(abs, b)

# The links that cross the time boundary set to zero, as the extension does for
# BoundaryCondition[4] = 0.
function zeroed_time_boundary(V)
    W = copy_configuration(V)
    f = W[4].U
    nw = f.nw
    view(f.A, :, :, (nw+1):(nw+f.PN[1]), (nw+1):(nw+f.PN[2]), (nw+1):(nw+f.PN[3]), nw + f.PN[4]) .= 0
    Gaugefields.LatticeMatricesCompat.mark_lattice_dirty!(f)
    Gaugefields.set_wing_U!(W[4])
    return W
end

@testset "JuliaQCD v1 extension" begin
    U = gauge_configuration(SHAPE; colors=3, start=:hot, seed=1, process_grid=(1, 1, 1, 1))
    @test gauge_group(U) == SU{3}()
    links = juliaqcd_links(U)
    @test size(links) == (3, 3, 4, SHAPE...)
    @test maximum(norm(links[:, :, μ, x...]' * links[:, :, μ, x...] - I)
                  for μ in 1:4, x in Iterators.product(axes(links)[4:7]...)) < 1e-12
    V = smear(U, HEX)

    @testset "$fermiontype, antiperiodic time" for (fermiontype, csw) in (("Wilson", nothing), ("WilsonClover", CSW))
        source = (1, 1, 1, 2)
        S = juliaqcd_point_propagator(V; κ=KAPPA, fermiontype, cSW=csw, BoundaryCondition=[1, 1, 1, -1],
            eps_CG=1e-24, method_CG="bicgstab", source_position=source)
        reference = wilson_clover_point_propagator(gauge_group(V), juliaqcd_links(V); kappa=KAPPA,
            csw=isnothing(csw) ? 0.0 : csw, boundary_condition=(1, 1, 1, -1), source_position=source,
            gammas=GAMMAS, tolerance=1e-12)
        @test S.metadata["maximum_external_residual"] < 1e-10
        @test relative_difference(S.data, reference.data) < 1e-9
    end

    @testset "Dirichlet wall" begin
        source = (1, 1, 1, 4)
        # Wilson: the zeroed boundary links give exactly the phase-0 wall.
        S = juliaqcd_point_propagator(V; κ=KAPPA, BoundaryCondition=[1, 1, 1, 0], eps_CG=1e-24,
            method_CG="bicgstab", source_position=source)
        phase0 = wilson_clover_point_propagator(gauge_group(V), juliaqcd_links(V); kappa=KAPPA, csw=0.0,
            boundary_condition=(1, 1, 1, 0), source_position=source, gammas=GAMMAS, tolerance=1e-12)
        @test relative_difference(S.data, phase0.data) < 1e-9
        # Clover: compared on the same zeroed links (the clover term differs from a
        # phase-0 wall on the two time slices next to it).
        S = juliaqcd_point_propagator(V; κ=KAPPA, fermiontype="WilsonClover", cSW=CSW,
            BoundaryCondition=[1, 1, 1, 0], eps_CG=1e-24, method_CG="bicgstab", source_position=source)
        W = zeroed_time_boundary(V)
        same = wilson_clover_point_propagator(gauge_group(W), juliaqcd_links(W); kappa=KAPPA, csw=CSW,
            boundary_condition=(1, 1, 1, 1), source_position=source, gammas=GAMMAS, tolerance=1e-12)
        @test relative_difference(S.data, same.data) < 1e-9
    end

    @testset "wall source NBS dataset in Coulomb gauge" begin
        W = copy_configuration(U)
        g = similar(W[1])
        work = [similar(W[1]) for _ in 1:6]
        before = measure_plaquette(W)
        gaugefixing!(W, g, 1.0, 200, 1.99, 5_000, 1e-14, 0, work; D_fix=3, min_iterations=0)
        @test abs(measure_plaquette(W) - before) < 1e-10
        @test coulomb_gauge_residual(juliaqcd_links(W)).maximum < 1e-6
        V = smear(W, HEX)
        progress = IOBuffer()
        dataset = juliaqcd_pipi_i2_nbs(V; κ=KAPPA, sample_id="ci", source_positions=[(1, 1, 1, 5)],
            BoundaryCondition=[1, 1, 1, 0], fermiontype="WilsonClover", cSW=CSW, source_type=:wall,
            method_CG="bicgstab", eps_CG=1e-24, MaxCGstep=10_000, progress)
        @test count(==('\n'), String(take!(progress))) == 12
        @test size(dataset.four_point)[1:4] == SHAPE
        @test all(>(0), real.(dataset.one_hadron_1[:, 1, 1]))
        path = joinpath(mktempdir(), "nbs")
        write_nbs_dataset(path, dataset)
        @test read_nbs_dataset(path).four_point == dataset.four_point
    end
end
