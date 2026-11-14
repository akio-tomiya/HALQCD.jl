# Tests of the JuliaQCD extension on MPILattice fields split over MPI processes.
# Every process builds the same SU(3) links on several process grids of
# Gaugefields.gauge_configuration (time only, :auto, and spatial splits) and once
# whole on its own process (MPI.COMM_SELF). The propagators must equal the
# matching time slices of the whole ones, and the NBS datasets must be equal.
# The start follows the MPI example of the Gaugefields.jl documentation. Run from
# the repository root with, for example, two processes:
#
#   julia --project=test/juliaqcd_v1 -e 'using MPI; run(`$(mpiexec()) -n 2 $(Base.julia_cmd()) --project=test/juliaqcd_v1 test/juliaqcd_v1/mpi_runtests.jl`)'
using MPI
import JACC
JACC.@init_backend
using Gaugefields
using LatticeDiracOperators
using HALQCD
using LinearAlgebra
using Random
using Test

MPI.Init()

const COMM = MPI.COMM_WORLD
const NPROCS = MPI.Comm_size(COMM)
const SHAPE = (4, 4, 4, 8)
const KAPPA = 0.12
const CSW = 1.0
const SOLVER = (eps_CG=1e-24, method_CG="bicgstab")
const GRIDS = Any[(1, 1, 1, NPROCS), :auto]
NPROCS == 2 && push!(GRIDS, (2, 1, 1, 1))
NPROCS == 4 && append!(GRIDS, [(2, 1, 1, 2), (2, 2, 1, 1)])

relative_difference(a, b) = maximum(abs, a .- b) / maximum(abs, b)

# The same links on every process (exp of a random antihermitian matrix).
function random_links(shape; seed=11)
    rng = MersenneTwister(seed)
    links = zeros(ComplexF64, 3, 3, 4, shape...)
    for index in CartesianIndices(shape), μ in 1:4
        a = 0.5 * randn(rng, ComplexF64, 3, 3)
        h = (a - a') / 2
        h -= tr(h) / 3 * I
        links[:, :, μ, Tuple(index)...] = exp(h)
    end
    return links
end

# Gauge field with `links`; setindex! of an MPILattice field takes local indices.
function load_links(links, process_grid, comm)
    U = gauge_configuration(SHAPE; colors=3, start=:cold, process_grid, comm)
    lattice = U[1].U
    offset = lattice.coords .* lattice.PN
    for μ in 1:4, t in 1:lattice.PN[4], z in 1:lattice.PN[3], y in 1:lattice.PN[2],
            x in 1:lattice.PN[1], b in 1:3, a in 1:3
        U[μ][a, b, x, y, z, t] = links[a, b, μ, (offset .+ (x, y, z, t))...]
    end
    foreach(set_wing_U!, U)
    return U
end

function compare_datasets(a, b)
    return max(relative_difference(a.four_point, b.four_point),
        relative_difference(a.one_hadron_1, b.one_hadron_1))
end

const PROPAGATOR_CASES = (
    ("Wilson", nothing, :point, -1, (2, 1, 3, 3), nothing),
    ("WilsonClover", CSW, :wall, 0, (1, 1, 1, 6), nothing),
    ("Wilson", nothing, :point, -1, (3, 2, 1, 5), (alpha=0.5, steps=2)),
)

@testset "JuliaQCD v1 extension on $NPROCS MPI processes" begin
    links = random_links(SHAPE)
    whole = load_links(links, (1, 1, 1, 1), MPI.COMM_SELF)
    references = Dict(
        case => juliaqcd_point_propagator(whole; κ=KAPPA, fermiontype=case[1], cSW=case[2],
            BoundaryCondition=[1, 1, 1, case[4]], source_position=case[5], source_type=case[3],
            source_smearing=case[6], SOLVER...)
        for case in PROPAGATOR_CASES)
    pipi_kwargs = (; κ=KAPPA, sample_id="mpi", source_positions=[(1, 1, 1, 1), (2, 2, 2, 6)],
        BoundaryCondition=[1, 1, 1, -1], SOLVER...)
    nn_kwargs = (; κ=KAPPA, sample_id="mpi", source_state=[0 1; -1 0] / sqrt(2),
        source_positions=[(1, 1, 1, 3)], source_type=:wall, fermiontype="WilsonClover",
        cSW=CSW, BoundaryCondition=[1, 1, 1, 0], SOLVER...)
    pipi_reference = juliaqcd_pipi_i2_nbs(whole; pipi_kwargs...)
    nn_reference = juliaqcd_two_nucleon_nbs(whole; nn_kwargs...)

    @testset "process_grid=$grid" for grid in GRIDS
        split = load_links(links, grid, COMM)
        @test relative_difference(juliaqcd_links(split), links) < 1e-15

        @testset "propagator, $(case[1]), $(case[3]) source, smearing $(case[6])" for case in PROPAGATOR_CASES
            S = juliaqcd_point_propagator(split; κ=KAPPA, fermiontype=case[1], cSW=case[2],
                BoundaryCondition=[1, 1, 1, case[4]], source_position=case[5],
                source_type=case[3], source_smearing=case[6], SOLVER...)
            times = S.metadata["local_times"]
            @test S.metadata["mpi_processes"] == NPROCS
            @test S.metadata["maximum_external_residual"] < 1e-10
            @test size(S.data, 8) == length(times)
            # every time slice is held by exactly one process
            counts = zeros(Int, SHAPE[4])
            counts[times] .= 1
            @test MPI.Allreduce(counts, MPI.SUM, COMM) == ones(Int, SHAPE[4])
            reference = references[case].data[:, :, :, :, :, :, :, times]
            @test isempty(times) || relative_difference(S.data, reference) < 1e-9
        end

        @testset "ππ I=2 dataset" begin
            a = juliaqcd_pipi_i2_nbs(split; pipi_kwargs...)
            @test a.metadata.mpi_processes == NPROCS
            @test compare_datasets(a, pipi_reference) < 1e-9
        end

        @testset "two-nucleon dataset, clover wall source, Dirichlet" begin
            a = juliaqcd_two_nucleon_nbs(split; nn_kwargs...)
            @test compare_datasets(a, nn_reference) < 1e-9
        end
    end
end
