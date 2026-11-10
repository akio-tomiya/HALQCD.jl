using LinearAlgebra
using Random

# Σ_x [tr A(x+r) tr A(x) - tr(A(x+r) A(x))] summed directly in position space.
function pipi_trace_formula_direct(propagator)
    ncolor, nspin = size(propagator, 1), size(propagator, 2)
    n = ncolor * nspin
    nx, ny, nz, nt = size(propagator)[5:8]
    result = zeros(ComplexF64, nx, ny, nz, nt)
    for t in 1:nt
        blocks = Dict(
            (x, y, z) => begin
                S = reshape(propagator[:, :, :, :, x, y, z, t], n, n)
                S' * S
            end
            for x in 1:nx, y in 1:ny, z in 1:nz
        )
        for (site_1, A1) in blocks, (site_2, A2) in blocks
            r = mod.(site_1 .- site_2, (nx, ny, nz)) .+ 1
            result[r..., t] += tr(A1) * tr(A2) - tr(A1 * A2)
        end
    end
    return result
end

function gauge_transform_point_propagator(propagator, sink_rotations, source_rotation)
    transformed = similar(propagator)
    nspin = size(propagator, 2)
    for index in CartesianIndices(size(propagator)[5:8])
        g = sink_rotations[index]
        for α in 1:nspin, β in 1:nspin
            transformed[:, α, :, β, index] =
                g * propagator[:, α, :, β, index] * source_rotation'
        end
    end
    return transformed
end

# Position-space Wilson operator in the hopping normalization of
# free_wilson_point_propagator, applied to one source column.
function apply_free_wilson(
    field, kappa, r, boundary_condition; gammas=HALQCD._EUCLIDEAN_GAMMAS,
)
    shape = size(field)[2:5]
    result = copy(field)
    for index in CartesianIndices(shape)
        site = Tuple(index)
        for μ in 1:4
            forward = collect(site)
            forward[μ] += 1
            forward_phase = 1
            if forward[μ] > shape[μ]
                forward[μ] = 1
                forward_phase = boundary_condition[μ]
            end
            backward = collect(site)
            backward[μ] -= 1
            backward_phase = 1
            if backward[μ] < 1
                backward[μ] = shape[μ]
                backward_phase = boundary_condition[μ]
            end
            result[:, site...] .-= kappa .* (
                forward_phase .* ((r * I - gammas[μ]) * field[:, forward...]) .+
                backward_phase .* ((r * I + gammas[μ]) * field[:, backward...])
            )
        end
    end
    return result
end

@testset "I=2 two-pseudoscalar four-point function" begin
    rng = MersenneTwister(20261104)

    @testset "Fourier, direct and explicit Wick paths agree" begin
        for (ncolor, shape) in ((3, (2, 3, 2, 2)), (2, (3, 2, 2, 2)))
            propagator = randn(rng, ComplexF64, ncolor, 4, ncolor, 4, shape...)
            fourier = pipi_i2_four_point(propagator)
            direct = pipi_trace_formula_direct(propagator)
            explicit = pipi_i2_four_point_explicit(propagator)
            @test size(fourier) == shape
            @test eltype(fourier) == Float64
            @test fourier ≈ real.(direct) rtol = 1e-12
            @test maximum(abs, imag.(direct)) <= 1e-10 * maximum(abs, direct)
            @test explicit ≈ direct rtol = 1e-12
        end
        # Seven color axes with the δ_ab meson invariant.
        propagator = randn(rng, ComplexF64, 7, 4, 7, 4, 2, 1, 1, 1)
        explicit = pipi_i2_four_point_explicit(
            propagator; color_invariant=kronecker_invariant(7),
        )
        @test pipi_i2_four_point(propagator) ≈ real.(explicit) rtol = 1e-12
    end

    @testset "representation of γ₅ does not matter" begin
        propagator = randn(rng, ComplexF64, 2, 4, 2, 4, 2, 2, 1, 2)
        dirac_gamma5 = ComplexF64[0 0 1 0; 0 0 0 1; 1 0 0 0; 0 1 0 0]
        @test pipi_i2_four_point_explicit(propagator; gamma5=dirac_gamma5) ≈
              pipi_i2_four_point_explicit(propagator) rtol = 1e-12
    end

    @testset "sum rule and one-pion blocks" begin
        propagator = randn(rng, ComplexF64, 3, 4, 3, 4, 2, 2, 3, 2)
        four_point = pipi_i2_four_point(propagator)
        pion = pion_correlator_point_source(propagator)
        for t in 1:2
            total_block = sum(
                begin
                    S = reshape(propagator[:, :, :, :, x, y, z, t], 12, 12)
                    S' * S
                end
                for x in 1:2, y in 1:2, z in 1:3
            )
            @test sum(four_point[:, :, :, t]) ≈ pion[t]^2 - sum(abs2, total_block) rtol = 1e-12
        end
    end

    @testset "gauge invariance" begin
        for ncolor in (2, 3, 4)
            group = SU{ncolor}()
            propagator = randn(rng, ComplexF64, ncolor, 4, ncolor, 4, 2, 2, 2, 2)
            sink_rotations = [
                random_group_element(group, rng)
                for x in 1:2, y in 1:2, z in 1:2, t in 1:2
            ]
            source_rotation = random_group_element(group, rng)
            transformed = gauge_transform_point_propagator(
                propagator, sink_rotations, source_rotation,
            )
            @test pipi_i2_four_point(transformed) ≈ pipi_i2_four_point(propagator) rtol = 1e-12
            @test pion_correlator_point_source(transformed) ≈
                  pion_correlator_point_source(propagator) rtol = 1e-12
        end
    end

    @testset "argument checks" begin
        @test_throws DimensionMismatch pipi_i2_four_point(
            zeros(ComplexF64, 3, 4, 2, 4, 2, 2, 2, 2),
        )
        @test_throws DimensionMismatch pipi_i2_four_point_explicit(
            zeros(ComplexF64, 3, 4, 3, 4, 1, 1, 1, 1);
            color_invariant=kronecker_invariant(2),
        )
        @test_throws DimensionMismatch pipi_i2_four_point_explicit(
            zeros(ComplexF64, 2, 4, 2, 4, 1, 1, 1, 1);
            gamma5=Matrix{ComplexF64}(I, 2, 2),
        )
    end
end

@testset "free Wilson point propagator" begin
    shape = (2, 2, 3, 4)
    kappa, r = 0.11, 1.0
    boundary_condition = (1, 1, 1, -1)
    source = (2, 1, 3, 3)
    propagator = free_wilson_point_propagator(
        shape; kappa, r, ncolor=2, boundary_condition, source_position=source,
    )
    @test size(propagator) == (2, 4, 2, 4, shape...)
    @test all(iszero, propagator[1, :, 2, :, :, :, :, :])
    for source_spin in 1:4
        column = propagator[1, :, 1, source_spin, :, :, :, :]
        applied = apply_free_wilson(column, kappa, r, boundary_condition)
        expected = zeros(ComplexF64, size(column))
        expected[source_spin, source...] = 1
        @test maximum(abs, applied .- expected) <= 1e-12
    end
    # Another basis γ'_μ = V γ_μ V† gives S' = V S V†.
    V = Matrix(qr(randn(MersenneTwister(5), ComplexF64, 4, 4)).Q)
    rotated_gammas = Tuple(V * gamma * V' for gamma in HALQCD._EUCLIDEAN_GAMMAS)
    rotated = free_wilson_point_propagator(
        shape; kappa, r, ncolor=2, boundary_condition, source_position=source,
        gammas=rotated_gammas,
    )
    column = rotated[2, :, 2, 3, :, :, :, :]
    expected = zeros(ComplexF64, size(column))
    expected[3, source...] = 1
    @test maximum(abs, apply_free_wilson(
        column, kappa, r, boundary_condition; gammas=rotated_gammas,
    ) .- expected) <= 1e-12
    @test maximum(
        maximum(abs, rotated[1, :, 1, :, index] .- V * propagator[1, :, 1, :, index] * V')
        for index in CartesianIndices(shape)
    ) <= 1e-12
    @test_throws DimensionMismatch free_wilson_point_propagator(
        shape; kappa, gammas=rotated_gammas[1:3],
    )
    @test_throws ArgumentError free_wilson_point_propagator(
        shape; kappa, boundary_condition=(1, 1, 1, 0),
    )
    @test_throws BoundsError free_wilson_point_propagator(
        shape; kappa, source_position=(3, 1, 1, 1),
    )
end

@testset "I=2 two-pseudoscalar NBS dataset" begin
    rng = MersenneTwister(11)
    shape = (2, 2, 2, 4)
    propagators = [randn(rng, ComplexF64, 3, 4, 3, 4, shape...) for _ in 1:2]
    positions = [(1, 1, 1, 1), (2, 1, 2, 3)]
    dataset = pipi_i2_nbs_dataset(
        propagators, positions;
        sample_id="random-test",
        code_commit="test",
        metadata=(gauge_group="SU(3)",),
    )
    @test size(dataset.four_point) == (shape..., 2, 1)
    @test size(dataset.one_hadron_1) == (4, 2, 1)
    @test dataset.metadata.channel == PIPI_I2_CHANNEL
    @test dataset.metadata.gauge_group == "SU(3)"
    @test dataset.metadata.ncolor == 3
    @test dataset.metadata.source_ids == ("point-1-1-1-1", "point-2-1-2-3")
    # The second source sits at t = 3, so τ = 1 is t = 3 and τ = 3 is t = 1.
    absolute = pipi_i2_four_point(propagators[2])
    @test dataset.four_point[:, :, :, 1, 2, 1] == absolute[:, :, :, 3]
    @test dataset.four_point[:, :, :, 3, 2, 1] == absolute[:, :, :, 1]
    @test dataset.one_hadron_1[2, 2, 1] == pion_correlator_point_source(propagators[2])[4]
    @test dataset.one_hadron_1 == dataset.one_hadron_2

    mktempdir() do directory
        path = write_nbs_dataset(joinpath(directory, "pipi"), dataset)
        loaded = read_nbs_dataset(path)
        @test loaded.four_point == dataset.four_point
        @test loaded.metadata.channel == PIPI_I2_CHANNEL
    end

    result = lo_potential_equal_mass(dataset, 2; mass=1.0)
    @test size(result.potential) == shape[1:3]
    @test all(isfinite, result.potential)

    @test_throws DimensionMismatch pipi_i2_nbs_dataset(
        propagators, positions[1:1]; sample_id="x", code_commit="test",
    )
    @test_throws BoundsError pipi_i2_nbs_dataset(
        propagators[1:1], [(1, 1, 1, 5)]; sample_id="x", code_commit="test",
    )
    @test_throws MethodError juliaqcd_pipi_i2_nbs(nothing; κ=0.1, sample_id="x")
    @test_throws MethodError HAL_NBS_measurement(nothing; sample_id="x")
end
