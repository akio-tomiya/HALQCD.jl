using LinearAlgebra
using Random

function random_links(group, rng, shape; scale=1.0)
    n = representation_dimension(group)
    links = zeros(ComplexF64, n, n, 4, shape...)
    for μ in 1:4, index in CartesianIndices(shape)
        links[:, :, μ, Tuple(index)...] = random_group_element(group, rng)
    end
    return links
end

function cold_links(n, shape)
    links = zeros(ComplexF64, n, n, 4, shape...)
    for μ in 1:4, index in CartesianIndices(shape), a in 1:n
        links[a, a, μ, Tuple(index)...] = 1
    end
    return links
end

function gauge_transform_links(links, rotations)
    shape = size(links)[4:7]
    transformed = similar(links)
    for μ in 1:4, index in CartesianIndices(shape)
        x = Tuple(index)
        xpμ = ntuple(ν -> ν == μ ? mod1(x[ν] + 1, shape[ν]) : x[ν], 4)
        transformed[:, :, μ, x...] = rotations[x...] * links[:, :, μ, x...] * rotations[xpμ...]'
    end
    return transformed
end

@testset "Lie algebra projections" begin
    rng = MersenneTwister(11)
    Z = randn(rng, ComplexF64, 3, 3)
    A = lie_algebra_projection(SU{3}(), Z)
    @test norm(A + A') < 1e-12
    @test abs(tr(A)) < 1e-12
    @test norm(lie_algebra_projection(SU{3}(), A) - A) < 1e-12
    @test group_defect(SU{3}(), exp(A)) < 1e-12
    @test_throws DimensionMismatch lie_algebra_projection(SU{2}(), Z)
end

@testset "Stout smearing" begin
    shape = (2, 2, 2, 3)
    @test average_plaquette(cold_links(3, shape)) ≈ 1
    @test stout_smear(SU{3}(), cold_links(3, shape); rho=0.1, levels=2) ≈ cold_links(3, shape)
    for group in (SU{2}(), SU{3}())
        rng = MersenneTwister(5)
        links = random_links(group, rng, shape; scale=0.4)
        rotations = [random_group_element(group, rng) for x in 1:2, y in 1:2, z in 1:2, t in 1:3]
        smeared = stout_smear(group, links; rho=0.11, levels=2)
        @test maximum(group_defect(group, smeared[:, :, μ, Tuple(i)...])
            for μ in 1:4, i in CartesianIndices(shape)) < 1e-10
        # Smearing commutes with gauge transformations.
        @test stout_smear(group, gauge_transform_links(links, rotations); rho=0.11, levels=2) ≈
            gauge_transform_links(smeared, rotations)
        # Smearing makes the field smoother.
        @test average_plaquette(smeared) > average_plaquette(links)
    end
    @test stout_smear(SU{4}(), cold_links(4, shape); rho=0.0, levels=3) == cold_links(4, shape)
    @test_throws DimensionMismatch stout_smear(SU{2}(), cold_links(3, shape); rho=0.1)
end

@testset "Clover field strength" begin
    # F = (Q - Q†)/8 with its trace kept, as in Bridge++ and LatticeMatrices.jl, for every N.
    shape = (2, 2, 2, 2)
    for N in (2, 3, 4)
        links = random_links(SU{N}(), MersenneTwister(40 + N), shape)
        L(μ, x) = links[:, :, μ, x...]
        sh(x, μ, s) = ntuple(ν -> ν == μ ? mod1(x[ν] + s, shape[ν]) : x[ν], 4)
        x, μ, ν = (1, 2, 1, 2), 2, 4
        Q = L(μ, x) * L(ν, sh(x, μ, 1)) * L(μ, sh(x, ν, 1))' * L(ν, x)' +
            L(ν, x) * L(μ, sh(sh(x, μ, -1), ν, 1))' * L(ν, sh(x, μ, -1))' * L(μ, sh(x, μ, -1)) +
            L(μ, sh(x, μ, -1))' * L(ν, sh(sh(x, μ, -1), ν, -1))' * L(μ, sh(sh(x, μ, -1), ν, -1)) * L(ν, sh(x, ν, -1)) +
            L(ν, sh(x, ν, -1))' * L(μ, sh(x, ν, -1)) * L(ν, sh(sh(x, μ, 1), ν, -1)) * L(μ, x)'
        F = clover_field_strength(SU{N}(), links, x, μ, ν)
        @test norm(F - (Q - Q') / 8) < 1e-13
        @test norm(F + F') < 1e-13
        # The trace vanishes identically for SU(2) only.
        N == 2 ? (@test abs(tr(F)) < 1e-13) : (@test abs(tr(F)) > 1e-6)
    end
    @test_throws DimensionMismatch clover_projection(SU{2}(), zeros(3, 3))
end

@testset "Wilson-clover propagator" begin
    shape = (2, 2, 2, 4)
    kappa = 0.11
    # Cold links: F = 0, so the clover term drops out and the free propagator remains.
    cold = wilson_clover_point_propagator(SU{3}(), cold_links(3, shape); kappa, csw=1.0,
        source_position=(2, 1, 2, 3))
    free = free_wilson_point_propagator(shape; kappa, ncolor=3, source_position=(2, 1, 2, 3))
    @test maximum(abs.(cold.data .- free)) / maximum(abs.(free)) < 1e-9

    for group in (SU{2}(), SU{3}())
        rng = MersenneTwister(17)
        links = random_links(group, rng, shape; scale=0.6)
        rotations = [random_group_element(group, rng) for x in 1:2, y in 1:2, z in 1:2, t in 1:4]
        source = (1, 2, 1, 2)
        solved = wilson_clover_point_propagator(group, links; kappa, csw=1.0, source_position=source)
        @test maximum(solved.true_residuals) < 1e-10
        # The clover term must change the propagator.
        wilson = wilson_clover_point_propagator(group, links; kappa, csw=0.0, source_position=source)
        @test maximum(abs.(solved.data .- wilson.data)) > 1e-4
        transformed = wilson_clover_point_propagator(group, gauge_transform_links(links, rotations);
            kappa, csw=1.0, source_position=source)
        expected = similar(solved.data)
        for index in CartesianIndices(shape), α in 1:4, β in 1:4
            expected[:, α, :, β, Tuple(index)...] =
                rotations[Tuple(index)...] * solved.data[:, α, :, β, Tuple(index)...] * rotations[source...]'
        end
        @test maximum(abs.(transformed.data .- expected)) / maximum(abs.(expected)) < 1e-9
        @test pion_correlator_point_source(transformed.data) ≈ pion_correlator_point_source(solved.data)
    end
    # Plain Wilson from a point source: the even-odd structure must not break BiCGStab.
    hot = random_links(SU{2}(), MersenneTwister(3), (4, 4, 2, 4))
    plain = wilson_clover_point_propagator(SU{2}(), hot; kappa, csw=0.0, solver=:bicgstab)
    @test maximum(plain.true_residuals) < 1e-10
    normal = wilson_clover_point_propagator(SU{2}(), hot; kappa, csw=1.0, solver=:cgnr)
    stabilized = wilson_clover_point_propagator(SU{2}(), hot; kappa, csw=1.0, solver=:bicgstab)
    @test maximum(abs.(normal.data .- stabilized.data)) / maximum(abs.(stabilized.data)) < 1e-9
    # Above the free critical value 1/8 the default solver still works.
    supercritical = wilson_clover_point_propagator(SU{2}(), cold_links(2, (4, 4, 4, 6)); kappa=0.13, csw=1.0)
    free_supercritical = free_wilson_point_propagator((4, 4, 4, 6); kappa=0.13, ncolor=2)
    @test maximum(abs.(supercritical.data .- free_supercritical)) / maximum(abs.(free_supercritical)) < 1e-9
    @test_throws ArgumentError wilson_clover_point_propagator(SU{2}(), hot; kappa, csw=1.0, solver=:gmres)
    @test_throws DimensionMismatch wilson_clover_point_propagator(SU{2}(), cold_links(3, shape);
        kappa, csw=1.0)
end

@testset "Dirichlet time boundary" begin
    shape = (4, 4, 2, 6)
    kappa = 0.11
    links = random_links(SU{2}(), MersenneTwister(5), shape)
    source = (2, 3, 1, 3)
    dirichlet = wilson_clover_point_propagator(SU{2}(), links; kappa, csw=0.0,
        boundary_condition=(1, 1, 1, 0), source_position=source)
    @test maximum(dirichlet.true_residuals) < 1e-10
    # The same operator: periodic, with the temporal links across the wall set to zero.
    cut = copy(links)
    cut[:, :, 4, :, :, :, shape[4]] .= 0
    periodic_cut = wilson_clover_point_propagator(SU{2}(), cut; kappa, csw=0.0,
        boundary_condition=(1, 1, 1, 1), source_position=source)
    @test maximum(abs.(dirichlet.data .- periodic_cut.data)) / maximum(abs.(dirichlet.data)) < 1e-9
    antiperiodic = wilson_clover_point_propagator(SU{2}(), links; kappa, csw=0.0, source_position=source)
    @test maximum(abs.(dirichlet.data .- antiperiodic.data)) > 1e-4
    # With clover the wall cuts the hops only: F_μν still uses the plaquettes across
    # the wall, so the result differs from the periodic operator on the cut links.
    clover = wilson_clover_point_propagator(SU{2}(), links; kappa, csw=1.0,
        boundary_condition=(1, 1, 1, 0), source_position=source)
    @test maximum(clover.true_residuals) < 1e-10
    clover_cut = wilson_clover_point_propagator(SU{2}(), cut; kappa, csw=1.0,
        boundary_condition=(1, 1, 1, 1), source_position=source)
    @test maximum(abs.(clover.data .- clover_cut.data)) / maximum(abs.(clover.data)) > 1e-6
    @test_throws ArgumentError wilson_clover_point_propagator(SU{2}(), links; kappa, csw=0.0,
        boundary_condition=(1, 1, 1, 2))
end

@testset "Wuppertal smearing" begin
    shape = (4, 4, 4, 2)
    alpha = 0.5
    point = zeros(ComplexF64, 2, 4, shape...)
    point[1, 2, 2, 2, 2, 1] = 1
    # Free field, one step: 1/(1+6α) at the point, α/(1+6α) at its six neighbors, sum kept.
    once = wuppertal_smear(cold_links(2, shape), point; alpha, steps=1)
    @test once[1, 2, 2, 2, 2, 1] ≈ 1 / (1 + 6alpha)
    @test once[1, 2, 3, 2, 2, 1] ≈ alpha / (1 + 6alpha)
    @test once[1, 2, 2, 2, 1, 1] ≈ alpha / (1 + 6alpha)
    @test sum(once) ≈ 1
    @test iszero(once[2, :, :, :, :, :]) && iszero(once[1, 1, :, :, :, :]) && iszero(once[:, :, :, :, :, 2])
    @test wuppertal_smear(cold_links(2, shape), point; alpha, steps=0) == point
    for group in (SU{2}(), SU{3}())
        rng = MersenneTwister(23)
        n = representation_dimension(group)
        links = random_links(group, rng, shape; scale=0.6)
        field = randn(rng, ComplexF64, n, 4, shape...)
        rotations = [random_group_element(group, rng) for x in 1:4, y in 1:4, z in 1:4, t in 1:2]
        smeared = wuppertal_smear(links, field; alpha, steps=5)
        rotated = similar(field)
        expected = similar(field)
        for index in CartesianIndices(shape), s in 1:4
            rotated[:, s, Tuple(index)...] = rotations[Tuple(index)...] * field[:, s, Tuple(index)...]
            expected[:, s, Tuple(index)...] = rotations[Tuple(index)...] * smeared[:, s, Tuple(index)...]
        end
        transformed = wuppertal_smear(gauge_transform_links(links, rotations), rotated; alpha, steps=5)
        @test maximum(abs.(transformed .- expected)) / maximum(abs.(expected)) < 1e-12
    end
    # A smeared-source propagator is covariant with the rotation at the source point.
    rng = MersenneTwister(29)
    pshape = (4, 4, 2, 4)
    links = random_links(SU{2}(), rng, pshape)
    rotations = [random_group_element(SU{2}(), rng) for x in 1:4, y in 1:4, z in 1:2, t in 1:4]
    source = (2, 1, 2, 2)
    smearing = (alpha=0.4, steps=3)
    solved = wilson_clover_point_propagator(SU{2}(), links; kappa=0.11, csw=0.0,
        source_position=source, source_smearing=smearing)
    @test maximum(solved.true_residuals) < 1e-10
    transformed = wilson_clover_point_propagator(SU{2}(), gauge_transform_links(links, rotations);
        kappa=0.11, csw=0.0, source_position=source, source_smearing=smearing)
    expected = similar(solved.data)
    for index in CartesianIndices(pshape), α in 1:4, β in 1:4
        expected[:, α, :, β, Tuple(index)...] =
            rotations[Tuple(index)...] * solved.data[:, α, :, β, Tuple(index)...] * rotations[source...]'
    end
    @test maximum(abs.(transformed.data .- expected)) / maximum(abs.(expected)) < 1e-9
    @test_throws DimensionMismatch wuppertal_smear(links, zeros(ComplexF64, 3, 4, pshape...); alpha=0.4, steps=1)
end
