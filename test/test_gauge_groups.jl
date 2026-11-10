using LinearAlgebra
using Random

function random_local_propagators(rng, flavors, ncolor, nspin)
    return Dict(
        flavor => randn(rng, ComplexF64, ncolor, nspin, ncolor, nspin)
        for flavor in unique(flavors)
    )
end

function sparse_random_spin_tensor(rng, nquark, nspin, nnonzero)
    tensor = zeros(ComplexF64, ntuple(_ -> nspin, nquark))
    for index in rand(rng, CartesianIndices(tensor), nnonzero)
        tensor[index] = randn(rng, ComplexF64)
    end
    return tensor
end

# S'[a,α,b,β] = Σ g_sink[a,a'] S[a',α,b',β] conj(g_source[b,b'])
function gauge_transform_local_propagator(propagator, g_sink, g_source)
    ncolor, nspin = size(propagator, 1), size(propagator, 2)
    transformed = similar(propagator)
    for α in 1:nspin, β in 1:nspin
        transformed[:, α, :, β] = g_sink * propagator[:, α, :, β] * g_source'
    end
    return transformed
end

@testset "gauge group types" begin
    @test representation_dimension(SU{2}()) == 2
    @test representation_dimension(SU{3}()) == 3
    @test representation_dimension(SU{5}()) == 5
    @test representation_reality(SU{2}()) == :pseudoreal
    @test representation_reality(SU{3}()) == :complex
    @test baryon_quark_number(SU{4}()) == 4
    @test_throws ArgumentError SU{1}()
    @test sprint(show, SU{3}()) == "SU(3)"
end

@testset "color invariants" begin
    for n in 1:5
        epsilon = levi_civita_invariant(n)
        @test color_rank(epsilon) == n
        @test length(epsilon) == factorial(n)
    end
    epsilon3 = dense_color_tensor(levi_civita_invariant(3))
    @test epsilon3[1, 2, 3] == 1
    @test epsilon3[2, 1, 3] == -1
    @test epsilon3[3, 1, 2] == 1
    @test epsilon3[1, 1, 2] == 0

    # The SU(3) default keeps its historical entry order but is the same tensor.
    @test dense_color_tensor(baryon_invariant(SU{3}())) == epsilon3

    @test_throws ArgumentError ColorInvariant("dup", 2, [((1, 2), 1), ((1, 2), 2)])
    @test_throws BoundsError ColorInvariant("range", 2, [((1, 3), 1)])
    @test length(ColorInvariant("zeros", 2, [((1, 2), 0.0), ((2, 1), 1.0)])) == 1

    rng = MersenneTwister(20261104)
    for N in (2, 3, 4)
        g = random_group_element(SU{N}(), rng)
        @test norm(g' * g - I) < 1e-12
        @test abs(det(g) - 1) < 1e-12
        @test invariance_error(baryon_invariant(SU{N}()), g) < 1e-12
        @test invariance_error(meson_invariant(SU{N}()), g; conjugate_axes=(2,)) < 1e-12

        # Negative controls: a U(N) element with det ≠ 1 changes ε by
        # det(g), and δ_ab is not invariant without conjugating one axis.
        θ = 0.3
        u = cis(θ) * g
        @test invariance_error(baryon_invariant(SU{N}()), u) ≈
            abs(cis(N * θ) - 1) atol = 1e-12
        @test invariance_error(meson_invariant(SU{N}()), g) > 1e-3
    end
end

@testset "generic local color-singlet contraction" begin
    rng = MersenneTwister(4)

    # The default three-quark path is the generic path with ε_abc and keeps
    # bitwise identical results.
    nspin = 4
    flavors = (:u, :u, :s)
    propagators = random_local_propagators(rng, flavors, 3, nspin)
    sink = sparse_random_spin_tensor(rng, 3, nspin, 6)
    source = sparse_random_spin_tensor(rng, 3, nspin, 6)
    default_value = local_three_quark_correlator(propagators, flavors, sink, source)
    generic_value = local_color_singlet_correlator(
        propagators,
        flavors,
        sink,
        source;
        color_invariant=baryon_invariant(SU{3}()),
    )
    @test default_value == generic_value
    @test local_three_quark_correlator_explicit(propagators, flavors, sink, source) ≈
        default_value rtol = 1e-12

    # SU(N) baryons with N quarks: determinant and explicit permutation sums
    # agree, and the correlator is invariant under independent sink and source
    # color rotations.
    for (N, flavors) in ((2, (:u, :d)), (2, (:u, :u)), (4, (:u, :u, :d, :s)))
        nspin = 2
        group = SU{N}()
        invariant = baryon_invariant(group)
        propagators = random_local_propagators(rng, flavors, N, nspin)
        sink = sparse_random_spin_tensor(rng, N, nspin, 4)
        source = sparse_random_spin_tensor(rng, N, nspin, 4)
        value = local_color_singlet_correlator(
            propagators, flavors, sink, source; color_invariant=invariant,
        )
        explicit = local_color_singlet_correlator_explicit(
            propagators, flavors, sink, source; color_invariant=invariant,
        )
        @test abs(value) > 1e-8
        @test value ≈ explicit rtol = 1e-11

        g_sink = random_group_element(group, rng)
        g_source = random_group_element(group, rng)
        rotated = Dict(
            flavor => gauge_transform_local_propagator(p, g_sink, g_source)
            for (flavor, p) in propagators
        )
        rotated_value = local_color_singlet_correlator(
            rotated, flavors, sink, source; color_invariant=invariant,
        )
        @test rotated_value ≈ value rtol = 1e-11

        # Negative control: a U(N) sink rotation with det = e^{iNθ} multiplies
        # the baryon correlator by that phase.
        θ = 0.2
        phased = Dict(
            flavor => gauge_transform_local_propagator(p, cis(θ) * g_sink, g_source)
            for (flavor, p) in propagators
        )
        phased_value = local_color_singlet_correlator(
            phased, flavors, sink, source; color_invariant=invariant,
        )
        @test phased_value ≈ cis(N * θ) * value rtol = 1e-11
        @test !isapprox(phased_value, value; rtol=1e-6)
    end

    # A rank-three tensor on seven colors exercises a larger color axis. The
    # tensor is random, so only the determinant/explicit agreement is tested.
    color_entries = [
        ((a, b, c), randn(rng))
        for a in 1:7, b in 1:7, c in 1:7 if rand(rng) < 0.12
    ]
    seven_color = ColorInvariant("random7", 7, vec(color_entries))
    flavors = (:u, :d, :u)
    propagators = random_local_propagators(rng, flavors, 7, 2)
    sink = sparse_random_spin_tensor(rng, 3, 2, 3)
    source = sparse_random_spin_tensor(rng, 3, 2, 3)
    value = local_three_quark_correlator(
        propagators, flavors, sink, source; color_invariant=seven_color,
    )
    @test value ≈ local_three_quark_correlator_explicit(
        propagators, flavors, sink, source; color_invariant=seven_color,
    ) rtol = 1e-11

    # Seven-color propagators without an explicit invariant are rejected
    # instead of being contracted with the SU(3) tensor.
    @test_throws DimensionMismatch local_three_quark_correlator(
        propagators, flavors, sink, source,
    )
    @test_throws DimensionMismatch local_three_quark_correlator(
        propagators, flavors, sink, source;
        color_invariant=baryon_invariant(SU{3}()),
    )
    @test_throws TypeError local_three_quark_correlator(
        propagators, flavors, sink, source;
        color_invariant=baryon_invariant(SU{4}()),
    )
end

@testset "generic two-baryon contraction" begin
    rng = MersenneTwister(11)
    for N in (2, 4)
        nspin = 2
        invariant = baryon_invariant(SU{N}())
        flavors = N == 2 ? ((:u, :d), (:u, :d)) : ((:u, :d, :s, :c), (:u, :d, :s, :c))
        propagators = Dict(
            flavor => randn(rng, ComplexF64, 2, N, nspin, 2, N, nspin)
            for flavor in unique(vcat(collect.(flavors)...))
        )
        # Each extra nonzero spin entry multiplies the number of 2N x 2N
        # determinants by (N!)^2 on both sides, so SU(4) keeps one entry.
        nnonzero = N == 2 ? 2 : 1
        sinks = (
            sparse_random_spin_tensor(rng, N, nspin, nnonzero),
            sparse_random_spin_tensor(rng, N, nspin, nnonzero),
        )
        sources = (
            sparse_random_spin_tensor(rng, N, nspin, nnonzero),
            sparse_random_spin_tensor(rng, N, nspin, nnonzero),
        )
        factorized = local_two_baryon_correlator(
            propagators, flavors, flavors, sinks, sources;
            color_invariant=invariant,
        )
        full = local_two_baryon_correlator(
            propagators, flavors, flavors, sinks, sources;
            color_invariant=invariant, factorize_flavor=false,
        )
        @test abs(full) > 1e-8
        @test factorized ≈ full rtol = 1e-10

        # Swapping the order of the two source blocks of N antiquarks gives
        # (-1)^(N*N): +1 for the bosonic SU(2) and SU(4) baryons here.
        swapped = local_two_baryon_correlator(
            propagators, flavors, flavors, sinks, sources;
            color_invariant=invariant, source_baryon_order=(1, 2),
            factorize_flavor=false,
        )
        @test swapped ≈ (-1)^(N * N) * full rtol = 1e-10
    end

    @test_throws ArgumentError local_two_baryon_correlator(
        Dict(:u => zeros(ComplexF64, 2, 2, 2, 2, 2, 2)),
        ((:u, :u, :u), (:u, :u, :u)),
        ((:u, :u, :u), (:u, :u, :u)),
        (zeros(2, 2, 2), zeros(2, 2, 2)),
        (zeros(2, 2, 2), zeros(2, 2, 2));
        color_invariant=baryon_invariant(SU{2}()),
    )
end

@testset "group-aware U(1) phase" begin
    links = uniform_u1_links((4, 4, 2, 2), 1; charge_ratio=-1)
    su3 = zeros(ComplexF64, 3, 3, size(links)...)
    for index in CartesianIndices(size(links))
        su3[:, :, index] = Matrix{ComplexF64}(I, 3, 3)
    end
    @test apply_u1_phase(SU{3}(), su3, links) == apply_u1_links(su3, links)
    @test_throws DimensionMismatch apply_u1_phase(SU{2}(), su3, links)
end
