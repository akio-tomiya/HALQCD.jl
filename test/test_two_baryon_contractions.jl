using LinearAlgebra
using Random

function random_propagator(rng, ncolor, shape)
    return randn(rng, ComplexF64, ncolor, 4, ncolor, 4, shape...)
end

function sparse_spin_tensor(rng, nquark, nonzero)
    tensor = zeros(ComplexF64, ntuple(_ -> 4, nquark))
    for _ in 1:nonzero
        tensor[rand(rng, CartesianIndices(tensor))] = randn(rng, ComplexF64)
    end
    return tensor
end

@testset "two-baryon block contraction" begin
    @testset "SU(3) against the direct sum over sites" begin
        rng = MersenneTwister(20261106)
        shape = (3, 2, 1, 2)
        propagators = Dict(
            :u => random_propagator(rng, 3, shape),
            :d => random_propagator(rng, 3, shape),
        )
        flavors = ((:u, :d, :u), (:u, :d, :d))
        tensor() = sparse_spin_tensor(rng, 3, 1)
        sink_terms = [(0.7 + 0.2im, tensor(), tensor()), (-0.4, tensor(), tensor())]
        source_terms = [(1.0 + 0.0im, tensor(), tensor())]
        for order in ((2, 1), (1, 2))
            fast = two_baryon_four_point(
                propagators, flavors, flavors, sink_terms, source_terms;
                source_baryon_order=order,
            )
            slow = two_baryon_four_point_explicit(
                propagators, flavors, flavors, sink_terms, source_terms;
                source_baryon_order=order,
            )
            @test size(fast) == (3, 2, 1, 2)
            @test maximum(abs, slow) > 1e-3
            @test maximum(abs, fast - slow) <= 1e-11 * maximum(abs, slow)
        end
    end

    @testset "one-baryon correlator" begin
        rng = MersenneTwister(7)
        shape = (2, 2, 1, 3)
        propagators = Dict(:u => random_propagator(rng, 3, shape), :d => random_propagator(rng, 3, shape))
        sink = sparse_spin_tensor(rng, 3, 3)
        source = sparse_spin_tensor(rng, 3, 3)
        fast = baryon_two_point(propagators, (:u, :d, :u), sink, source)
        for t in 1:3
            slow = sum(
                local_three_quark_correlator(
                    Dict(f => propagators[f][:, :, :, :, x, y, 1, t] for f in (:u, :d)),
                    (:u, :d, :u), sink, source,
                )
                for x in 1:2, y in 1:2
            )
            @test fast[t] ≈ slow rtol = 1e-12
        end
    end

    @testset "gauge invariance" begin
        rng = MersenneTwister(11)
        shape = (2, 2, 2, 1)
        S = random_propagator(rng, 3, shape)
        rotated = similar(S)
        G0 = random_group_element(SU{3}(), rng)
        for index in CartesianIndices(shape)
            G = random_group_element(SU{3}(), rng)
            for s in 1:4, s0 in 1:4
                rotated[:, s, :, s0, Tuple(index)...] = G * S[:, s, :, s0, Tuple(index)...] * G0'
            end
        end
        flavors = ((:u, :d, :u), (:u, :d, :d))
        terms = [(1.0 + 0im, sparse_spin_tensor(rng, 3, 4), sparse_spin_tensor(rng, 3, 4))]
        sources = [(1.0 + 0im, sparse_spin_tensor(rng, 3, 4), sparse_spin_tensor(rng, 3, 4))]
        before = two_baryon_four_point(Dict(:u => S, :d => S), flavors, flavors, terms, sources)
        after = two_baryon_four_point(Dict(:u => rotated, :d => rotated), flavors, flavors, terms, sources)
        @test maximum(abs, before - after) <= 1e-11 * maximum(abs, before)
    end

    @testset "generic groups" begin
        rng = MersenneTwister(4)
        # SU(4): four quarks per baryon.
        shape = (2, 1, 1, 1)
        propagators = Dict(:u => random_propagator(rng, 4, shape), :d => random_propagator(rng, 4, shape))
        flavors = ((:u, :u, :d, :d), (:u, :d, :d, :u))
        # A subset of ε_abcd keeps the direct sum short; the identity tested
        # here does not need an invariant tensor.
        invariant = ColorInvariant{4,Int}("epsilon4_subset", 4,
            [(colors, value) for (colors, value) in baryon_invariant(SU{4}()).entries if colors[1] == 1])
        # One fixed spin component per baryon. Many single components give an
        # exactly vanishing correlator, for example both u quarks of a baryon in
        # one spin (Julia 1.10 drew such a component at random) or the same
        # component for both baryons; this choice does not.
        spin_tensor(index) = (tensor = zeros(ComplexF64, 4, 4, 4, 4); tensor[index...] = randn(rng, ComplexF64); tensor)
        terms = [(1.0 + 0im, spin_tensor((1, 2, 3, 4)), spin_tensor((2, 1, 4, 3)))]
        sources = [(1.0 + 0im, spin_tensor((1, 2, 3, 4)), spin_tensor((2, 1, 4, 3)))]
        fast = two_baryon_four_point(propagators, flavors, flavors, terms, sources; color_invariant=invariant)
        slow = two_baryon_four_point_explicit(propagators, flavors, flavors, terms, sources; color_invariant=invariant)
        @test maximum(abs, slow) > 1e-3
        @test maximum(abs, fast - slow) <= 1e-10 * maximum(abs, slow)

        # Seven colors with a rank-three tensor on one triple of them.
        seven = ColorInvariant{3,Int}("one_triple", 7,
            [((p[1], p[2], p[3]), s) for (p, s) in (((1, 2, 7), 1), ((2, 7, 1), 1), ((7, 1, 2), 1),
                ((2, 1, 7), -1), ((1, 7, 2), -1), ((7, 2, 1), -1))])
        # Two flavors: three identical quarks in one spin component would
        # vanish by the Pauli principle.
        propagators = Dict(:u => random_propagator(rng, 7, shape), :d => random_propagator(rng, 7, shape))
        flavors = ((:u, :d, :u), (:u, :d, :d))
        terms = [(1.0 + 0im, sparse_spin_tensor(rng, 3, 2), sparse_spin_tensor(rng, 3, 2))]
        sources = [(1.0 + 0im, sparse_spin_tensor(rng, 3, 2), sparse_spin_tensor(rng, 3, 2))]
        fast = two_baryon_four_point(propagators, flavors, flavors, terms, sources; color_invariant=seven)
        slow = two_baryon_four_point_explicit(propagators, flavors, flavors, terms, sources; color_invariant=seven)
        @test maximum(abs, slow) > 1e-3
        @test maximum(abs, fast - slow) <= 1e-10 * maximum(abs, slow)
    end
end

@testset "nucleon operators" begin
    for gammas in (HALQCD._EUCLIDEAN_GAMMAS,)
        C = charge_conjugation_matrix(gammas)
        for γ in gammas
            @test norm(C * γ / C + transpose(γ)) <= 1e-12
        end
        @test norm(C + transpose(C)) <= 1e-12
        up, down = nucleon_spin_basis(gammas)
        basis = hcat(up, down)
        Σ = (-im * gammas[2] * gammas[3], -im * gammas[3] * gammas[1], -im * gammas[1] * gammas[2])
        pauli = (ComplexF64[0 1; 1 0], ComplexF64[0 -im; im 0], ComplexF64[1 0; 0 -1])
        for k in 1:3
            @test norm(basis' * Σ[k] * basis - pauli[k]) <= 1e-12
        end
        @test norm(gammas[4] * basis - basis) <= 1e-12
    end
    @test two_nucleon_spin_state(0, 0) ≈ [0 1; -1 0] / sqrt(2)
    @test_throws ArgumentError two_nucleon_spin_state(1, 2)

    # Free quarks: the projected proton propagates forward with a positive
    # correlator.
    S = free_wilson_point_propagator((2, 2, 2, 12); kappa=0.1, ncolor=3)
    quarks = Dict(:u => S, :d => S)
    C = nucleon_two_point(quarks; gammas=HALQCD._EUCLIDEAN_GAMMAS)
    @test all(real(C[t]) > 0 for t in 2:5)
    @test all(abs(imag(C[t])) <= 1e-10 * abs(C[t]) for t in 2:5)
    @test nucleon_two_point(quarks; gammas=HALQCD._EUCLIDEAN_GAMMAS, spin=2) ≈ C

    χ = two_nucleon_spin_state(0, 0)
    dataset = two_nucleon_nbs_dataset([S], [(1, 1, 1, 1)];
        gammas=HALQCD._EUCLIDEAN_GAMMAS, source_state=χ, sample_id="free")
    @test size(dataset.four_point) == (2, 2, 2, 12, 1, 1)
    @test dataset.one_hadron_1[:, 1, 1] ≈ C
    # The spin singlet does not depend on the spin quantization axis, so the
    # free four-point function is invariant under the cubic rotations.
    F = dataset.four_point[:, :, :, 3, 1, 1]
    @test F ≈ permutedims(F, (2, 3, 1)) rtol = 1e-10
end

@testset "tensor force and two-source N2LO" begin
    n = 6
    rng = MersenneTwister(3)
    radius2(x, y, z) = sum(abs2, (HALQCD._nearest_image(x, n)[1], HALQCD._nearest_image(y, n)[1],
        HALQCD._nearest_image(z, n)[1]))
    VC = [0.3 * exp(-radius2(x, y, z) / 4) for x in 1:n, y in 1:n, z in 1:n]
    VT = [-0.2 * exp(-radius2(x, y, z) / 6) for x in 1:n, y in 1:n, z in 1:n]

    # The spin singlet has no tensor force.
    singlet = zeros(ComplexF64, 2, 2, n, n, n)
    for x in 1:n, y in 1:n, z in 1:n
        singlet[:, :, x, y, z] .= randn(rng) .* two_nucleon_spin_state(0, 0)
    end
    @test maximum(abs, tensor_operator_s12(singlet)) <= 1e-12

    # Synthetic R with K R = (V_C + V_T S₁₂) R by construction.
    mass = 1.3
    R = randn(rng, ComplexF64, 2, 2, n, n, n)
    S12R = tensor_operator_s12(R)
    rplus = similar(R)
    for a in 1:2, b in 1:2
        target = VC .* R[a, b, :, :, :] .+ VT .* S12R[a, b, :, :, :]
        rest = target .- laplacian_periodic(R[a, b, :, :, :], (1.0, 1.0, 1.0)) ./ mass
        # With rminus = R: ∂ₜR = b/2 and ∂ₜ²R = b for rplus = R + b.
        rplus[a, b, :, :, :] .= R[a, b, :, :, :] .+ rest ./ (1 / (4 * mass) - 1 / 2)
    end
    result = lo_central_tensor_potential(R, R, rplus; mass)
    @test count(result.valid) > n^3 ÷ 2
    @test maximum(abs.(result.central[result.valid] .- VC[result.valid])) <= 1e-9
    @test maximum(abs.(result.tensor[result.valid] .- VT[result.valid])) <= 1e-9

    V0 = VC
    V2 = [0.05 * exp(-radius2(x, y, z) / 5) for x in 1:n, y in 1:n, z in 1:n]
    function synthetic(R0)
        target = V0 .* R0 .+ V2 .* laplacian_periodic(R0, (1.0, 1.0, 1.0))
        rest = target .- laplacian_periodic(R0, (1.0, 1.0, 1.0)) ./ mass
        return (R0, R0, R0 .+ rest ./ (1 / (4 * mass) - 1 / 2))
    end
    Ra = 1 .+ 0.3 .* rand(rng, n, n, n)
    Rb = 1 .+ 0.3 .* rand(rng, n, n, n)
    n2lo = n2lo_potential_two_sources(synthetic(Ra), synthetic(Rb); mass)
    @test count(n2lo.valid) > n^3 ÷ 2
    @test maximum(abs.(n2lo.V0[n2lo.valid] .- V0[n2lo.valid])) <= 1e-8
    @test maximum(abs.(n2lo.V2[n2lo.valid] .- V2[n2lo.valid])) <= 1e-8
end

@testset "Coulomb gauge residual" begin
    links = zeros(ComplexF64, 3, 3, 4, 2, 2, 2, 2)
    for μ in 1:4, index in CartesianIndices((2, 2, 2, 2))
        links[:, :, μ, Tuple(index)...] .= Matrix{ComplexF64}(I, 3, 3)
    end
    @test coulomb_gauge_residual(links).maximum == 0
    rng = MersenneTwister(5)
    links[:, :, 1, 1, 1, 1, 1] .= random_group_element(SU{3}(), rng)
    @test coulomb_gauge_residual(links).maximum > 0
end
