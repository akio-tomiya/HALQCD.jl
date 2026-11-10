using LinearAlgebra

function explicit_three_quark_permutation_sum(
    propagators,
    flavors,
    sink_tensor,
    source_tensor,
)
    nspin = size(sink_tensor, 1)
    levi_civita(i, j, k) =
        length(Set((i, j, k))) < 3 ? 0 : (isodd((i > j) + (i > k) + (j > k)) ? -1 : 1)
    total = 0.0 + 0.0im

    for a in 1:3, b in 1:3, c in 1:3
        sink_sign = levi_civita(a, b, c)
        iszero(sink_sign) && continue
        sink_colors = (a, b, c)
        for ap in 1:3, bp in 1:3, cp in 1:3
            source_sign = levi_civita(ap, bp, cp)
            iszero(source_sign) && continue
            source_colors = (ap, bp, cp)
            for α in 1:nspin, β in 1:nspin, γ in 1:nspin
                sink_coefficient = sink_tensor[α, β, γ]
                iszero(sink_coefficient) && continue
                sink_spins = (α, β, γ)
                for αp in 1:nspin, βp in 1:nspin, γp in 1:nspin
                    source_coefficient = source_tensor[αp, βp, γp]
                    iszero(source_coefficient) && continue
                    source_spins = (αp, βp, γp)

                    for p1 in 1:3, p2 in 1:3, p3 in 1:3
                        wick_sign = levi_civita(p1, p2, p3)
                        iszero(wick_sign) && continue
                        source_assignment = (p1, p2, p3)
                        product = 1.0 + 0.0im
                        for line in 1:3
                            source_line = source_assignment[line]
                            if flavors[line] != flavors[source_line]
                                product = 0.0 + 0.0im
                                break
                            end
                            product *= propagators[flavors[line]][
                                sink_colors[line],
                                sink_spins[line],
                                source_colors[source_line],
                                source_spins[source_line],
                            ]
                        end
                        total += (
                            sink_sign *
                            source_sign *
                            wick_sign *
                            sink_coefficient *
                            source_coefficient *
                            product
                        )
                    end
                end
            end
        end
    end
    return total
end

@testset "baryon spin tensor construction" begin
    diquark = ComplexF64[
        1 + 2im 0
        0 -3im
    ]
    tensor = baryon_spin_tensor(diquark, 2)
    @test size(tensor) == (2, 2, 2)
    @test tensor[:, :, 2] == diquark
    @test all(iszero, tensor[:, :, 1])
    @test count(!iszero, tensor) == count(!iszero, diquark)
    @test_throws BoundsError baryon_spin_tensor(diquark, 0)
    @test_throws DimensionMismatch baryon_spin_tensor(zeros(2, 3), 1)
end

@testset "XiXi flavor determinant factorization" begin
    flavors = (:s, :u, :s, :s, :u, :s)
    wick_matrix = zeros(ComplexF64, 6, 6)
    for sink_line in 1:6, source_line in 1:6
        if flavors[sink_line] == flavors[source_line]
            wick_matrix[sink_line, source_line] = (
                sin(0.31sink_line + 0.17source_line) +
                im * cos(0.13sink_line * source_line + 0.11sink_line)
            )
        end
    end

    strange_lines = findall(==(:s), flavors)
    light_lines = findall(==(:u), flavors)
    factorized = (
        det(wick_matrix[strange_lines, strange_lines]) *
        det(wick_matrix[light_lines, light_lines])
    )
    @test det(wick_matrix) ≈ factorized rtol = 2e-13 atol = 2e-13
    @test flavor_factorized_wick_determinant(
        wick_matrix,
        flavors,
        flavors,
    ) ≈ det(wick_matrix) rtol = 2e-13 atol = 2e-13

    explicit_sum = 0.0 + 0.0im
    flavor_preserving_terms = 0
    ranges = ntuple(_ -> 1:6, 6)
    for source_assignment in Iterators.product(ranges...)
        length(Set(source_assignment)) == 6 || continue
        all(
            flavors[sink_line] == flavors[source_assignment[sink_line]]
            for sink_line in 1:6
        ) || continue
        inversions = sum(
            source_assignment[i] > source_assignment[j]
            for i in 1:5 for j in (i + 1):6
        )
        wick_sign = isodd(inversions) ? -1 : 1
        explicit_sum += wick_sign * prod(
            wick_matrix[sink_line, source_assignment[sink_line]]
            for sink_line in 1:6
        )
        flavor_preserving_terms += 1
    end
    @test flavor_preserving_terms == factorial(4) * factorial(2) == 48
    @test det(wick_matrix) ≈ explicit_sum rtol = 2e-12 atol = 2e-12

    # Negative control: contracting the two baryons separately drops
    # same-flavor quark exchange between baryons and is not the six-quark
    # correlator.
    separate_baryon_product = (
        det(wick_matrix[1:3, 1:3]) *
        det(wick_matrix[4:6, 4:6])
    )
    @test !isapprox(
        separate_baryon_product,
        det(wick_matrix);
        rtol = 1e-6,
        atol = 1e-8,
    )

    reordered_source_flavors = (:u, :s, :s, :u, :s, :s)
    reordered_matrix = Array{ComplexF64}(undef, 6, 6)
    flavor_zeroed = zeros(ComplexF64, 6, 6)
    for sink_line in 1:6, source_line in 1:6
        value = (
            sin(0.23sink_line + 0.19source_line) +
            im * cos(0.07sink_line * source_line + 0.17source_line)
        )
        reordered_matrix[sink_line, source_line] = value
        if flavors[sink_line] == reordered_source_flavors[source_line]
            flavor_zeroed[sink_line, source_line] = value
        end
    end
    @test flavor_factorized_wick_determinant(
        reordered_matrix,
        flavors,
        reordered_source_flavors,
    ) ≈ det(flavor_zeroed) rtol = 2e-12 atol = 2e-12

    # Exhaust all 15 placements of the two light-quark lines independently at
    # sink and source. This stresses both permutation signs, rather than only
    # one hand-picked ordering.
    flavor_arrangements = NTuple{6,Symbol}[]
    for first_light in 1:5, second_light in (first_light + 1):6
        arrangement = fill(:s, 6)
        arrangement[[first_light, second_light]] .= :u
        push!(flavor_arrangements, Tuple(arrangement))
    end
    permutation_stress_matrix = Array{ComplexF64}(undef, 6, 6)
    for sink_line in 1:6, source_line in 1:6
        permutation_stress_matrix[sink_line, source_line] = (
            sin(0.29sink_line + 0.37source_line) +
            im * cos(0.21sink_line * source_line - 0.08source_line)
        )
    end
    maximum_permutation_error = 0.0
    for sink_order in flavor_arrangements
        for source_order in flavor_arrangements
            stress_flavor_zeroed = zeros(ComplexF64, 6, 6)
            for sink_line in 1:6, source_line in 1:6
                if sink_order[sink_line] == source_order[source_line]
                    stress_flavor_zeroed[sink_line, source_line] =
                        permutation_stress_matrix[sink_line, source_line]
                end
            end
            factorized_stress = flavor_factorized_wick_determinant(
                permutation_stress_matrix,
                sink_order,
                source_order,
            )
            maximum_permutation_error = max(
                maximum_permutation_error,
                abs(factorized_stress - det(stress_flavor_zeroed)),
            )
        end
    end
    @test length(flavor_arrangements)^2 == 225
    @test maximum_permutation_error < 2e-12

    source_baryon_swap = (4, 5, 6, 1, 2, 3)
    @test flavor_factorized_wick_determinant(
        wick_matrix[:, collect(source_baryon_swap)],
        flavors,
        flavors,
    ) ≈ -det(wick_matrix) rtol = 2e-12 atol = 2e-12
    @test iszero(
        flavor_factorized_wick_determinant(
            wick_matrix,
            flavors,
            (:u, :u, :s, :s, :u, :s),
        ),
    )
    @test_throws DimensionMismatch flavor_factorized_wick_determinant(
        zeros(ComplexF64, 5, 6),
        flavors,
        flavors,
    )
end

@testset "local two-baryon contractions" begin
    xi_flavors = ((:s, :u, :s), (:s, :u, :s))
    one_spin_tensor = ones(ComplexF64, 1, 1, 1)
    baryon_tensors = (one_spin_tensor, one_spin_tensor)

    diagonal_propagator = zeros(ComplexF64, 2, 3, 1, 2, 3, 1)
    for baryon in 1:2, color in 1:3
        diagonal_propagator[baryon, color, 1, baryon, color, 1] = 1
    end
    diagonal_propagators = Dict(
        :s => diagonal_propagator,
        :u => copy(diagonal_propagator),
    )
    single_identity_block = diagonal_propagator[1, :, :, 1, :, :]
    single_identity = local_three_quark_correlator(
        Dict(:s => single_identity_block, :u => copy(single_identity_block)),
        (:s, :u, :s),
        one_spin_tensor,
        one_spin_tensor,
    )
    diagonal_two_baryon = local_two_baryon_correlator(
        diagonal_propagators,
        xi_flavors,
        xi_flavors,
        baryon_tensors,
        baryon_tensors,
    )
    @test diagonal_two_baryon ≈ -single_identity^2
    @test local_two_baryon_correlator(
        diagonal_propagators,
        xi_flavors,
        xi_flavors,
        baryon_tensors,
        baryon_tensors;
        source_baryon_order=(1, 2),
    ) ≈ -diagonal_two_baryon

    dense_two_baryon_propagators =
        Dict{Symbol,Array{ComplexF64,6}}()
    for (flavor_index, flavor) in enumerate((:u, :s))
        propagator = Array{ComplexF64}(undef, 2, 3, 1, 2, 3, 1)
        for sink_baryon in 1:2, sink_color in 1:3,
                source_baryon in 1:2, source_color in 1:3
            propagator[
                sink_baryon,
                sink_color,
                1,
                source_baryon,
                source_color,
                1,
            ] = (
                sin(
                    0.17flavor_index +
                    0.23sink_baryon +
                    0.11sink_color +
                    0.19source_baryon +
                    0.07source_color,
                ) +
                im * cos(
                    0.13sink_baryon * source_color +
                    0.05source_baryon * sink_color +
                    0.09flavor_index,
                )
            )
        end
        dense_two_baryon_propagators[flavor] = propagator
    end
    factorized_two_baryon = local_two_baryon_correlator(
        dense_two_baryon_propagators,
        xi_flavors,
        xi_flavors,
        baryon_tensors,
        baryon_tensors,
    )
    full_determinant_two_baryon = local_two_baryon_correlator(
        dense_two_baryon_propagators,
        xi_flavors,
        xi_flavors,
        baryon_tensors,
        baryon_tensors;
        factorize_flavor=false,
    )
    @test factorized_two_baryon ≈ full_determinant_two_baryon rtol = 2e-11 atol = 2e-11
    @test local_two_baryon_correlator(
        dense_two_baryon_propagators,
        xi_flavors,
        xi_flavors,
        baryon_tensors,
        baryon_tensors;
        source_baryon_order=(1, 2),
    ) ≈ -factorized_two_baryon rtol = 2e-11 atol = 2e-11

    separate_baryon_product = -prod(
        local_three_quark_correlator(
            Dict(
                flavor => propagator[
                    baryon,
                    :,
                    :,
                    baryon,
                    :,
                    :,
                ]
                for (flavor, propagator) in dense_two_baryon_propagators
            ),
            (:s, :u, :s),
            one_spin_tensor,
            one_spin_tensor,
        )
        for baryon in 1:2
    )
    relative_interbaryon_exchange_effect = (
        abs(factorized_two_baryon - separate_baryon_product) /
        max(
            abs(factorized_two_baryon),
            abs(separate_baryon_product),
            eps(),
        )
    )
    @test relative_interbaryon_exchange_effect > 0.1

    sink_rotations = (
        ComplexF64[
            cos(0.31) sin(0.31) 0
            -sin(0.31) cos(0.31) 0
            0 0 1
        ],
        ComplexF64[
            1 0 0
            0 cos(-0.27) im*sin(-0.27)
            0 im*sin(-0.27) cos(-0.27)
        ],
    )
    source_rotations = (
        ComplexF64[
            cos(-0.19) 0 sin(-0.19)
            0 1 0
            -sin(-0.19) 0 cos(-0.19)
        ],
        ComplexF64[
            cos(0.23) im*sin(0.23) 0
            im*sin(0.23) cos(0.23) 0
            0 0 1
        ],
    )
    for rotation in (sink_rotations..., source_rotations...)
        @test rotation' * rotation ≈ I
        @test det(rotation) ≈ 1
    end
    rotated_two_baryon_propagators =
        Dict{Symbol,Array{ComplexF64,6}}()
    for (flavor, propagator) in dense_two_baryon_propagators
        rotated = similar(propagator)
        for sink_baryon in 1:2, source_baryon in 1:2
            rotated[
                sink_baryon,
                :,
                1,
                source_baryon,
                :,
                1,
            ] .= (
                sink_rotations[sink_baryon] *
                propagator[
                    sink_baryon,
                    :,
                    1,
                    source_baryon,
                    :,
                    1,
                ] *
                source_rotations[source_baryon]'
            )
        end
        rotated_two_baryon_propagators[flavor] = rotated
    end
    @test local_two_baryon_correlator(
        rotated_two_baryon_propagators,
        xi_flavors,
        xi_flavors,
        baryon_tensors,
        baryon_tensors,
    ) ≈ factorized_two_baryon rtol = 5e-11 atol = 5e-11

    @test_throws KeyError local_two_baryon_correlator(
        Dict(:s => diagonal_propagator),
        xi_flavors,
        xi_flavors,
        baryon_tensors,
        baryon_tensors,
    )
    @test_throws DimensionMismatch local_two_baryon_correlator(
        Dict(
            :s => zeros(ComplexF64, 2, 3, 1, 2, 3, 2),
            :u => zeros(ComplexF64, 2, 3, 1, 2, 3, 2),
        ),
        xi_flavors,
        xi_flavors,
        baryon_tensors,
        baryon_tensors,
    )
    @test_throws ArgumentError local_two_baryon_correlator(
        diagonal_propagators,
        xi_flavors,
        xi_flavors,
        baryon_tensors,
        baryon_tensors;
        source_baryon_order=(1, 1),
    )
end

@testset "local three-quark contractions" begin
    identity_propagator = zeros(ComplexF64, 3, 4, 3, 4)
    for color in 1:3, spin in 1:4
        identity_propagator[color, spin, color, spin] = 1
    end
    propagators = Dict(
        :u => identity_propagator,
        :d => copy(identity_propagator),
        :s => copy(identity_propagator),
    )

    tensor = zeros(ComplexF64, 4, 4, 4)
    tensor[1, 2, 3] = 1
    @test local_three_quark_correlator(
        propagators,
        (:u, :d, :s),
        tensor,
        tensor,
    ) ≈ 6
    @test local_three_quark_correlator(
        propagators,
        (:u, :d, :u),
        tensor,
        tensor,
    ) ≈ 6

    repeated_spin = zeros(ComplexF64, 4, 4, 4)
    repeated_spin[1, 2, 1] = 1
    @test local_three_quark_correlator(
        propagators,
        (:u, :d, :s),
        repeated_spin,
        repeated_spin,
    ) ≈ 6
    # The two identical u fields are permitted because the color wavefunction
    # is antisymmetric. Their direct and exchange Wick terms add, rather than
    # cancel, for this spin tensor.
    @test local_three_quark_correlator(
        propagators,
        (:u, :d, :u),
        repeated_spin,
        repeated_spin,
    ) ≈ 12

    deterministic = Array{ComplexF64}(undef, 3, 4, 3, 4)
    for sink_color in 1:3, sink_spin in 1:4,
            source_color in 1:3, source_spin in 1:4
        real_part =
            0.13sink_color + 0.07sink_spin -
            0.11source_color + 0.05source_spin
        imag_part =
            0.03sink_color * source_spin -
            0.02source_color * sink_spin
        deterministic[
            sink_color,
            sink_spin,
            source_color,
            source_spin,
        ] = real_part + im * imag_part
    end
    original = Dict(:u => deterministic, :d => 0.7deterministic)

    permutations = (
        ((1, 2, 3), 1),
        ((2, 3, 1), 1),
        ((3, 1, 2), 1),
        ((1, 3, 2), -1),
        ((3, 2, 1), -1),
        ((2, 1, 3), -1),
    )
    explicit_direct_minus_exchange = 0.0 + 0.0im
    for (sink_colors, sink_sign) in permutations
        a, b, c = sink_colors
        for (source_colors, source_sign) in permutations
            ap, bp, cp = source_colors
            direct = (
                original[:u][a, 1, ap, 1] *
                original[:d][b, 2, bp, 2] *
                original[:u][c, 3, cp, 3]
            )
            exchange = (
                original[:u][a, 1, cp, 3] *
                original[:d][b, 2, bp, 2] *
                original[:u][c, 3, ap, 1]
            )
            explicit_direct_minus_exchange +=
                sink_sign * source_sign * (direct - exchange)
        end
    end
    @test local_three_quark_correlator(
        original,
        (:u, :d, :u),
        tensor,
        tensor,
    ) ≈ explicit_direct_minus_exchange rtol = 2e-14 atol = 2e-14

    # Independent baseline: enumerate all color and Wick permutations without
    # calling the package's permutation table or closed-form determinant.
    dense_propagators = Dict{Symbol,Array{ComplexF64,4}}()
    for (flavor_index, flavor) in enumerate((:u, :d, :s))
        block = Array{ComplexF64}(undef, 3, 2, 3, 2)
        for a in 1:3, α in 1:2, ap in 1:3, αp in 1:2
            block[a, α, ap, αp] = (
                0.17flavor_index + 0.11a - 0.07ap +
                0.05α + 0.03αp +
                im * (0.02a * αp - 0.04ap * α + 0.01flavor_index)
            )
        end
        dense_propagators[flavor] = block
    end
    dense_sink_tensor = Array{ComplexF64}(undef, 2, 2, 2)
    dense_source_tensor = similar(dense_sink_tensor)
    for α in 1:2, β in 1:2, γ in 1:2
        dense_sink_tensor[α, β, γ] =
            0.13α - 0.09β + 0.07γ + im * (0.03α * γ - 0.02β)
        dense_source_tensor[α, β, γ] =
            -0.04α + 0.12β + 0.05γ + im * (0.02β * γ + 0.01α)
    end
    for flavor_order in (
        (:u, :d, :s),
        (:u, :d, :u),
        (:s, :u, :s),
        (:u, :u, :u),
    )
        determinant_value = local_three_quark_correlator(
            dense_propagators,
            flavor_order,
            dense_sink_tensor,
            dense_source_tensor,
        )
        explicit_value = explicit_three_quark_permutation_sum(
            dense_propagators,
            flavor_order,
            dense_sink_tensor,
            dense_source_tensor,
        )
        package_explicit_value = local_three_quark_correlator_explicit(
            dense_propagators,
            flavor_order,
            dense_sink_tensor,
            dense_source_tensor,
        )
        @test determinant_value ≈ explicit_value rtol = 5e-12 atol = 5e-12
        @test package_explicit_value ≈
              explicit_value rtol = 5e-12 atol = 5e-12
    end

    θ = 0.37
    ϕ = -0.29
    sink_rotation = ComplexF64[
        cos(θ) sin(θ) 0
        -sin(θ) cos(θ) 0
        0 0 1
    ]
    source_rotation = ComplexF64[
        1 0 0
        0 cos(ϕ) im*sin(ϕ)
        0 im*sin(ϕ) cos(ϕ)
    ]
    @test det(sink_rotation) ≈ 1
    @test det(source_rotation) ≈ 1
    @test sink_rotation' * sink_rotation ≈ I
    @test source_rotation' * source_rotation ≈ I

    transformed = Dict{Symbol,Array{ComplexF64,4}}()
    for (flavor, propagator) in original
        rotated = similar(propagator)
        for sink_spin in 1:4, source_spin in 1:4
            rotated[:, sink_spin, :, source_spin] .= (
                sink_rotation *
                propagator[:, sink_spin, :, source_spin] *
                source_rotation'
            )
        end
        transformed[flavor] = rotated
    end

    sink_tensor = zeros(ComplexF64, 4, 4, 4)
    source_tensor = zeros(ComplexF64, 4, 4, 4)
    sink_tensor[1, 2, 3] = 1
    sink_tensor[2, 1, 4] = -0.4 + 0.2im
    source_tensor[1, 2, 3] = 0.3 - 0.1im
    source_tensor[2, 1, 4] = 0.8
    before = local_three_quark_correlator(
        original,
        (:u, :d, :u),
        sink_tensor,
        source_tensor,
    )
    after = local_three_quark_correlator(
        transformed,
        (:u, :d, :u),
        sink_tensor,
        source_tensor,
    )
    @test after ≈ before rtol = 2e-13 atol = 2e-13

    @test_throws KeyError local_three_quark_correlator(
        propagators,
        (:u, :d, :missing),
        tensor,
        tensor,
    )
    @test_throws DimensionMismatch local_three_quark_correlator(
        propagators,
        (:u, :d, :s),
        tensor,
        zeros(ComplexF64, 4, 4, 3),
    )
end
