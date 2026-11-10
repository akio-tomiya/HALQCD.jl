using LinearAlgebra: det

# Permutations of three quark lines with their signs.  The same table, read
# as color indices, gives the nonzero entries of the SU(3) Levi-Civita tensor.
const _THREE_LINE_PERMUTATIONS = (
    ((1, 2, 3), 1),
    ((2, 3, 1), 1),
    ((3, 1, 2), 1),
    ((1, 3, 2), -1),
    ((3, 2, 1), -1),
    ((2, 1, 3), -1),
)

# Same entry order as the table above, so SU(3) sums keep their previous
# floating-point order.
const _SU3_EPSILON = ColorInvariant{3,Int}(
    "epsilon3",
    3,
    [(colors, sign) for (colors, sign) in _THREE_LINE_PERMUTATIONS],
)

baryon_invariant(::SU{3}) = _SU3_EPSILON

function _resolve_color_invariant(color_invariant, nquark)
    if isnothing(color_invariant)
        nquark == 3 ||
            throw(ArgumentError("color_invariant is required for $nquark quarks"))
        return _SU3_EPSILON
    end
    color_rank(color_invariant) == nquark ||
        throw(
            DimensionMismatch(
                "color invariant has rank $(color_rank(color_invariant)) " *
                "but the operator has $nquark quarks",
            ),
        )
    return color_invariant
end

"""
    baryon_spin_tensor(diquark_tensor, spectator_spin)

Embed a two-spin diquark tensor in a zero-initialized rank-three baryon spin
tensor. The first two axes are the diquark spins and the third axis is the
spectator-quark spin. Explicit zero initialization is part of the contract:
entries outside the selected spectator-spin slice are exactly zero.
"""
function baryon_spin_tensor(
    diquark_tensor::AbstractMatrix,
    spectator_spin::Integer,
)
    nspin_sink, nspin_source = size(diquark_tensor)
    nspin_sink == nspin_source ||
        throw(DimensionMismatch("diquark spin tensor must be square"))
    1 <= spectator_spin <= nspin_sink ||
        throw(BoundsError(axes(diquark_tensor, 1), spectator_spin))
    tensor = zeros(eltype(diquark_tensor), nspin_sink, nspin_sink, nspin_sink)
    tensor[:, :, spectator_spin] .= diquark_tensor
    return tensor
end

@inline function _permutation_sign(indices)
    length(indices) < 2 && return 1
    inversions = sum(
        indices[i] > indices[j]
        for i in 1:(length(indices) - 1) for j in (i + 1):length(indices)
    )
    return isodd(inversions) ? -1 : 1
end

"""
    flavor_factorized_wick_determinant(
        wick_matrix,
        sink_flavors,
        source_flavors,
    )

Evaluate a many-quark Wick determinant using flavor conservation. Rows of
`wick_matrix` label sink quark lines and columns label source antiquark lines.
Only entries whose sink and source flavors match are used. The full determinant
is factored into one smaller determinant per flavor.

The sink and source flavor sequences may use different orderings. Their
permutation signs are included, so the result equals the determinant of the
full matrix after setting flavor-off-diagonal entries to zero. If the two sides
do not contain the same flavor multiplicities, the contraction is zero.
"""
function flavor_factorized_wick_determinant(
    wick_matrix::AbstractMatrix,
    sink_flavors,
    source_flavors,
)
    nsink = length(sink_flavors)
    nsource = length(source_flavors)
    size(wick_matrix) == (nsink, nsource) ||
        throw(
            DimensionMismatch(
                "wick_matrix shape must match sink and source flavor counts",
            ),
        )
    nsink == nsource ||
        throw(DimensionMismatch("a Wick determinant must be square"))
    iszero(nsink) && return one(eltype(wick_matrix))

    factorization_plan =
        _flavor_factorization_plan(sink_flavors, source_flavors)
    isnothing(factorization_plan) && return zero(eltype(wick_matrix))
    return _factorized_determinant_from_plan(
        wick_matrix,
        factorization_plan,
    )
end

@inline function _det3(
    a11,
    a12,
    a13,
    a21,
    a22,
    a23,
    a31,
    a32,
    a33,
)
    return (
        a11 * (a22 * a33 - a23 * a32) -
        a12 * (a21 * a33 - a23 * a31) +
        a13 * (a21 * a32 - a22 * a31)
    )
end

@inline function _det4(
    a11,
    a12,
    a13,
    a14,
    a21,
    a22,
    a23,
    a24,
    a31,
    a32,
    a33,
    a34,
    a41,
    a42,
    a43,
    a44,
)
    return (
        a11 * _det3(a22, a23, a24, a32, a33, a34, a42, a43, a44) -
        a12 * _det3(a21, a23, a24, a31, a33, a34, a41, a43, a44) +
        a13 * _det3(a21, a22, a24, a31, a32, a34, a41, a42, a44) -
        a14 * _det3(a21, a22, a23, a31, a32, a33, a41, a42, a43)
    )
end

@inline function _selected_determinant(matrix, rows, columns)
    n = length(rows)
    n == length(columns) ||
        throw(DimensionMismatch("selected determinant must be square"))
    iszero(n) && return one(eltype(matrix))
    n == 1 && return @inbounds matrix[rows[1], columns[1]]
    n == 2 &&
        return @inbounds (
            matrix[rows[1], columns[1]] * matrix[rows[2], columns[2]] -
            matrix[rows[1], columns[2]] * matrix[rows[2], columns[1]]
        )
    n == 3 &&
        return @inbounds _det3(
            matrix[rows[1], columns[1]],
            matrix[rows[1], columns[2]],
            matrix[rows[1], columns[3]],
            matrix[rows[2], columns[1]],
            matrix[rows[2], columns[2]],
            matrix[rows[2], columns[3]],
            matrix[rows[3], columns[1]],
            matrix[rows[3], columns[2]],
            matrix[rows[3], columns[3]],
        )
    n == 4 &&
        return @inbounds _det4(
            matrix[rows[1], columns[1]],
            matrix[rows[1], columns[2]],
            matrix[rows[1], columns[3]],
            matrix[rows[1], columns[4]],
            matrix[rows[2], columns[1]],
            matrix[rows[2], columns[2]],
            matrix[rows[2], columns[3]],
            matrix[rows[2], columns[4]],
            matrix[rows[3], columns[1]],
            matrix[rows[3], columns[2]],
            matrix[rows[3], columns[3]],
            matrix[rows[3], columns[4]],
            matrix[rows[4], columns[1]],
            matrix[rows[4], columns[2]],
            matrix[rows[4], columns[3]],
            matrix[rows[4], columns[4]],
        )
    return det(matrix[rows, columns])
end

function _flavor_factorization_plan(sink_flavors, source_flavors)
    flavor_order = unique(sink_flavors)
    grouped_sink_indices = Int[]
    grouped_source_indices = Int[]
    groups = Tuple{Vector{Int},Vector{Int}}[]

    for flavor in flavor_order
        sink_indices = findall(==(flavor), sink_flavors)
        source_indices = findall(==(flavor), source_flavors)
        length(sink_indices) == length(source_indices) || return nothing
        append!(grouped_sink_indices, sink_indices)
        append!(grouped_source_indices, source_indices)
        push!(groups, (sink_indices, source_indices))
    end

    length(grouped_source_indices) == length(source_flavors) || return nothing
    sign = (
        _permutation_sign(grouped_sink_indices) *
        _permutation_sign(grouped_source_indices)
    )
    return sign, groups
end

@inline function _factorized_determinant_from_plan(wick_matrix, plan)
    sign, groups = plan
    determinant_product = one(eltype(wick_matrix))
    for (sink_indices, source_indices) in groups
        determinant_product *=
            _selected_determinant(wick_matrix, sink_indices, source_indices)
    end
    return sign * determinant_product
end

@inline function _has_exact_zero_row_or_column(matrix)
    nrows, ncolumns = size(matrix)
    for row in 1:nrows
        all_zero = true
        for column in 1:ncolumns
            value = @inbounds matrix[row, column]
            if !iszero(value)
                all_zero = false
                break
            end
        end
        all_zero && return true
    end
    for column in 1:ncolumns
        all_zero = true
        for row in 1:nrows
            value = @inbounds matrix[row, column]
            if !iszero(value)
                all_zero = false
                break
            end
        end
        all_zero && return true
    end
    return false
end

_invariant_value_type(::ColorInvariant{R,T}) where {R,T} = T

function _check_local_propagators(propagators, flavors, ncolor, nspin)
    for flavor in unique(flavors)
        haskey(propagators, flavor) ||
            throw(KeyError(flavor))
        propagator = propagators[flavor]
        ndims(propagator) == 4 ||
            throw(DimensionMismatch("each propagator must have four axes"))
        size(propagator) == (ncolor, nspin, ncolor, nspin) ||
            throw(
                DimensionMismatch(
                    "propagator axes must be ($ncolor,Nspin,$ncolor,Nspin)",
                ),
            )
    end
    return nothing
end

function _check_spin_tensors(sink_tensor, source_tensor, nquark)
    size(sink_tensor) == size(source_tensor) ||
        throw(DimensionMismatch("sink and source spin tensors must match"))
    nspin = size(sink_tensor, 1)
    all(size(sink_tensor, axis) == nspin for axis in 1:nquark) ||
        throw(
            DimensionMismatch(
                "spin tensors must have $nquark axes of length Nspin",
            ),
        )
    return nspin
end

"""
    local_color_singlet_correlator(
        propagators,
        flavors,
        sink_tensor,
        source_tensor;
        color_invariant,
        source_color_invariant=color_invariant,
    )

Contract one local color-singlet operator made of `N` quarks at a sink point
with its source operator. `propagators[f]` has axes
`(sink_color, sink_spin, source_color, source_spin)` and contains
`<q_f(x) q̄_f(0)>`. `flavors` gives the ordered flavors of the `N` quark
fields. The operator is

```text
O(x) = I[a₁,…,a_N] T[α₁,…,α_N] q₁[a₁,α₁] ⋯ q_N[a_N,α_N]
```

where `I` is the rank-`N` [`ColorInvariant`](@ref) and `T` the spin tensor.
The source antiquarks are ordered in reverse, `q̄_N ⋯ q̄₁`, with color
coefficient `source_color_invariant` and spin coefficient `source_tensor`.
For a complex invariant the caller passes the conjugated tensor as
`source_color_invariant`; no adjoint is inferred here.

The color group enters only through the invariant and the length of the color
axes, so the same function serves `SU(N)` with the Levi-Civita tensor and
any other group through its invariant. The Wick contraction over quark lines is
evaluated as an `N × N` determinant with flavor-off-diagonal entries set to
zero.
"""
function local_color_singlet_correlator(
    propagators::AbstractDict,
    flavors::NTuple{N},
    sink_tensor::AbstractArray{<:Number,N},
    source_tensor::AbstractArray{<:Number,N};
    color_invariant::ColorInvariant{N},
    source_color_invariant::ColorInvariant{N}=color_invariant,
) where {N}
    color_invariant.ncolor == source_color_invariant.ncolor ||
        throw(DimensionMismatch("sink and source invariants act on different colors"))
    nspin = _check_spin_tensors(sink_tensor, source_tensor, N)
    _check_local_propagators(propagators, flavors, color_invariant.ncolor, nspin)

    value_type = promote_type(
        eltype(sink_tensor),
        eltype(source_tensor),
        _invariant_value_type(color_invariant),
        _invariant_value_type(source_color_invariant),
        (eltype(propagators[flavor]) for flavor in flavors)...,
    )
    total = zero(value_type)
    sink_indices = findall(!iszero, sink_tensor)
    source_indices = findall(!iszero, source_tensor)
    wick_matrix = Matrix{value_type}(undef, N, N)

    for (sink_colors, sink_color_value) in color_invariant.entries
        for (source_colors, source_color_value) in source_color_invariant.entries
            color_coefficient = sink_color_value * source_color_value
            for sink_index in sink_indices
                sink_spins = Tuple(sink_index)
                sink_coefficient = sink_tensor[sink_index]
                for source_index in source_indices
                    source_spins = Tuple(source_index)
                    source_coefficient = source_tensor[source_index]
                    for i in 1:N, j in 1:N
                        @inbounds wick_matrix[i, j] = if flavors[i] == flavors[j]
                            propagators[flavors[i]][
                                sink_colors[i],
                                sink_spins[i],
                                source_colors[j],
                                source_spins[j],
                            ]
                        else
                            zero(value_type)
                        end
                    end
                    wick_determinant =
                        _selected_determinant(wick_matrix, 1:N, 1:N)
                    total += (
                        color_coefficient *
                        sink_coefficient *
                        source_coefficient *
                        wick_determinant
                    )
                end
            end
        end
    end

    return total
end

"""
    local_color_singlet_correlator_explicit(
        propagators,
        flavors,
        sink_tensor,
        source_tensor;
        color_invariant,
        source_color_invariant=color_invariant,
    )

Slow validation oracle for [`local_color_singlet_correlator`](@ref). The Wick
contraction is written as the signed sum over all `N!` assignments of source
antiquark lines to sink quark lines instead of a determinant.
"""
function local_color_singlet_correlator_explicit(
    propagators::AbstractDict,
    flavors::NTuple{N},
    sink_tensor::AbstractArray{<:Number,N},
    source_tensor::AbstractArray{<:Number,N};
    color_invariant::ColorInvariant{N},
    source_color_invariant::ColorInvariant{N}=color_invariant,
) where {N}
    color_invariant.ncolor == source_color_invariant.ncolor ||
        throw(DimensionMismatch("sink and source invariants act on different colors"))
    nspin = _check_spin_tensors(sink_tensor, source_tensor, N)
    _check_local_propagators(propagators, flavors, color_invariant.ncolor, nspin)

    value_type = promote_type(
        eltype(sink_tensor),
        eltype(source_tensor),
        _invariant_value_type(color_invariant),
        _invariant_value_type(source_color_invariant),
        (eltype(propagators[flavor]) for flavor in flavors)...,
    )
    line_assignments = if N == 3
        # Keep the summation order of the original three-quark oracle.
        collect(_THREE_LINE_PERMUTATIONS)
    else
        [
            (Tuple(permutation), _permutation_sign(permutation))
            for permutation in _permutations(collect(1:N))
        ]
    end
    total = zero(value_type)
    sink_indices = findall(!iszero, sink_tensor)
    source_indices = findall(!iszero, source_tensor)

    for (sink_colors, sink_color_value) in color_invariant.entries
        for (source_colors, source_color_value) in source_color_invariant.entries
            color_coefficient = sink_color_value * source_color_value
            for sink_index in sink_indices
                sink_spins = Tuple(sink_index)
                sink_coefficient = sink_tensor[sink_index]
                for source_index in source_indices
                    source_spins = Tuple(source_index)
                    source_coefficient = source_tensor[source_index]
                    for (source_assignment, wick_sign) in line_assignments
                        all(
                            flavors[sink_line] ==
                                flavors[source_assignment[sink_line]]
                            for sink_line in 1:N
                        ) || continue
                        product = one(value_type)
                        for sink_line in 1:N
                            source_line = source_assignment[sink_line]
                            product *= propagators[flavors[sink_line]][
                                sink_colors[sink_line],
                                sink_spins[sink_line],
                                source_colors[source_line],
                                source_spins[source_line],
                            ]
                        end
                        total += (
                            color_coefficient *
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

"""
    local_three_quark_correlator(
        propagators,
        flavors,
        sink_tensor,
        source_tensor;
        color_invariant=nothing,
    )

Contract one local color-singlet three-quark operator at a sink point with its
source operator. `propagators[f]` has axes
`(sink_color, sink_spin, source_color, source_spin)` and contains
`<q_f(x) q̄_f(0)>`. `flavors` gives the ordered flavors of the three quark
fields.

The spin tensors define

```text
B(x) = I[a,b,c] T[α,β,γ] q₁[a,α] q₂[b,β] q₃[c,γ]
```

and the source antiquarks are ordered in reverse,
`q̄₃ q̄₂ q̄₁`, with coefficient `source_tensor`. The function deliberately
does not infer the Euclidean adjoint of an interpolating operator; callers
must construct the appropriate source tensor for their gamma-matrix
convention.

The color invariant `I` defaults to the SU(3) Levi-Civita tensor, which
requires three colors. Pass a rank-three [`ColorInvariant`](@ref) to use
another group. This is
[`local_color_singlet_correlator`](@ref) with three quarks.

The Wick contraction is evaluated as a three-by-three determinant. Flavor
off-diagonal propagators are zero, so repeated flavors automatically include
the required direct and exchange terms.
"""
function local_three_quark_correlator(
    propagators::AbstractDict,
    flavors::NTuple{3},
    sink_tensor::AbstractArray{<:Number,3},
    source_tensor::AbstractArray{<:Number,3};
    color_invariant::Union{Nothing,ColorInvariant{3}}=nothing,
)
    return local_color_singlet_correlator(
        propagators,
        flavors,
        sink_tensor,
        source_tensor;
        color_invariant=_resolve_color_invariant(color_invariant, 3),
    )
end

"""
    local_three_quark_correlator_explicit(
        propagators,
        flavors,
        sink_tensor,
        source_tensor;
        color_invariant=nothing,
    )

Slow validation oracle for [`local_three_quark_correlator`](@ref). This
function enumerates the six Wick permutations directly instead of evaluating
a three-by-three determinant. It is intended for small regression inputs and
independent numerical checks, not production contractions.
"""
function local_three_quark_correlator_explicit(
    propagators::AbstractDict,
    flavors::NTuple{3},
    sink_tensor::AbstractArray{<:Number,3},
    source_tensor::AbstractArray{<:Number,3};
    color_invariant::Union{Nothing,ColorInvariant{3}}=nothing,
)
    return local_color_singlet_correlator_explicit(
        propagators,
        flavors,
        sink_tensor,
        source_tensor;
        color_invariant=_resolve_color_invariant(color_invariant, 3),
    )
end

function _local_baryon_terms(tensor, color_invariant)
    nonzero_indices = findall(!iszero, tensor)
    return [
        (
            colors=colors,
            spins=Tuple(index),
            coefficient=color_value * tensor[index],
        )
        for (colors, color_value) in color_invariant.entries
        for index in nonzero_indices
    ]
end

"""
    local_two_baryon_correlator(
        propagators,
        sink_flavors,
        source_flavors,
        sink_tensors,
        source_tensors;
        source_baryon_order=(2, 1),
        factorize_flavor=true,
        color_invariant=nothing,
    )

Contract two local color-singlet baryon operators, retaining quark exchange
between the baryons. Each baryon has `N` quarks, where `N` is the rank of
`color_invariant`; the default is the SU(3) Levi-Civita tensor with `N=3`.
`sink_flavors` and `source_flavors` each contain two ordered `N`-flavor
tuples. The corresponding tensor tuples contain the two rank-`N` spin tensors.

For each flavor `f`, `propagators[f]` has axes

```text
(sink_baryon, sink_color, sink_spin,
 source_baryon, source_color, source_spin)
```

with both baryon axes of length two and color axes of length
`color_invariant.ncolor`. The `2N`-quark Wick determinant is evaluated with
[`flavor_factorized_wick_determinant`](@ref), so same-flavor exchange within
and between baryons is included.

The default source order is reversed because `(B₁ B₂)† = B₂† B₁†`. Changing
the order of two `N`-antiquark blocks changes the Wick determinant by
`(-1)^(N*N)`, which is `-1` for three quarks. As for
[`local_three_quark_correlator`](@ref), callers must construct source tensors
for their Euclidean-adjoint convention. Coupled baryon-spin states are formed
by linear combinations of calls to this single-tensor-product function. Set
`factorize_flavor=false` only as a small full-determinant validation path.
"""
function local_two_baryon_correlator(
    propagators::AbstractDict,
    sink_flavors::Tuple,
    source_flavors::Tuple,
    sink_tensors::Tuple,
    source_tensors::Tuple;
    source_baryon_order::NTuple{2,Int}=(2, 1),
    factorize_flavor::Bool=true,
    color_invariant::Union{Nothing,ColorInvariant}=nothing,
)
    all(length(collection) == 2 for collection in (
        sink_flavors,
        source_flavors,
        sink_tensors,
        source_tensors,
    )) || throw(ArgumentError("two sink and two source baryons are required"))
    nquark = isnothing(color_invariant) ? 3 : color_rank(color_invariant)
    invariant = _resolve_color_invariant(color_invariant, nquark)
    ncolor = invariant.ncolor
    all(length(flavors) == nquark for flavors in (sink_flavors..., source_flavors...)) ||
        throw(ArgumentError("each baryon must contain $nquark ordered flavors"))
    sort(collect(source_baryon_order)) == [1, 2] ||
        throw(ArgumentError("source_baryon_order must be a permutation of (1,2)"))
    all(
        tensor isa AbstractArray{<:Number,nquark}
        for tensor in (sink_tensors..., source_tensors...)
    ) || throw(ArgumentError("each spin tensor must be a rank-$nquark numeric array"))

    nspin = size(sink_tensors[1], 1)
    expected_tensor_shape = ntuple(_ -> nspin, nquark)
    all(
        size(tensor) == expected_tensor_shape
        for tensor in (sink_tensors..., source_tensors...)
    ) || throw(DimensionMismatch("all spin tensors must have $nquark axes of length Nspin"))

    sink_line_flavors = [
        flavor for baryon_flavors in sink_flavors for flavor in baryon_flavors
    ]
    source_line_flavors = [
        flavor
        for baryon in source_baryon_order
        for flavor in source_flavors[baryon]
    ]
    all_flavors = unique(vcat(sink_line_flavors, source_line_flavors))
    expected_propagator_shape = (2, ncolor, nspin, 2, ncolor, nspin)
    for flavor in all_flavors
        haskey(propagators, flavor) || throw(KeyError(flavor))
        size(propagators[flavor]) == expected_propagator_shape ||
            throw(
                DimensionMismatch(
                    "propagator axes must be (2,$ncolor,Nspin,2,$ncolor,Nspin)",
                ),
            )
    end

    value_type = promote_type(
        (eltype(tensor) for tensor in (sink_tensors..., source_tensors...))...,
        _invariant_value_type(invariant),
        (eltype(propagators[flavor]) for flavor in all_flavors)...,
    )
    total = zero(value_type)
    sink_terms = map(tensor -> _local_baryon_terms(tensor, invariant), sink_tensors)
    source_terms = map(tensor -> _local_baryon_terms(tensor, invariant), source_tensors)
    nline = 2 * nquark
    sink_baryon_by_line = (
        ntuple(_ -> 1, nquark)...,
        ntuple(_ -> 2, nquark)...,
    )
    source_baryon_by_line = (
        ntuple(_ -> source_baryon_order[1], nquark)...,
        ntuple(_ -> source_baryon_order[2], nquark)...,
    )
    factorization_plan = if factorize_flavor
        _flavor_factorization_plan(
            sink_line_flavors,
            source_line_flavors,
        )
    else
        nothing
    end
    factorize_flavor && isnothing(factorization_plan) && return total
    factorization_sign, factorized_groups = if factorize_flavor
        sign, groups = factorization_plan
        prepared_groups = [
            (
                sink_lines=sink_indices,
                source_lines=source_indices,
                propagator=propagators[sink_line_flavors[first(sink_indices)]],
                has_exact_zeros=any(
                    iszero,
                    propagators[sink_line_flavors[first(sink_indices)]],
                ),
                matrix=Matrix{value_type}(
                    undef,
                    length(sink_indices),
                    length(source_indices),
                ),
            )
            for (sink_indices, source_indices) in groups
        ]
        # Multiplication of the flavor determinants is commutative.  Inspect
        # the smaller blocks first so an exactly singular sparse block can
        # terminate the contraction before the larger blocks are populated.
        sort!(prepared_groups; by=group -> length(group.sink_lines))
        sign, prepared_groups
    else
        1, nothing
    end
    wick_matrix = factorize_flavor ? nothing : Matrix{value_type}(undef, nline, nline)

    for sink_term_1 in sink_terms[1], sink_term_2 in sink_terms[2]
        sink_colors = (sink_term_1.colors..., sink_term_2.colors...)
        sink_spins = (sink_term_1.spins..., sink_term_2.spins...)
        sink_coefficient =
            sink_term_1.coefficient * sink_term_2.coefficient

        for source_term_first in source_terms[source_baryon_order[1]]
            for source_term_second in source_terms[source_baryon_order[2]]
                source_colors = (
                    source_term_first.colors...,
                    source_term_second.colors...,
                )
                source_spins = (
                    source_term_first.spins...,
                    source_term_second.spins...,
                )
                source_coefficient = (
                    source_term_first.coefficient *
                    source_term_second.coefficient
                )

                wick_determinant = if factorize_flavor
                    determinant_product = one(value_type)
                    for group in factorized_groups
                        for (sink_group_index, sink_line) in
                                enumerate(group.sink_lines)
                            for (source_group_index, source_line) in
                                    enumerate(group.source_lines)
                                @inbounds group.matrix[
                                    sink_group_index,
                                    source_group_index,
                                ] = group.propagator[
                                    sink_baryon_by_line[sink_line],
                                    sink_colors[sink_line],
                                    sink_spins[sink_line],
                                    source_baryon_by_line[source_line],
                                    source_colors[source_line],
                                    source_spins[source_line],
                                ]
                            end
                        end
                        if group.has_exact_zeros &&
                                _has_exact_zero_row_or_column(group.matrix)
                            determinant_product = zero(value_type)
                            break
                        end
                        determinant_product *= _selected_determinant(
                            group.matrix,
                            axes(group.matrix, 1),
                            axes(group.matrix, 2),
                        )
                    end
                    factorization_sign * determinant_product
                else
                    for sink_line in 1:nline, source_line in 1:nline
                        if sink_line_flavors[sink_line] ==
                                source_line_flavors[source_line]
                            flavor = sink_line_flavors[sink_line]
                            @inbounds wick_matrix[sink_line, source_line] =
                                propagators[flavor][
                                    sink_baryon_by_line[sink_line],
                                    sink_colors[sink_line],
                                    sink_spins[sink_line],
                                    source_baryon_by_line[source_line],
                                    source_colors[source_line],
                                    source_spins[source_line],
                                ]
                        else
                            @inbounds wick_matrix[sink_line, source_line] =
                                zero(value_type)
                        end
                    end
                    det(wick_matrix)
                end
                total += (
                    sink_coefficient * source_coefficient * wick_determinant
                )
            end
        end
    end
    return total
end
