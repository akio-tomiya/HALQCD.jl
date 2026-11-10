using LinearAlgebra: mul!, transpose
using SparseArrays: sparse, sparsevec

# Spatially resolved baryon contractions with a block algorithm.
#
# A sink baryon B(y) = I[a] T[α] q₁[a₁,α₁](y) ⋯ q_N[a_N,α_N](y) contracted with
# point-to-all (or smeared-source, or wall-source) propagators is a block
#
#     𝔅[ξ₁,…,ξ_N](y) = Σ_{a,α} I[a] T[α] S₁(y)[a₁α₁, ξ₁] ⋯ S_N(y)[a_Nα_N, ξ_N],
#
# where ξ = (source color, source spin). Every source quark sits on the same
# source, so each Wick term of a two-baryon correlator is a product of a block
# at x + r and a block at x with permuted source indices. Summing the signed
# permutations against the source operator once gives a sparse matrix 𝒯 with
#
#     F(r) = Σ_x Σ_{η,ζ} 𝔅₁[η](x + r) 𝒯[η, ζ] 𝔅₂[ζ](x),
#
# and the sum over x is a correlation evaluated with spatial Fourier
# transforms. This is the unified contraction of Doi and Endres, Comput. Phys.
# Commun. 184 (2013) 117, arXiv:1205.0585, organized around the sink blocks.

# Combined color-spin index of the propagator site matrix: c + (s - 1) Nc.
@inline _color_spin_index(color, spin, ncolor) = color + (spin - 1) * ncolor

"""
    dense_baryon_tensor(color_invariant, spin_tensor)

Dense rank-`N` tensor `W[u₁,…,u_N] = I[a₁,…,a_N] T[α₁,…,α_N]` of a local
baryon operator over combined color-spin indices `u = a + (α - 1) Nc`, the
row and column order of the site matrices of a point propagator. `I` is a
rank-`N` [`ColorInvariant`](@ref) and `T` a rank-`N` spin tensor.
"""
function dense_baryon_tensor(
    color_invariant::ColorInvariant{N},
    spin_tensor::AbstractArray{<:Number,N},
) where {N}
    ncolor = color_invariant.ncolor
    nspin = size(spin_tensor, 1)
    all(size(spin_tensor, axis) == nspin for axis in 1:N) ||
        throw(DimensionMismatch("spin tensor must have $N axes of equal length"))
    dimension = ncolor * nspin
    value_type = promote_type(_invariant_value_type(color_invariant), eltype(spin_tensor), ComplexF64)
    tensor = zeros(value_type, ntuple(_ -> dimension, N))
    for (colors, color_value) in color_invariant.entries
        for index in findall(!iszero, spin_tensor)
            spins = Tuple(index)
            u = ntuple(i -> _color_spin_index(colors[i], spins[i], ncolor), N)
            tensor[u...] += color_value * spin_tensor[index]
        end
    end
    return tensor
end

function _check_spatial_propagators(propagators, flavors, ncolor)
    shape = nothing
    for flavor in unique(flavors)
        haskey(propagators, flavor) || throw(KeyError(flavor))
        propagator = propagators[flavor]
        ndims(propagator) == 8 ||
            throw(DimensionMismatch("each propagator must have eight axes"))
        size(propagator, 1) == size(propagator, 3) == ncolor ||
            throw(DimensionMismatch("propagator color axes must have length $ncolor"))
        size(propagator, 2) == size(propagator, 4) ||
            throw(DimensionMismatch("propagator spin axes must match"))
        isnothing(shape) && (shape = size(propagator))
        size(propagator) == shape ||
            throw(DimensionMismatch("all propagators must have the same shape"))
    end
    return shape
end

"""
    baryon_blocks(propagators, flavors, sink_tensor, t; color_invariant)

Sink blocks of one baryon on time slice `t`,

```text
𝔅[ξ₁,…,ξ_N](x) = Σ_u W[u₁,…,u_N] S_{f₁}(x,t)[u₁, ξ₁] ⋯ S_{f_N}(x,t)[u_N, ξ_N],
```

with `W` the [`dense_baryon_tensor`](@ref) of `color_invariant` and
`sink_tensor`, `flavors` the ordered quark flavors, and `S_f(x,t)` the
`(Nc Nspin) × (Nc Nspin)` site matrix of `propagators[f]` (axes
`(sink_color, sink_spin, source_color, source_spin, x, y, z, t)`). The result
has axes `(ξ, x, y, z)` with the `N` source indices flattened in column-major
order. Each site is a product of `N` mode multiplications, so the cost is
`N (Nc Nspin)^(N+1)` per site.
"""
function baryon_blocks(
    propagators::AbstractDict,
    flavors::NTuple{N},
    sink_tensor::AbstractArray{<:Number,N},
    t::Integer;
    color_invariant::ColorInvariant{N},
) where {N}
    ncolor = color_invariant.ncolor
    shape = _check_spatial_propagators(propagators, flavors, ncolor)
    nspin = shape[2]
    size(sink_tensor) == ntuple(_ -> nspin, N) ||
        throw(DimensionMismatch("sink tensor must have $N axes of length $nspin"))
    W = dense_baryon_tensor(color_invariant, sink_tensor)
    return _baryon_blocks(propagators, flavors, W, t, ncolor, nspin, shape)
end

function _baryon_blocks(propagators, flavors, W, t, ncolor, nspin, shape)
    N = ndims(W)
    nx, ny, nz = shape[5:7]
    dimension = ncolor * nspin
    rest = dimension^(N - 1)
    blocks = Array{ComplexF64}(undef, dimension^N, nx, ny, nz)
    current = Matrix{ComplexF64}(undef, dimension, rest)
    product = Matrix{ComplexF64}(undef, dimension, rest)
    initial = reshape(ComplexF64.(W), dimension, rest)
    for z in 1:nz, y in 1:ny, x in 1:nx
        copyto!(current, initial)
        for k in 1:N
            S = reshape(view(propagators[flavors[k]], :, :, :, :, x, y, z, t), dimension, dimension)
            # Contract the leading sink index u_k with S[u_k, ξ_k], then move
            # ξ_k to the end; after N passes the axes are (ξ₁, …, ξ_N).
            mul!(product, transpose(S), current)
            permutedims!(reshape(current, rest, dimension), product, (2, 1))
        end
        blocks[:, x, y, z] .= vec(current)
    end
    return blocks
end

# Bijections from sink quark lines to source antiquark lines that keep the
# flavor, with the sign of the permutation (the Wick determinant expansion).
function _flavor_preserving_permutations(sink_line_flavors, source_line_flavors)
    n = length(sink_line_flavors)
    length(source_line_flavors) == n ||
        throw(DimensionMismatch("sink and source must have the same number of quark lines"))
    permutations = Tuple{Vector{Int},Int}[]
    assignment = zeros(Int, n)
    used = falses(n)
    function extend(line)
        if line > n
            push!(permutations, (copy(assignment), _permutation_sign(assignment)))
            return
        end
        for column in 1:n
            used[column] && continue
            source_line_flavors[column] == sink_line_flavors[line] || continue
            used[column] = true
            assignment[line] = column
            extend(line + 1)
            used[column] = false
        end
    end
    extend(1)
    return permutations
end

# Linear index of a combined color-spin multi-index, column-major.
@inline function _flat_index(indices, dimension)
    flat = 0
    for k in length(indices):-1:1
        flat = flat * dimension + (indices[k] - 1)
    end
    return flat + 1
end

"""
    two_baryon_source_matrix(source_terms, sink_line_flavors, source_line_flavors,
                             dimension, nquark)

Sparse matrix `𝒯[η, ζ]` that contracts the blocks of the two sink baryons
with the source operator, including the signed sum over the Wick
permutations. `source_terms` is a vector of `(coefficient, WA, WB)` with `WA`
and `WB` the dense source tensors of the first and second source antiquark
blocks (in the column order of the Wick determinant), so the source operator
is `Σ_k c_k WA_k ⊗ WB_k`. Exposed for tests.
"""
function two_baryon_source_matrix(
    source_terms,
    sink_line_flavors,
    source_line_flavors,
    dimension::Integer,
    nquark::Integer,
)
    permutations = _flavor_preserving_permutations(sink_line_flavors, source_line_flavors)
    rows = Int[]
    columns = Int[]
    values = ComplexF64[]
    line_indices = zeros(Int, 2 * nquark)
    for (coefficient, WA, WB) in source_terms
        A = findall(!iszero, WA)
        B = findall(!iszero, WB)
        for a in A, b in B
            value = coefficient * WA[a] * WB[b]
            iszero(value) && continue
            for k in 1:nquark
                line_indices[k] = a[k]
                line_indices[nquark + k] = b[k]
            end
            for (permutation, sign) in permutations
                # Sink line i takes the source index of column permutation[i].
                η = ntuple(i -> line_indices[permutation[i]], nquark)
                ζ = ntuple(i -> line_indices[permutation[nquark + i]], nquark)
                push!(rows, _flat_index(η, dimension))
                push!(columns, _flat_index(ζ, dimension))
                push!(values, sign * value)
            end
        end
    end
    n = dimension^nquark
    return sparse(rows, columns, values, n, n)
end

function _two_baryon_lines(sink_flavors, source_flavors, source_baryon_order)
    sink_line_flavors = [flavor for baryon in sink_flavors for flavor in baryon]
    source_line_flavors = [
        flavor for baryon in source_baryon_order for flavor in source_flavors[baryon]
    ]
    return sink_line_flavors, source_line_flavors
end

"""
    two_baryon_four_point(propagators, sink_flavors, source_flavors,
                          sink_terms, source_terms;
                          source_baryon_order=(2, 1), color_invariant=nothing)

Spatially resolved two-baryon four-point function

```text
F(r, t) = Σ_x ⟨ B₁(x + r, t) B₂(x, t) J̄(t₀) ⟩
```

for propagators whose source quarks all sit on one source (a point, a smeared
point, or a wall in a fixed gauge). `propagators[f]` has axes
`(sink_color, sink_spin, source_color, source_spin, x, y, z, t)`.
`sink_flavors` and `source_flavors` hold two ordered `N`-flavor tuples. Spin
structures are linear combinations: `sink_terms` is a vector of
`(coefficient, T₁, T₂)` with rank-`N` sink spin tensors of `B₁` and `B₂`, and
`source_terms` a vector of `(coefficient, T₁′, T₂′)` with the source spin
tensors of the two source baryons. The conventions are those of
[`local_two_baryon_correlator`](@ref): the source antiquark blocks are taken in
`source_baryon_order`, the source color coefficient is the same invariant,
and callers pass source tensors for their Euclidean adjoint. Same-flavor
exchange within and between baryons is included.

The array index `n` of the result stores the displacement `n - 1` modulo the
spatial extent, and the time axis is the absolute sink time. The cost per
time slice is two block computations, one sparse product, and the Fourier
transforms of `2 (Nc Nspin)^N` fields; there is no sum over pairs of sites.
"""
function two_baryon_four_point(
    propagators::AbstractDict,
    sink_flavors::Tuple,
    source_flavors::Tuple,
    sink_terms::AbstractVector,
    source_terms::AbstractVector;
    source_baryon_order::NTuple{2,Int}=(2, 1),
    color_invariant::Union{Nothing,ColorInvariant}=nothing,
)
    length(sink_flavors) == length(source_flavors) == 2 ||
        throw(ArgumentError("two sink and two source baryons are required"))
    sort(collect(source_baryon_order)) == [1, 2] ||
        throw(ArgumentError("source_baryon_order must be a permutation of (1,2)"))
    nquark = length(first(sink_flavors))
    invariant = _resolve_color_invariant(color_invariant, nquark)
    all(length(flavors) == nquark for flavors in (sink_flavors..., source_flavors...)) ||
        throw(ArgumentError("each baryon must contain $nquark ordered flavors"))
    ncolor = invariant.ncolor
    sink_line_flavors, source_line_flavors =
        _two_baryon_lines(sink_flavors, source_flavors, source_baryon_order)
    shape = _check_spatial_propagators(propagators, sink_line_flavors, ncolor)
    nspin = shape[2]
    dimension = ncolor * nspin
    nx, ny, nz, nt = shape[5:8]

    dense(tensor) = dense_baryon_tensor(invariant, tensor)
    # Source term k gives WA ⊗ WB in the column order of the Wick determinant.
    ordered_source_terms = [
        (term[1], dense(term[1 + source_baryon_order[1]]), dense(term[1 + source_baryon_order[2]]))
        for term in source_terms
    ]
    𝒯 = two_baryon_source_matrix(
        ordered_source_terms, sink_line_flavors, source_line_flavors, dimension, nquark,
    )
    sink_dense = [(coefficient, dense(T1), dense(T2)) for (coefficient, T1, T2) in sink_terms]

    forward = (_dft_matrix(nx), _dft_matrix(ny), _dft_matrix(nz))
    backward = conj.(forward)
    volume = nx * ny * nz
    four_point = zeros(ComplexF64, nx, ny, nz, nt)
    for t in 1:nt
        spectrum = zeros(ComplexF64, nx, ny, nz)
        for (coefficient, W1, W2) in sink_dense
            B1 = _baryon_blocks(propagators, sink_flavors[1], W1, t, ncolor, nspin, shape)
            B2 = _baryon_blocks(propagators, sink_flavors[2], W2, t, ncolor, nspin, shape)
            H = reshape(𝒯 * reshape(B2, size(B2, 1), :), size(B2))
            # Σ_x f(x + r) h(x) has Fourier transform f̂(p) ĥ(-p), and the
            # backward transform of h gives ĥ(-p).
            B1_momenta = _spatial_dft(B1, forward)
            H_momenta = _spatial_dft(H, backward)
            spectrum .+= coefficient .* dropdims(sum(B1_momenta .* H_momenta; dims=4); dims=4)
        end
        correlation = _spatial_dft(reshape(spectrum, 1, nx, ny, nz), backward)
        four_point[:, :, :, t] .= dropdims(correlation; dims=4) ./ volume
    end
    return four_point
end

"""
    two_baryon_four_point_explicit(propagators, sink_flavors, source_flavors,
                                   sink_terms, source_terms;
                                   source_baryon_order=(2, 1), color_invariant=nothing)

Slow reference for [`two_baryon_four_point`](@ref): for every displacement
and center it calls [`local_two_baryon_correlator`](@ref) with the sink
baryons at `x + r` and `x`. The cost grows as the square of the spatial
volume, so use it on tiny lattices only.
"""
function two_baryon_four_point_explicit(
    propagators::AbstractDict,
    sink_flavors::Tuple,
    source_flavors::Tuple,
    sink_terms::AbstractVector,
    source_terms::AbstractVector;
    source_baryon_order::NTuple{2,Int}=(2, 1),
    color_invariant::Union{Nothing,ColorInvariant}=nothing,
)
    nquark = length(first(sink_flavors))
    invariant = _resolve_color_invariant(color_invariant, nquark)
    flavors = unique(vcat(collect(sink_flavors[1]), collect(sink_flavors[2])))
    shape = _check_spatial_propagators(propagators, flavors, invariant.ncolor)
    ncolor, nspin = shape[1], shape[2]
    nx, ny, nz, nt = shape[5:8]
    four_point = zeros(ComplexF64, nx, ny, nz, nt)
    local_propagators = Dict(
        flavor => zeros(ComplexF64, 2, ncolor, nspin, 2, ncolor, nspin) for flavor in flavors
    )
    for t in 1:nt, rz in 0:(nz - 1), ry in 0:(ny - 1), rx in 0:(nx - 1)
        total = zero(ComplexF64)
        for z in 1:nz, y in 1:ny, x in 1:nx
            shifted = (mod(x + rx - 1, nx) + 1, mod(y + ry - 1, ny) + 1, mod(z + rz - 1, nz) + 1)
            for flavor in flavors, (baryon, site) in ((1, shifted), (2, (x, y, z)))
                block = view(propagators[flavor], :, :, :, :, site..., t)
                for source_baryon in 1:2
                    local_propagators[flavor][baryon, :, :, source_baryon, :, :] .= block
                end
            end
            for (sink_coefficient, T1, T2) in sink_terms, (source_coefficient, S1, S2) in source_terms
                total += sink_coefficient * source_coefficient * local_two_baryon_correlator(
                    local_propagators, sink_flavors, source_flavors, (T1, T2), (S1, S2);
                    source_baryon_order, color_invariant=invariant,
                )
            end
        end
        four_point[rx + 1, ry + 1, rz + 1, t] = total
    end
    return four_point
end

"""
    baryon_two_point(propagators, flavors, sink_tensor, source_tensor;
                     color_invariant=nothing)

Zero-momentum one-baryon correlator `C(t) = Σ_x ⟨B(x, t) B̄(t₀)⟩` with the
conventions of [`local_color_singlet_correlator`](@ref), evaluated from the
sink blocks: the signed Wick permutations of the source tensor are summed once
into a sparse vector that is contracted with the block at every site. The time
axis is the absolute sink time.
"""
function baryon_two_point(
    propagators::AbstractDict,
    flavors::NTuple{N},
    sink_tensor::AbstractArray{<:Number,N},
    source_tensor::AbstractArray{<:Number,N};
    color_invariant::Union{Nothing,ColorInvariant}=nothing,
) where {N}
    invariant = _resolve_color_invariant(color_invariant, N)
    ncolor = invariant.ncolor
    shape = _check_spatial_propagators(propagators, flavors, ncolor)
    nspin = shape[2]
    dimension = ncolor * nspin
    W = dense_baryon_tensor(invariant, sink_tensor)
    source = dense_baryon_tensor(invariant, source_tensor)
    permutations = _flavor_preserving_permutations(collect(flavors), collect(flavors))
    indices = Int[]
    values = ComplexF64[]
    for index in findall(!iszero, source), (permutation, sign) in permutations
        ξ = Tuple(index)
        push!(indices, _flat_index(ntuple(i -> ξ[permutation[i]], N), dimension))
        push!(values, sign * source[index])
    end
    contraction = sparsevec(indices, values, dimension^N)
    nt = shape[8]
    correlator = zeros(ComplexF64, nt)
    for t in 1:nt
        B = _baryon_blocks(propagators, flavors, W, t, ncolor, nspin, shape)
        correlator[t] = sum(transpose(contraction) * reshape(B, size(B, 1), :))
    end
    return correlator
end
