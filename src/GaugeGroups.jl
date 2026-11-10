using LinearAlgebra: det, qr, Diagonal
using Random: AbstractRNG

"""
    AbstractGaugeGroup

Supertype of the gauge groups understood by the hadron-contraction layer.
Group-dependent behavior is selected by dispatch on a concrete subtype. The
potential-extraction layer does not take a group argument because it is
independent of the gauge group.

A new group is added outside this package by a subtype with methods of
[`representation_dimension`](@ref), [`representation_reality`](@ref),
[`baryon_quark_number`](@ref), [`baryon_invariant`](@ref),
[`lie_algebra_projection`](@ref), [`group_defect`](@ref) and
[`random_group_element`](@ref); the contractions, stout smearing and the
Wilson-clover operator then work for it.
"""
abstract type AbstractGaugeGroup end

"""
    SU{N}()

Special unitary group `SU(N)` with quarks in the `N`-dimensional fundamental
representation.
"""
struct SU{N} <: AbstractGaugeGroup
    function SU{N}() where {N}
        N isa Integer && N >= 2 ||
            throw(ArgumentError("SU{N} requires an integer N >= 2"))
        return new{N}()
    end
end

Base.show(io::IO, ::SU{N}) where {N} = print(io, "SU(", N, ")")

"""
    representation_dimension(group)

Dimension of the quark representation, which is the length of every color
axis of a quark propagator.
"""
representation_dimension(::SU{N}) where {N} = N

"""
    representation_reality(group)

Return `:complex`, `:pseudoreal`, or `:real` for the quark representation.
For a pseudoreal or real representation a quark and an antiquark belong to
the same multiplet, so two-quark color singlets exist.
"""
representation_reality(::SU{2}) = :pseudoreal
representation_reality(::SU{N}) where {N} = :complex

"""
    baryon_quark_number(group)

Number of quarks in the lightest fermionic color-singlet built from quarks
alone with the baryon invariant of [`baryon_invariant`](@ref).
"""
baryon_quark_number(::SU{N}) where {N} = N

"""
    ColorInvariant(name, ncolor, entries)

Sparse rank-`R` color tensor stored as its nonzero entries. Each entry is a
tuple of `R` color indices and its value. The contraction functions sum over
these entries, so the number of nonzero entries sets the color cost.
"""
struct ColorInvariant{R,T<:Number}
    name::String
    ncolor::Int
    entries::Vector{Tuple{NTuple{R,Int},T}}

    function ColorInvariant{R,T}(
        name::AbstractString,
        ncolor::Integer,
        entries::AbstractVector,
    ) where {R,T<:Number}
        ncolor >= 1 || throw(ArgumentError("ncolor must be positive"))
        stored = Tuple{NTuple{R,Int},T}[]
        seen = Set{NTuple{R,Int}}()
        for (indices, value) in entries
            length(indices) == R ||
                throw(DimensionMismatch("every entry must have $R color indices"))
            all(1 <= index <= ncolor for index in indices) ||
                throw(BoundsError(1:ncolor, indices))
            key = NTuple{R,Int}(indices)
            key in seen && throw(ArgumentError("duplicate color entry $key"))
            push!(seen, key)
            iszero(value) || push!(stored, (key, T(value)))
        end
        return new{R,T}(String(name), Int(ncolor), stored)
    end
end

function ColorInvariant(
    name::AbstractString,
    ncolor::Integer,
    entries::AbstractVector{<:Tuple{NTuple{R,<:Integer},T}},
) where {R,T<:Number}
    return ColorInvariant{R,T}(name, ncolor, entries)
end

color_rank(::ColorInvariant{R}) where {R} = R
Base.length(invariant::ColorInvariant) = length(invariant.entries)

"""
    dense_color_tensor(invariant)

Dense array with `ncolor` along each of the rank axes. Intended for tests and
small checks; the contractions use the sparse entries.
"""
function dense_color_tensor(invariant::ColorInvariant{R,T}) where {R,T}
    tensor = zeros(T, ntuple(_ -> invariant.ncolor, R))
    for (indices, value) in invariant.entries
        tensor[indices...] = value
    end
    return tensor
end

"""
    levi_civita_invariant(n)

Rank-`n` totally antisymmetric tensor `ε_{a₁…aₙ}` with `ε_{1…n}=1`. It has
`n!` nonzero entries.
"""
function levi_civita_invariant(n::Integer)
    n >= 1 || throw(ArgumentError("rank must be positive"))
    entries = Tuple{NTuple{n,Int},Int}[]
    for permutation in _permutations(collect(1:n))
        push!(entries, (Tuple(permutation), _permutation_sign(permutation)))
    end
    return ColorInvariant{n,Int}("epsilon$n", n, entries)
end

"""
    kronecker_invariant(n)

Rank-two invariant `δ_ab` for an `n`-dimensional representation.
"""
function kronecker_invariant(n::Integer)
    return ColorInvariant{2,Int}("delta$n", n, [((a, a), 1) for a in 1:n])
end

"""
    meson_invariant(group)

Color invariant pairing a quark with an antiquark. It is `δ_ab` for every
supported group.
"""
meson_invariant(group::AbstractGaugeGroup) =
    kronecker_invariant(representation_dimension(group))

"""
    baryon_invariant(group)

Totally antisymmetric invariant used for baryons. For `SU(N)` it is the
rank-`N` Levi-Civita tensor. Another group defines its own method, which
makes every baryon contraction work for it.
"""
baryon_invariant(::SU{N}) where {N} = levi_civita_invariant(N)

"""
    random_group_element(group, rng)

Random group element in the quark representation, used to test color
invariance. For `SU(N)` the matrix is Haar distributed.
"""
function random_group_element(::SU{N}, rng::AbstractRNG) where {N}
    z = (randn(rng, N, N) .+ im .* randn(rng, N, N)) ./ sqrt(2)
    factorization = qr(z)
    q = Matrix(factorization.Q)
    r = factorization.R
    phases = [iszero(r[i, i]) ? one(ComplexF64) : r[i, i] / abs(r[i, i]) for i in 1:N]
    u = q * Diagonal(phases)
    u[:, N] .*= conj(det(u))
    return u
end


"""
    transform_color_tensor(tensor, g; conjugate_axes=())

Apply the matrix `g` to every axis of a dense color tensor,
`T'[a₁…] = Σ g[a₁,b₁] ⋯ T[b₁…]`. Axes listed in `conjugate_axes` carry an
antiquark index and are transformed with `conj(g)` instead.
"""
function transform_color_tensor(
    tensor::AbstractArray,
    g::AbstractMatrix;
    conjugate_axes=(),
)
    n = size(g, 1)
    size(g, 2) == n || throw(DimensionMismatch("g must be square"))
    all(==(n), size(tensor)) ||
        throw(DimensionMismatch("every tensor axis must match size(g)"))
    result = Array{promote_type(eltype(tensor), eltype(g))}(tensor)
    rank = ndims(tensor)
    for axis in 1:rank
        moved = permutedims(result, (axis, setdiff(1:rank, axis)...))
        matrix = axis in conjugate_axes ? conj(g) : g
        updated = reshape(matrix * reshape(moved, n, :), size(moved))
        result = permutedims(updated, invperm((axis, setdiff(1:rank, axis)...)))
    end
    return result
end

"""
    invariance_error(invariant, g; conjugate_axes=())

Maximum absolute change of the dense invariant under `g`. Quark axes are
transformed with `g` and the axes in `conjugate_axes` with `conj(g)`. A meson
invariant `δ_ab` pairs a quark with an antiquark, so it is invariant with
`conjugate_axes=(2,)`.
"""
function invariance_error(
    invariant::ColorInvariant,
    g::AbstractMatrix;
    conjugate_axes=(),
)
    size(g, 1) == invariant.ncolor ||
        throw(DimensionMismatch("g does not act on $(invariant.ncolor) colors"))
    tensor = dense_color_tensor(invariant)
    transformed = transform_color_tensor(tensor, g; conjugate_axes)
    return maximum(abs, transformed .- tensor)
end

function _permutations(items::Vector{Int})
    length(items) <= 1 && return [copy(items)]
    result = Vector{Vector{Int}}()
    for (position, item) in enumerate(items)
        rest = vcat(items[1:(position - 1)], items[(position + 1):end])
        for tail in _permutations(rest)
            push!(result, vcat(item, tail))
        end
    end
    return result
end
