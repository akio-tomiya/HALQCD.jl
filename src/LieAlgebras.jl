using LinearAlgebra: I, det, norm, tr

"""
    lie_algebra_basis(group)

Generators of the Lie algebra of `group` in the quark representation. HALQCD.jl
does not need them for `SU(N)`, where the projection has a closed form; a group
whose projection is built from its generators defines a method.
"""
function lie_algebra_basis end

"""
    lie_algebra_projection(group, M)

Orthogonal projection, in the inner product `tr(A† B)`, of a matrix `M` onto
the Lie algebra of `group` in the quark representation. For `SU(N)` it is the
traceless anti-Hermitian part `(M - M†)/2 - tr(M - M†)/(2N)`.

Stout smearing calls this function, and so does the clover field strength for
groups other than `SU(N)` (see [`clover_projection`](@ref)), so adding a method
for a new group makes both work for it.
"""
function lie_algebra_projection(::SU{N}, M::AbstractMatrix) where {N}
    size(M) == (N, N) || throw(DimensionMismatch("SU($N) projection needs an $N×$N matrix"))
    A = (M - M') / 2
    return A - (tr(A) / N) * I
end

"""
    group_defect(group, g)

Largest deviation of the matrix `g` from the group, used to check links read
from other codes. For `SU(N)` it is the larger of `‖g†g - 1‖` and
`|det g - 1|`.
"""
function group_defect(::SU{N}, g::AbstractMatrix) where {N}
    size(g) == (N, N) || throw(DimensionMismatch("expected an $N×$N matrix"))
    return max(norm(g' * g - I), abs(det(g) - 1))
end
