const _AXIS_PERMUTATIONS = (
    (1, 2, 3),
    (1, 3, 2),
    (2, 1, 3),
    (2, 3, 1),
    (3, 1, 2),
    (3, 2, 1),
)

"""
    a1plus_project!(out, field)

Project a cubic three-dimensional field onto the fully symmetric, even-parity
`A1+` representation. The projector averages the 48 signed permutations of
the spatial axes about the periodic origin at index `(1, 1, 1)`.

This is a cubic-group projection, not a spherical-shell average. It is a
`B=0` operation and should not be applied when a background magnetic field
has reduced the spatial symmetry.
"""
function a1plus_project!(
    out::AbstractArray{TO,3},
    field::AbstractArray{TI,3},
) where {TO,TI}
    size(out) == size(field) ||
        throw(DimensionMismatch("out and field must have identical sizes"))
    size(field, 1) == size(field, 2) == size(field, 3) ||
        throw(DimensionMismatch("A1+ projection requires a cubic lattice"))

    n = size(field, 1)
    source = Base.mightalias(out, field) ? copy(field) : field

    @inbounds for z in axes(out, 3), y in axes(out, 2), x in axes(out, 1)
        coordinates = (x - 1, y - 1, z - 1)
        value = zero(TO)
        for permutation in _AXIS_PERMUTATIONS
            c1 = coordinates[permutation[1]]
            c2 = coordinates[permutation[2]]
            c3 = coordinates[permutation[3]]
            for s3 in (-1, 1), s2 in (-1, 1), s1 in (-1, 1)
                i = mod(s1 * c1, n) + 1
                j = mod(s2 * c2, n) + 1
                k = mod(s3 * c3, n) + 1
                value += source[i, j, k]
            end
        end
        out[x, y, z] = value / 48
    end
    return out
end

"""
    a1plus_project(field)

Allocating form of [`a1plus_project!`](@ref).
"""
function a1plus_project(field::AbstractArray{T,3}) where {T}
    Tout = Base.promote_op(/, T, Int)
    out = similar(field, Tout)
    return a1plus_project!(out, field)
end

@inline function _periodic_coordinate(index::Integer, extent::Integer)
    coordinate = index - 1
    return coordinate <= extent ÷ 2 ? coordinate : coordinate - extent
end

"""
    shell_average(field; spacing=1, valid=nothing)

Average a three-dimensional field over spherical lattice shells. Sites are
grouped by the integer squared radius
`r2 = x_periodic^2 + y_periodic^2 + z_periodic^2`; `spacing` only converts the
reported radius to physical units.

The return value is a vector of named tuples `(r2, radius, mean, count)`, sorted
by `r2`. `valid` may be a Boolean mask with the same shape as `field`.

This operation is deliberately separate from [`a1plus_project`](@ref).
Neither operation is appropriate at nonzero background magnetic field unless
the reduced symmetry is handled explicitly.
"""
function shell_average(
    field::AbstractArray{T,3};
    spacing::Real=1,
    valid::Union{Nothing,AbstractArray{Bool,3}}=nothing,
) where {T}
    isfinite(spacing) && spacing > 0 ||
        throw(ArgumentError("spacing must be finite and positive"))
    if valid !== nothing
        size(valid) == size(field) ||
            throw(DimensionMismatch("valid and field must have identical sizes"))
    end

    Tmean = Base.promote_op(/, T, Int)
    sums = Dict{Int,Tmean}()
    counts = Dict{Int,Int}()

    @inbounds for z in axes(field, 3), y in axes(field, 2), x in axes(field, 1)
        valid !== nothing && !valid[x, y, z] && continue
        cx = _periodic_coordinate(x, size(field, 1))
        cy = _periodic_coordinate(y, size(field, 2))
        cz = _periodic_coordinate(z, size(field, 3))
        r2 = cx^2 + cy^2 + cz^2
        sums[r2] = get(sums, r2, zero(Tmean)) + field[x, y, z]
        counts[r2] = get(counts, r2, 0) + 1
    end

    return [
        (
            r2=r2,
            radius=spacing * sqrt(r2),
            mean=sums[r2] / counts[r2],
            count=counts[r2],
        )
        for r2 in sort!(collect(keys(sums)))
    ]
end
