function _check_spatial_field(field::AbstractArray{<:Any,3})
    all(>(0), size(field)) ||
        throw(ArgumentError("spatial lattice extents must be positive"))
    return nothing
end

function _check_spacing(spacing::NTuple{3,<:Real})
    all(s -> isfinite(s) && s > 0, spacing) ||
        throw(ArgumentError("all lattice spacings must be finite and positive"))
    return nothing
end

"""
    laplacian_periodic!(out, field, spacing)

Compute the nearest-neighbor three-dimensional lattice Laplacian with periodic
boundary conditions. Scalar fields use axis order `(x, y, z)`, and `spacing`
is `(a_x, a_y, a_z)`.

`out` must not alias `field`.
"""
function laplacian_periodic!(
    out::AbstractArray{<:Number,3},
    field::AbstractArray{<:Number,3},
    spacing::NTuple{3,<:Real},
)
    size(out) == size(field) ||
        throw(DimensionMismatch("out and field must have identical sizes"))
    Base.mightalias(out, field) &&
        throw(ArgumentError("out must not alias field"))
    _check_spatial_field(field)
    _check_spacing(spacing)

    nx, ny, nz = size(field)
    iax2 = inv(spacing[1]^2)
    iay2 = inv(spacing[2]^2)
    iaz2 = inv(spacing[3]^2)

    @inbounds for z in 1:nz, y in 1:ny, x in 1:nx
        xp, xm = mod1(x + 1, nx), mod1(x - 1, nx)
        yp, ym = mod1(y + 1, ny), mod1(y - 1, ny)
        zp, zm = mod1(z + 1, nz), mod1(z - 1, nz)

        out[x, y, z] =
            (field[xp, y, z] - 2 * field[x, y, z] + field[xm, y, z]) * iax2 +
            (field[x, yp, z] - 2 * field[x, y, z] + field[x, ym, z]) * iay2 +
            (field[x, y, zp] - 2 * field[x, y, z] + field[x, y, zm]) * iaz2
    end
    return out
end

"""
    laplacian_periodic(field, spacing)

Allocating form of [`laplacian_periodic!`](@ref).
"""
function laplacian_periodic(
    field::AbstractArray{T,3},
    spacing::NTuple{3,S},
) where {T<:Number,S<:Real}
    _check_spatial_field(field)
    _check_spacing(spacing)
    Tout = Base.promote_op(/, T, S)
    out = similar(field, Tout)
    return laplacian_periodic!(out, field, spacing)
end

"""
    time_derivatives(rminus, rzero, rplus, at)

Return second-order centered estimates of `∂R/∂t` and `∂²R/∂t²` from three
adjacent Euclidean-time slices separated by `at`.
"""
function time_derivatives(
    rminus::AbstractArray,
    rzero::AbstractArray,
    rplus::AbstractArray,
    at::Real,
)
    size(rminus) == size(rzero) == size(rplus) ||
        throw(DimensionMismatch("all time slices must have identical sizes"))
    isfinite(at) && at > 0 ||
        throw(ArgumentError("at must be finite and positive"))

    dt = (rplus .- rminus) ./ (2 * at)
    d2t = (rplus .- 2 .* rzero .+ rminus) ./ (at^2)
    return dt, d2t
end
