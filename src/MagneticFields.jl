"""
    flux_qBa2(nflux, nx, ny; charge_ratio=1)

Return the dimensionless flux `q B a²`. `nflux` quantizes the field for the
smallest charge, and integer `charge_ratio = q/q_min` selects the particle
charge:

`q B a² = 2π charge_ratio nflux / (nx ny)`.
"""
function flux_qBa2(
    nflux::Integer,
    nx::Integer,
    ny::Integer;
    charge_ratio::Integer=1,
)
    nx > 0 && ny > 0 || throw(ArgumentError("nx and ny must be positive"))
    return 2 * pi * charge_ratio * nflux / (nx * ny)
end

"""
    flux_eBa2(nflux, nx, ny; qmin_over_e=1/3)

Return `e B a²` when flux quantization is imposed with smallest charge
`q_min = qmin_over_e * e`. For QCD quarks, the default gives
`e B a² = 6π nflux/(nx ny)`.
"""
function flux_eBa2(
    nflux::Integer,
    nx::Integer,
    ny::Integer;
    qmin_over_e::Real=1 / 3,
)
    isfinite(qmin_over_e) && qmin_over_e > 0 ||
        throw(ArgumentError("qmin_over_e must be finite and positive"))
    return flux_qBa2(nflux, nx, ny) / qmin_over_e
end

"""
    uniform_u1_links((nx, ny, nz, nt), nflux;
                     charge_ratio=1, float_type=Float64)

Construct periodic U(1) links for a uniform magnetic field in the `+z`
direction using lattice Landau gauge. The array order is `(mu, x, y, z, t)`,
where `mu = 1,2,3,4` corresponds to `x,y,z,t`.

`nflux` is an integer flux quantum for the smallest charge. Internally,
`x₀=x-1` and `y₀=y-1`; the compensating boundary link is placed at `x=nx`.
"""
function uniform_u1_links(
    dims::NTuple{4,Int},
    nflux::Integer;
    charge_ratio::Integer=1,
    float_type::Type{T}=Float64,
) where {T<:AbstractFloat}
    nx, ny, nz, nt = dims
    all(>(0), dims) || throw(ArgumentError("all lattice extents must be positive"))

    qBa2 = T(2) * T(pi) * T(charge_ratio) * T(nflux) / T(nx * ny)
    links = ones(Complex{T}, 4, nx, ny, nz, nt)
    @inbounds for t in 1:nt, z in 1:nz, y in 1:ny, x in 1:nx
        x0 = x - 1
        y0 = y - 1
        links[2, x, y, z, t] = cis(qBa2 * x0)
        if x == nx
            links[1, x, y, z, t] = cis(-qBa2 * nx * y0)
        end
    end
    return links
end

"""
    u1_plaquette_xy(links, x, y, z, t)

Return the oriented U(1) plaquette
`u_x(x) u_y(x+x̂) u_x(x+ŷ)† u_y(x)†`.
"""
function u1_plaquette_xy(
    links::AbstractArray{<:Number,5},
    x::Integer,
    y::Integer,
    z::Integer,
    t::Integer,
)
    size(links, 1) == 4 ||
        throw(DimensionMismatch("first dimension must be mu=1:4"))
    nx, ny = size(links, 2), size(links, 3)
    xp, yp = mod1(x + 1, nx), mod1(y + 1, ny)
    return links[1, x, y, z, t] *
           links[2, xp, y, z, t] *
           conj(links[1, x, yp, z, t]) *
           conj(links[2, x, y, z, t])
end

"""
    apply_u1_links!(out, color_links, u1links)

Multiply color links by U(1) phases. Color links use
`(color_out, color_in, mu, x, y, z, t)` and U(1) links use
`(mu, x, y, z, t)`.

`out === color_links` is supported for an explicitly destructive update.
"""
function apply_u1_links!(
    out::AbstractArray{<:Number,7},
    color_links::AbstractArray{<:Number,7},
    u1links::AbstractArray{<:Number,5},
)
    size(out) == size(color_links) ||
        throw(DimensionMismatch("out and color_links must have identical sizes"))
    size(color_links, 1) == size(color_links, 2) ||
        throw(DimensionMismatch("color dimensions must be square"))
    size(color_links, 3) == 4 ||
        throw(DimensionMismatch("third dimension of color_links must be mu=1:4"))
    (
        size(color_links, 3),
        size(color_links, 4),
        size(color_links, 5),
        size(color_links, 6),
        size(color_links, 7),
    ) == size(u1links) ||
        throw(DimensionMismatch("u1links dimensions do not match color_links"))

    nc = size(color_links, 1)
    _, nx, ny, nz, nt = size(u1links)
    @inbounds for t in 1:nt, z in 1:nz, y in 1:ny, x in 1:nx,
                  mu in 1:4, j in 1:nc, i in 1:nc
        out[i, j, mu, x, y, z, t] =
            u1links[mu, x, y, z, t] * color_links[i, j, mu, x, y, z, t]
    end
    return out
end

"""
    apply_u1_links(color_links, u1links)

Allocating form of [`apply_u1_links!`](@ref). The input color links are not
modified.

These array functions do not check the gauge group. Use
[`apply_u1_phase`](@ref) when the group is known: for a real representation
the product with a complex phase leaves the group.
"""
function apply_u1_links(
    color_links::AbstractArray{TS,7},
    u1links::AbstractArray{TU,5},
) where {TS<:Number,TU<:Number}
    Tout = promote_type(TS, TU)
    out = similar(color_links, Tout)
    return apply_u1_links!(out, color_links, u1links)
end

"""
    covariant_laplacian_periodic!(out, field, links; time=1)

Apply the nearest-neighbor, gauge-covariant spatial Laplacian

`D²ψ(x) = Σᵢ[Uᵢ(x)ψ(x+i) + Uᵢ(x-i)†ψ(x-i) - 2ψ(x)]`

for `i = x,y,z`. `field` and `out` use `(x,y,z)` order; `links` use
`(mu,x,y,z,t)`. All boundaries are periodic.

Under `ψ(x) → g(x)ψ(x)` and
`Uᵢ(x) → g(x)Uᵢ(x)g(x+i)†`, the result transforms as `D²ψ → g D²ψ`.
"""
function covariant_laplacian_periodic!(
    out::AbstractArray{TO,3},
    field::AbstractArray{TF,3},
    links::AbstractArray{TL,5};
    time::Integer=1,
) where {TO,TF,TL}
    size(out) == size(field) ||
        throw(DimensionMismatch("out and field must have identical sizes"))
    size(links, 1) == 4 ||
        throw(DimensionMismatch("first link dimension must be mu=1:4"))
    size(links)[2:4] == size(field) ||
        throw(DimensionMismatch("spatial link dimensions must match field"))
    1 <= time <= size(links, 5) ||
        throw(BoundsError(axes(links, 5), time))

    source = Base.mightalias(out, field) ? copy(field) : field
    nx, ny, nz = size(field)
    @inbounds for z in 1:nz, y in 1:ny, x in 1:nx
        xp, xm = mod1(x + 1, nx), mod1(x - 1, nx)
        yp, ym = mod1(y + 1, ny), mod1(y - 1, ny)
        zp, zm = mod1(z + 1, nz), mod1(z - 1, nz)
        out[x, y, z] =
            links[1, x, y, z, time] * source[xp, y, z] +
            conj(links[1, xm, y, z, time]) * source[xm, y, z] +
            links[2, x, y, z, time] * source[x, yp, z] +
            conj(links[2, x, ym, z, time]) * source[x, ym, z] +
            links[3, x, y, z, time] * source[x, y, zp] +
            conj(links[3, x, y, zm, time]) * source[x, y, zm] -
            6 * source[x, y, z]
    end
    return out
end

"""
    covariant_laplacian_periodic(field, links; time=1)

Allocating form of [`covariant_laplacian_periodic!`](@ref).
"""
function covariant_laplacian_periodic(
    field::AbstractArray{TF,3},
    links::AbstractArray{TL,5};
    time::Integer=1,
) where {TF,TL}
    Tout = promote_type(TF, TL)
    out = similar(field, Tout)
    return covariant_laplacian_periodic!(out, field, links; time=time)
end

"""
    apply_u1_phase(group, color_links, u1links)

Group-aware form of [`apply_u1_links`](@ref). For `SU(N)` the product of an
`SU(N)` link and a U(1) phase is a `U(N)` link and is returned as a complex
array.

A group with a real quark representation should define a method that throws:
multiplying a real link by a complex phase gives a link outside the group.
"""
function apply_u1_phase(
    group::SU,
    color_links::AbstractArray{<:Number,7},
    u1links::AbstractArray{<:Number,5},
)
    size(color_links, 1) == representation_dimension(group) ||
        throw(DimensionMismatch("color links do not match $(group)"))
    return apply_u1_links(color_links, u1links)
end
