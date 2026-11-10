using LinearAlgebra: I, tr

# Gauge links are plain arrays links[a, b, μ, x, y, z, t]: U_μ(x) is the link
# from x to x + μ̂. The JuliaQCD extension
# converts Gaugefields.jl fields to this layout.

function _check_links(links::AbstractArray{<:Number,7})
    ncolor = size(links, 1)
    size(links, 2) == ncolor && size(links, 3) == 4 ||
        throw(DimensionMismatch("links must have axes (Nc, Nc, 4, x, y, z, t)"))
    return ncolor, size(links)[4:7]
end

_link(links, μ, site) = links[:, :, μ, site...]
_shifted(site, μ, step, shape) =
    ntuple(ν -> ν == μ ? mod1(site[ν] + step, shape[ν]) : site[ν], 4)

"""
    average_plaquette(links)

Average of `Re tr U_μν(x) / Nc` over all sites and the six planes, with
`U_μν(x) = U_μ(x) U_ν(x+μ̂) U_μ(x+ν̂)† U_ν(x)†`. This is the normalization of
the Gaugefields.jl plaquette.
"""
function average_plaquette(links::AbstractArray{<:Number,7})
    ncolor, shape = _check_links(links)
    total = 0.0
    for index in CartesianIndices(shape), μ in 1:4, ν in μ+1:4
        x = Tuple(index)
        total += real(tr(
            _link(links, μ, x) * _link(links, ν, _shifted(x, μ, 1, shape)) *
            _link(links, μ, _shifted(x, ν, 1, shape))' * _link(links, ν, x)',
        ))
    end
    return total / (ncolor * 6 * prod(shape))
end

# Ω_μ(x) = Σ_{ν≠μ} [U_ν(x) U_μ(x+ν̂) U_ν(x+μ̂)† + U_ν(x-ν̂)† U_μ(x-ν̂) U_ν(x-ν̂+μ̂)] U_μ(x)†
function _stout_omega(links, μ, x, shape)
    staple = zero(_link(links, μ, x))
    xpμ = _shifted(x, μ, 1, shape)
    for ν in 1:4
        ν == μ && continue
        xpν, xmν = _shifted(x, ν, 1, shape), _shifted(x, ν, -1, shape)
        staple .+= _link(links, ν, x) * _link(links, μ, xpν) * _link(links, ν, xpμ)'
        staple .+= _link(links, ν, xmν)' * _link(links, μ, xmν) *
            _link(links, ν, _shifted(xmν, μ, 1, shape))
    end
    return staple * _link(links, μ, x)'
end

"""
    stout_smear(group, links; rho, levels=1, smear_directions=(true, true, true, true))

Isotropic stout smearing (Morningstar and Peardon) of the links, `levels`
times. One level updates every link at once from the previous level,

```text
U'_μ(x) = exp(ρ P[Ω_μ(x)]) U_μ(x),
Ω_μ(x)  = Σ_{ν≠μ} (staples from x to x+μ̂) U_μ(x)†,
```

with `P` = [`lie_algebra_projection`](@ref) for `group`. For `SU(N)` this is
`exp(iQ)` of Morningstar and Peardon.
Directions with `smear_directions[μ] == false` are copied, but their links
still enter the staples of the other directions.
"""
function stout_smear(
    group::AbstractGaugeGroup,
    links::AbstractArray{<:Number,7};
    rho::Real,
    levels::Integer=1,
    smear_directions::NTuple{4,Bool}=(true, true, true, true),
)
    ncolor, shape = _check_links(links)
    ncolor == representation_dimension(group) ||
        throw(DimensionMismatch("links have $ncolor colors, $(group) needs $(representation_dimension(group))"))
    levels >= 0 || throw(ArgumentError("levels must be nonnegative"))
    current = Array{ComplexF64}(links)
    next = similar(current)
    sites = vec(collect(CartesianIndices(shape)))
    for _ in 1:levels
        Threads.@threads for index in sites
            x = Tuple(index)
            for μ in 1:4
                if smear_directions[μ] && !iszero(rho)
                    generator = rho .* lie_algebra_projection(group, _stout_omega(current, μ, x, shape))
                    next[:, :, μ, x...] = exp(generator) * _link(current, μ, x)
                else
                    next[:, :, μ, x...] = _link(current, μ, x)
                end
            end
        end
        current, next = next, current
    end
    return current
end

"""
    coulomb_gauge_residual(links)

Largest and mean violation of the Coulomb gauge condition, a check of links
fixed with `Gaugefields.gaugefixing!(…; D_fix=3)` before a wall source is used.
At each site `Δ(x) = Σ_{k=1}^{3} [U_k(x) - U_k(x - k̂)]`, and its traceless
anti-Hermitian part `A(x)` vanishes in Coulomb gauge. Returns
`(maximum, mean)` of `tr(A A†) / Nc` over the sites. The normalization is
HALQCD's own and is not the residual printed by Gaugefields.jl. For `SU(N)`
links only.
"""
function coulomb_gauge_residual(links::AbstractArray{<:Number,7})
    ncolor, shape = _check_links(links)
    worst = 0.0
    total = 0.0
    identity = Matrix{ComplexF64}(I, ncolor, ncolor)
    for index in CartesianIndices(shape)
        x = Tuple(index)
        Δ = zeros(ComplexF64, ncolor, ncolor)
        for k in 1:3
            Δ .+= _link(links, k, x) .- _link(links, k, _shifted(x, k, -1, shape))
        end
        A = (Δ .- Δ') ./ 2
        A .-= tr(A) / ncolor .* identity
        value = real(tr(A * A')) / ncolor
        worst = max(worst, value)
        total += value
    end
    return (maximum=worst, mean=total / prod(shape))
end
