"""
    wuppertal_smear(links, field; alpha, steps)

Gauge-covariant Gaussian (Wuppertal) smearing in the three spatial
directions, applied `steps` times on every time slice:

```text
ψ(x) ← [ψ(x) + α Σ_{j=1}^{3} (U_j(x) ψ(x+ĵ) + U_j(x-ĵ)† ψ(x-ĵ))] / (1 + 6α).
```

`links` has the layout of [`juliaqcd_links`](@ref), axes `(Nc, Nc, μ, x, y, z, t)`,
and `field` the layout of the sources of [`wilson_clover_point_propagator`](@ref),
axes `(Nc, 4, x, y, z, t)`; spin is untouched and space is periodic. Under a
local gauge transformation `U_μ(x) → Ω(x) U_μ(x) Ω(x+μ̂)†` the result
transforms like the input, `ψ(x) → Ω(x) ψ(x)`. For a free field one step
spreads a point by `⟨x_j²⟩ = 2α/(1 + 6α)` in each direction. JuliaQCD
(Gaugefields.jl) smears links but has no quark-source smearing, so the source
is smeared here.
"""
function wuppertal_smear(links::AbstractArray{<:Number,7}, field::AbstractArray{<:Number,6};
        alpha::Real, steps::Integer)
    ncolor, shape = _check_links(links)
    size(field) == (ncolor, 4, shape...) ||
        throw(DimensionMismatch("field must have axes (Nc, 4, x, y, z, t) matching the links"))
    alpha >= 0 && steps >= 0 || throw(ArgumentError("alpha and steps must be nonnegative"))
    current = Array{ComplexF64}(field)
    next = copy(current)
    scale = 1 / (1 + 6alpha)
    # Slices are smeared independently, so slices that are zero stay zero.
    slices = [t for t in 1:shape[4] if any(!iszero, view(current, :, :, :, :, :, t))]
    for _ in 1:steps
        _wuppertal_step!(next, current, links, ncolor, shape, slices, alpha, scale)
        current, next = next, current
    end
    return current
end

# One smearing step, current → next. A separate function, so that the threaded loop
# does not capture the swapped arrays (captured variables that are reassigned are
# boxed and every access becomes dynamic).
function _wuppertal_step!(next, current, links, ncolor, shape, slices, alpha, scale)
    spatial = CartesianIndices((shape[1], shape[2], shape[3]))
    nspatial = length(spatial)
    Threads.@threads for k in 1:(nspatial*length(slices))
        index = spatial[mod1(k, nspatial)]
        x = (Tuple(index)..., slices[cld(k, nspatial)])
        @inbounds for s in 1:4, c in 1:ncolor
            accumulator = zero(ComplexF64)
            for j in 1:3
                forward = _shifted(x, j, 1, shape)
                backward = _shifted(x, j, -1, shape)
                for b in 1:ncolor
                    accumulator += links[c, b, j, x...] * current[b, s, forward...] +
                                   conj(links[b, c, j, backward...]) * current[b, s, backward...]
                end
            end
            next[c, s, x...] = scale * (current[c, s, x...] + alpha * accumulator)
        end
    end
    return next
end
