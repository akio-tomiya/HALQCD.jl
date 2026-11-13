using LinearAlgebra: dot, norm
using Random: MersenneTwister

const _CLOVER_PLANES = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))

"""
    clover_projection(group, M)

The part of `M = Q_μν/4` that [`clover_field_strength`](@ref) keeps. For `SU(N)`
it is the anti-Hermitian part `(M - M†)/2` with its trace kept, for every `N`, as
in Bridge++ and LatticeMatrices.jl (`F_μν = (Q_μν - Q_μν†)/8`; checked against
Bridge++ 2.1.3 on an SU(3) configuration to 1e-13). For `SU(2)` the trace vanishes
anyway. Other groups use [`lie_algebra_projection`](@ref); a package that adds a
group whose convention differs defines a method.
"""
clover_projection(group::AbstractGaugeGroup, M::AbstractMatrix) = lie_algebra_projection(group, M)

function clover_projection(::SU{N}, M::AbstractMatrix) where {N}
    size(M) == (N, N) || throw(DimensionMismatch("SU($N) clover field strength needs an $N×$N matrix"))
    return (M - M') / 2
end

"""
    clover_field_strength(group, links, site, μ, ν)

Clover field strength `F_μν(x) = P[Q_μν(x)/4]` at `site`, where `Q_μν` is the
sum of the four plaquettes around `x` in the μν plane, all starting and ending
at `x` and running counterclockwise, and `P` is [`clover_projection`](@ref) for
`group`. For `SU(N)` this is `(Q_μν - Q_μν†)/8` with its trace kept, as in
Bridge++ and LatticeMatrices.jl. LatticeDiracOperators.jl 0.6.3 and its PR #80
remove the trace, which differs for `N ≥ 3`.
"""
function clover_field_strength(group::AbstractGaugeGroup, links, site, μ, ν)
    _, shape = _check_links(links)
    L(direction, x) = _link(links, direction, x)
    x = site
    xpμ, xpν = _shifted(x, μ, 1, shape), _shifted(x, ν, 1, shape)
    xmμ, xmν = _shifted(x, μ, -1, shape), _shifted(x, ν, -1, shape)
    xmμpν, xmμmν = _shifted(xmμ, ν, 1, shape), _shifted(xmμ, ν, -1, shape)
    xpμmν = _shifted(xpμ, ν, -1, shape)
    leaves =
        L(μ, x) * L(ν, xpμ) * L(μ, xpν)' * L(ν, x)' +
        L(ν, x) * L(μ, xmμpν)' * L(ν, xmμ)' * L(μ, xmμ) +
        L(μ, xmμ)' * L(ν, xmμmν)' * L(μ, xmμmν) * L(ν, xmν) +
        L(ν, xmν)' * L(μ, xmν) * L(ν, xpμmν) * L(μ, x)'
    # P removes the Hermitian part, so P[Q/4] = P[(Q - Q†)/8].
    return clover_projection(group, leaves ./ 4)
end

struct _WilsonCloverOperator{G<:AbstractGaugeGroup}
    group::G
    links::Array{ComplexF64,7}
    field_strength::Array{ComplexF64,7}   # (Nc, Nc, plane, x, y, z, t)
    forward::NTuple{4,Matrix{ComplexF64}}  # transpose(r - γ_μ)
    backward::NTuple{4,Matrix{ComplexF64}} # transpose(r + γ_μ)
    sigma::NTuple{6,Matrix{ComplexF64}}    # transpose(γ_μ γ_ν), μ < ν
    gamma5::Matrix{ComplexF64}             # γ_1 γ_2 γ_3 γ_4
    kappa::Float64
    csw::Float64
    boundary_condition::NTuple{4,Int}
end

function _WilsonCloverOperator(group, links; kappa, csw, r, boundary_condition, gammas)
    ncolor, shape = _check_links(links)
    ncolor == representation_dimension(group) ||
        throw(DimensionMismatch("links have $ncolor colors, $(group) needs $(representation_dimension(group))"))
    length(gammas) == 4 && all(gamma -> size(gamma) == (4, 4), gammas) ||
        throw(DimensionMismatch("gammas must be four 4×4 matrices"))
    all(in((1, -1, 0)), boundary_condition) ||
        throw(ArgumentError("boundary_condition entries must be 1, -1 or 0"))
    # A Dirichlet wall (0) cuts the hops only; the clover term is built from the
    # links as given, as for the phase 0 of LatticeMatrices.jl.
    γ = Tuple(Matrix{ComplexF64}(gamma) for gamma in gammas)
    identity4 = Matrix{ComplexF64}(I, 4, 4)
    field_strength = zeros(ComplexF64, ncolor, ncolor, 6, shape...)
    if !iszero(csw)
        sites = vec(collect(CartesianIndices(shape)))
        Threads.@threads for index in sites
            for (plane, (μ, ν)) in enumerate(_CLOVER_PLANES)
                field_strength[:, :, plane, Tuple(index)...] =
                    clover_field_strength(group, links, Tuple(index), μ, ν)
            end
        end
    end
    return _WilsonCloverOperator(
        group, Array{ComplexF64}(links), field_strength,
        Tuple(transpose(r * identity4 - γ[μ]) |> Matrix for μ in 1:4),
        Tuple(transpose(r * identity4 + γ[μ]) |> Matrix for μ in 1:4),
        Tuple(transpose(γ[μ] * γ[ν]) |> Matrix for (μ, ν) in _CLOVER_PLANES),
        γ[1] * γ[2] * γ[3] * γ[4], Float64(kappa), Float64(csw), Tuple(Int.(boundary_condition)),
    )
end

# out[:, :, x] += factor (M ψ[:, :, y]) S, with M = U_μ(linksite) or its adjoint
# and S a 4×4 spin matrix (stored transposed); work is an Nc×4 buffer.
@inline function _add_hop!(out, work, links, μ, linksite, adjoint_link, ψ, y, spin, factor, x)
    ncolor = size(links, 1)
    @inbounds for t in 1:4, c in 1:ncolor
        accumulator = zero(ComplexF64)
        if adjoint_link
            for b in 1:ncolor
                accumulator += conj(links[b, c, μ, linksite...]) * ψ[b, t, y...]
            end
        else
            for b in 1:ncolor
                accumulator += links[c, b, μ, linksite...] * ψ[b, t, y...]
            end
        end
        work[c, t] = accumulator
    end
    _add_spin_product!(out, work, spin, factor, x)
    return out
end

@inline function _add_spin_product!(out, work, spin, factor, x)
    @inbounds for s in 1:4, t in 1:4
        weight = factor * spin[t, s]
        iszero(weight) && continue
        for c in 1:size(work, 1)
            out[c, s, x...] += weight * work[c, t]
        end
    end
    return out
end

function _apply_sites!(out, op, ψ, sites, work)
    ncolor = size(op.links, 1)
    shape = size(ψ)[3:6]
    clover = -op.kappa * op.csw
    for index in sites
        x = Tuple(index)
        @inbounds for s in 1:4, c in 1:ncolor
            out[c, s, x...] = ψ[c, s, x...]
        end
        for μ in 1:4
            forward = _shifted(x, μ, 1, shape)
            backward = _shifted(x, μ, -1, shape)
            forward_phase = x[μ] == shape[μ] ? op.boundary_condition[μ] : 1
            backward_phase = x[μ] == 1 ? op.boundary_condition[μ] : 1
            _add_hop!(out, work, op.links, μ, x, false, ψ, forward, op.forward[μ],
                -op.kappa * forward_phase, x)
            _add_hop!(out, work, op.links, μ, backward, true, ψ, backward, op.backward[μ],
                -op.kappa * backward_phase, x)
        end
        iszero(clover) && continue
        for plane in 1:6
            @inbounds for t in 1:4, c in 1:ncolor
                accumulator = zero(ComplexF64)
                for b in 1:ncolor
                    accumulator += op.field_strength[c, b, plane, x...] * ψ[b, t, x...]
                end
                work[c, t] = accumulator
            end
            _add_spin_product!(out, work, op.sigma[plane], clover, x)
        end
    end
    return out
end

function _apply!(out::Array{ComplexF64,6}, op::_WilsonCloverOperator, ψ::Array{ComplexF64,6})
    sites = vec(collect(CartesianIndices(size(ψ)[3:6])))
    chunks = Iterators.partition(sites, cld(length(sites), Threads.nthreads()))
    ncolor = size(op.links, 1)
    @sync for chunk in chunks
        Threads.@spawn _apply_sites!(out, op, ψ, chunk, zeros(ComplexF64, ncolor, 4))
    end
    return out
end

# BiCGStab for D x = b, started from x = 0. Stops when ‖b - Dx‖ ≤ tolerance ‖b‖
# for the recursive residual; the true residual is returned separately. The
# shadow residual is a fixed random vector: with r̂ = b and a point source the
# even-odd structure of the Wilson operator makes ⟨r̂, D p⟩ vanish.
function _bicgstab(op, b; tolerance, maximum_iterations)
    x = zero(b)
    r = copy(b)
    r̂ = randn(MersenneTwister(20261105), ComplexF64, size(b))
    p, v, s, t = zero(b), zero(b), zero(b), zero(b)
    ρ = α = ω = one(ComplexF64)
    target = tolerance * norm(b)
    iterations = 0
    while iterations < maximum_iterations && norm(r) > target
        iterations += 1
        ρ_new = dot(r̂, r)
        β = (ρ_new / ρ) * (α / ω)
        p .= r .+ β .* (p .- ω .* v)
        _apply!(v, op, p)
        α = ρ_new / dot(r̂, v)
        s .= r .- α .* v
        if norm(s) <= target
            x .+= α .* p
            r .= s
            break
        end
        _apply!(t, op, s)
        ω = dot(t, s) / dot(t, t)
        x .+= α .* p .+ ω .* s
        r .= s .- ω .* t
        ρ = ρ_new
    end
    _apply!(t, op, x)
    true_residual = norm(b .- t) / norm(b)
    return x, iterations, true_residual
end

# out = γ5 ψ in spin space.
function _apply_gamma5!(out, gamma5, ψ)
    ncolor = size(ψ, 1)
    a = reshape(ψ, ncolor, 4, :)
    b = reshape(out, ncolor, 4, :)
    Threads.@threads for site in axes(a, 3)
        @inbounds for s in 1:4, c in 1:ncolor
            accumulator = zero(ComplexF64)
            for t in 1:4
                accumulator += gamma5[s, t] * a[c, t, site]
            end
            b[c, s, site] = accumulator
        end
    end
    return out
end

# D† ψ = γ5 D γ5 ψ, which holds for Hermitian γ matrices.
function _apply_adjoint!(out, op, ψ, scratch)
    _apply_gamma5!(scratch, op.gamma5, ψ)
    _apply!(out, op, scratch)
    _apply_gamma5!(scratch, op.gamma5, out)
    copyto!(out, scratch)
    return out
end

# CG on the normal equations D†D x = D†b (CGNR), started from x = 0. The true
# residual s = b - Dx is updated alongside, and the iteration stops when
# ‖s‖ ≤ tolerance ‖b‖. Works when the spectrum of D surrounds the origin, where
# BiCGStab can stall.
function _cgnr(op, b; tolerance, maximum_iterations)
    x = zero(b)
    s = copy(b)
    r, q, scratch = zero(b), zero(b), zero(b)
    _apply_adjoint!(r, op, s, scratch)
    p = copy(r)
    γ = real(dot(r, r))
    target = tolerance * norm(b)
    iterations = 0
    while iterations < maximum_iterations && norm(s) > target
        iterations += 1
        _apply!(q, op, p)
        α = γ / real(dot(q, q))
        x .+= α .* p
        s .-= α .* q
        _apply_adjoint!(r, op, s, scratch)
        γ_new = real(dot(r, r))
        p .= r .+ (γ_new / γ) .* p
        γ = γ_new
    end
    _apply!(q, op, x)
    true_residual = norm(b .- q) / norm(b)
    return x, iterations, true_residual
end

"""
    wilson_clover_point_propagator(group, links; kappa, csw, r=1,
                                   boundary_condition=(1, 1, 1, -1),
                                   source_position=(1, 1, 1, 1),
                                   gammas=<chiral basis>,
                                   tolerance=1e-12, maximum_iterations=10_000,
                                   solver=:cgnr, source_smearing=nothing)

Point-source propagator of the Wilson-clover operator

```text
D ψ(x) = ψ(x) - κ Σ_μ [(r - γ_μ) U_μ(x) ψ(x+μ̂) + (r + γ_μ) U_μ(x-μ̂)† ψ(x-μ̂)]
         - κ c_SW Σ_{μ<ν} γ_μ γ_ν F_μν(x) ψ(x),
```

with `F_μν` from [`clover_field_strength`](@ref) for `group`, so the gauge
group enters only through [`clover_projection`](@ref). The sign and `F_μν` are
those of Bridge++ and LatticeMatrices.jl (`D_SW = D_W - κ c_SW Σ σ_μν F_μν`,
`σ_μν = [γ_μ, γ_ν]/2`, `F_μν = (Q_μν - Q_μν†)/8`): the pion correlator agrees with
Bridge++ 2.1.3 to 1e-13 on its 4³×8 SU(3) configuration. LatticeDiracOperators.jl
0.6.3 (`"WilsonClover"` on non-MPILattice fields) adds the clover term with the
opposite sign and a traceless `F_μν`, so its `cSW` corresponds to `csw = -cSW`
here only for `SU(2)`; with the sign fix proposed for its PR #80, `csw = cSW`
for `SU(2)`. `boundary_condition` multiplies hops that wrap around the
lattice, as `boundarycondition` does in LatticeDiracOperators.jl: `1` periodic,
`-1` antiperiodic, `0` a Dirichlet wall between the last and the first slice.
The wall cuts the hops only; `F_μν` is built from the links as given, including
the plaquettes across the wall, as with the phase 0 of LatticeMatrices.jl
(1.2.10 and later). `source_smearing=(alpha=α, steps=n)` replaces the point
source by its [`wuppertal_smear`](@ref) with the same links; the sink stays a
point. Each of the `4 Nc` columns is solved to the relative residual
`tolerance` with CG on the normal equations (`solver=:cgnr`, using
`D† = γ₅ D γ₅`, so the γ matrices must be Hermitian) or with BiCGStab
(`solver=:bicgstab`, faster for heavy quarks but it can stall when κ is near or
above the critical value).

Returns `(data, iterations, true_residuals)`; `data` has axes
`(sink_color, sink_spin, source_color, source_spin, x, y, z, t)`.
"""
function wilson_clover_point_propagator(
    group::AbstractGaugeGroup,
    links::AbstractArray{<:Number,7};
    kappa::Real,
    csw::Real,
    r::Real=1,
    boundary_condition::NTuple{4,Integer}=(1, 1, 1, -1),
    source_position::NTuple{4,Integer}=(1, 1, 1, 1),
    gammas=_EUCLIDEAN_GAMMAS,
    tolerance::Real=1e-12,
    maximum_iterations::Integer=10_000,
    solver::Symbol=:cgnr,
    source_smearing::Union{Nothing,NamedTuple}=nothing,
)
    ncolor, shape = _check_links(links)
    all(1 .<= source_position .<= shape) || throw(BoundsError(shape, source_position))
    solver in (:cgnr, :bicgstab) || throw(ArgumentError("solver must be :cgnr or :bicgstab"))
    solver == :cgnr && !all(gamma -> gamma ≈ gamma', gammas) &&
        throw(ArgumentError("solver=:cgnr needs Hermitian γ matrices"))
    op = _WilsonCloverOperator(group, links; kappa, csw, r, boundary_condition, gammas)
    solve = solver == :cgnr ? _cgnr : _bicgstab
    propagator = zeros(ComplexF64, ncolor, 4, ncolor, 4, shape...)
    iterations = zeros(Int, ncolor, 4)
    true_residuals = zeros(ncolor, 4)
    source = zeros(ComplexF64, ncolor, 4, shape...)
    for source_spin in 1:4, source_color in 1:ncolor
        fill!(source, 0)
        source[source_color, source_spin, source_position...] = 1
        isnothing(source_smearing) ||
            (source = wuppertal_smear(links, source; alpha=source_smearing.alpha, steps=source_smearing.steps))
        solution, steps, residual = solve(op, source; tolerance, maximum_iterations)
        residual <= 100 * tolerance ||
            error("Wilson-clover solve did not converge: column ($source_color, $source_spin), " *
                  "true residual $residual after $steps iterations")
        propagator[:, :, source_color, source_spin, ntuple(_ -> Colon(), 4)...] = solution
        iterations[source_color, source_spin] = steps
        true_residuals[source_color, source_spin] = residual
    end
    return (data=propagator, iterations=iterations, true_residuals=true_residuals)
end
