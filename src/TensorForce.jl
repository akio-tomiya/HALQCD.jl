# Leading-order central and tensor potentials of the J^P = 1⁺ two-nucleon
# channel, and the next-to-next-to-leading-order central potential from two
# sources.

const _PAULI = (
    ComplexF64[0 1; 1 0],
    ComplexF64[0 -im; im 0],
    ComplexF64[1 0; 0 -1],
)

# Displacement of array index i on a periodic axis of length n, taken to the
# nearest periodic image. The second value is true on the plane |d| = n/2,
# where the direction of r is ambiguous.
@inline function _nearest_image(i, n)
    d = i - 1
    d > n ÷ 2 && (d -= n)
    return d, iseven(n) && abs(d) == n ÷ 2
end

"""
    tensor_operator_s12(psi)

`S₁₂ ψ` for a two-nucleon wave function `psi` with axes `(α, β, x, y, z)`,
where `α` and `β` are the spins of the nucleons at `x + r` and `x` in the basis
`(↑, ↓)`:

```text
S₁₂ = 3 (σ₁·r̂)(σ₂·r̂) - σ₁·σ₂,   [σ₂ ψ]_{αβ} = Σ_β′ σ_{ββ′} ψ_{αβ′}.
```

`r` is the nearest periodic image of the array displacement. `S₁₂` is set to
zero at `r = 0`.
"""
function tensor_operator_s12(psi::AbstractArray{<:Number,5})
    size(psi, 1) == size(psi, 2) == 2 ||
        throw(DimensionMismatch("psi must have axes (2, 2, x, y, z)"))
    nx, ny, nz = size(psi)[3:5]
    out = zeros(ComplexF64, size(psi))
    for z in 1:nz, y in 1:ny, x in 1:nx
        r = (_nearest_image(x, nx)[1], _nearest_image(y, ny)[1], _nearest_image(z, nz)[1])
        all(iszero, r) && continue
        n̂ = r ./ sqrt(sum(abs2, r))
        σn = sum(n̂[k] * _PAULI[k] for k in 1:3)
        ψ = psi[:, :, x, y, z]
        value = 3 * σn * ψ * transpose(σn)
        for k in 1:3
            value -= _PAULI[k] * ψ * transpose(_PAULI[k])
        end
        out[:, :, x, y, z] .= value
    end
    return out
end

function _parity_even(field::AbstractArray{<:Number,3})
    nx, ny, nz = size(field)
    out = similar(field, promote_type(eltype(field), Float64))
    for z in 1:nz, y in 1:ny, x in 1:nx
        out[x, y, z] = (field[x, y, z] + field[mod(1 - x, nx) + 1, mod(1 - y, ny) + 1, mod(1 - z, nz) + 1]) / 2
    end
    return out
end

"""
    lo_central_tensor_potential(rminus, rzero, rplus; mass,
                                spacing=(1, 1, 1), at=1, component=(2, 1),
                                rtol=sqrt(eps()), atol=0)

Leading-order central and tensor potentials of the `J^P = 1⁺` channel from
the time-dependent HAL QCD equation

```text
K R = [V_C(r) + V_T(r) S₁₂] R,   K = (1/(4m))∂ₜ² - ∂ₜ + ∇²/m.
```

`rminus`, `rzero` and `rplus` are the spin-resolved `R` correlators at three
consecutive times, with axes `(α, β, x, y, z)` (sink spins of the nucleons at
`x + r` and `x`) for a source in the `J = 1` state. The equation is split by
`P`, the `A₁⁺` projection acting on `r` only, and `Q = P_even - P`, where
`P_even` keeps the part even under `r → -r`. The `component` `(α, β)` of the
two projected equations gives

```text
V_C = ([P K R] [Q S₁₂R] - [Q K R] [P S₁₂R]) / Δ,
V_T = ([P R] [Q K R] - [Q R] [P K R]) / Δ,
Δ   = [P R] [Q S₁₂R] - [Q R] [P S₁₂R],
```

the time-dependent form of Aoki, Hatsuda and Ishii, Prog. Theor. Phys. 123
(2010) 89, arXiv:0909.5585, Eqs. (5.16)-(5.18), which use the component
`(2, 1)` and a source with `J_z = 0`. A site is valid when `|Δ|` exceeds
`max(atol, rtol max|Δ|)` and `r` is not on a plane `|r_i| = L/2`, where the
direction of `r` is ambiguous (Murano et al., arXiv:1305.2293). The `A₁⁺`
part also contains `ℓ ≥ 4` and `Q` contains `ℓ = 2, 4, …`.

Returns a named tuple `(central, tensor, valid, determinant)`.
"""
function lo_central_tensor_potential(
    rminus::AbstractArray{<:Number,5},
    rzero::AbstractArray{<:Number,5},
    rplus::AbstractArray{<:Number,5};
    mass::Real,
    spacing::NTuple{3,<:Real}=(1.0, 1.0, 1.0),
    at::Real=1.0,
    component::NTuple{2,Int}=(2, 1),
    rtol::Real=sqrt(eps(Float64)),
    atol::Real=0,
)
    isfinite(mass) && mass > 0 || throw(ArgumentError("mass must be finite and positive"))
    size(rminus) == size(rzero) == size(rplus) ||
        throw(DimensionMismatch("all time slices must have identical sizes"))
    size(rzero, 1) == size(rzero, 2) == 2 ||
        throw(DimensionMismatch("R must have axes (2, 2, x, y, z)"))
    all(in(1:2), component) || throw(ArgumentError("component entries must be 1 or 2"))
    nx, ny, nz = size(rzero)[3:5]
    α, β = component

    # K acts on each spin component and P, Q act on r only, so only the
    # chosen component of K R is needed; S₁₂ mixes all four.
    R = rzero[α, β, :, :, :]
    dt, d2t = time_derivatives(rminus[α, β, :, :, :], R, rplus[α, β, :, :, :], at)
    kinetic = laplacian_periodic(R, spacing) ./ mass .- dt .+ d2t ./ (4 * mass)
    s12 = tensor_operator_s12(rzero)
    project_p(field) = a1plus_project(field)
    project_q(field) = _parity_even(field) .- a1plus_project(field)
    SR = s12[α, β, :, :, :]
    pR, qR = project_p(R), project_q(R)
    pS, qS = project_p(SR), project_q(SR)
    pK, qK = project_p(kinetic), project_q(kinetic)

    determinant = pR .* qS .- qR .* pS
    scale = maximum(abs, determinant)
    cutoff = max(atol, rtol * scale)
    central = zeros(ComplexF64, nx, ny, nz)
    tensor = zeros(ComplexF64, nx, ny, nz)
    valid = falses(nx, ny, nz)
    for z in 1:nz, y in 1:ny, x in 1:nx
        on_boundary = _nearest_image(x, nx)[2] || _nearest_image(y, ny)[2] || _nearest_image(z, nz)[2]
        Δ = determinant[x, y, z]
        (on_boundary || !isfinite(abs(Δ)) || abs(Δ) <= cutoff) && continue
        central[x, y, z] = (pK[x, y, z] * qS[x, y, z] - qK[x, y, z] * pS[x, y, z]) / Δ
        tensor[x, y, z] = (pR[x, y, z] * qK[x, y, z] - qR[x, y, z] * pK[x, y, z]) / Δ
        valid[x, y, z] = true
    end
    return (central=central, tensor=tensor, valid=valid, determinant=determinant)
end

"""
    n2lo_potential_two_sources(rs_a, rs_b; mass, spacing=(1, 1, 1), at=1,
                               rtol=sqrt(eps()), atol=0)

Central potential to next-to-next-to-leading order of the derivative
expansion, `U = (V₀ + V₂ ∇²) δ`, from the `R` correlators of two sources,
following Iritani et al., Phys. Rev. D 99 (2019) 014514, arXiv:1805.02365,
Eqs. (11) and (15). `rs_a` and `rs_b` are tuples `(rminus, rzero, rplus)` of
three consecutive times of each source. With `V^LO` from
[`lo_potential_equal_mass`](@ref) and `L = ∇²R/R`,

```text
V^LO = V₀ + V₂ L,   V₂ = (V^LO_a - V^LO_b) / (L_a - L_b),   V₀ = V^LO_a - V₂ L_a.
```

The two sources must give different `L`; sites with
`|L_a - L_b| ≤ max(atol, rtol max|L_a - L_b|)` are invalid. Iritani et al.
fit `V₂(r)` with a smooth function before forming `V₀`, because `L_a - L_b`
crosses zero; this function returns the pointwise values for that fit.

Returns `(V0, V2, difference, valid, lo_a, lo_b)`.
"""
function n2lo_potential_two_sources(
    rs_a::NTuple{3,AbstractArray{<:Number,3}},
    rs_b::NTuple{3,AbstractArray{<:Number,3}};
    mass::Real,
    spacing::NTuple{3,<:Real}=(1.0, 1.0, 1.0),
    at::Real=1.0,
    rtol::Real=sqrt(eps(Float64)),
    atol::Real=0,
)
    lo_a = lo_potential_equal_mass(rs_a...; mass, spacing, at)
    lo_b = lo_potential_equal_mass(rs_b...; mass, spacing, at)
    ratio_a = laplacian_periodic(rs_a[2], spacing) ./ rs_a[2]
    ratio_b = laplacian_periodic(rs_b[2], spacing) ./ rs_b[2]
    difference = ratio_a .- ratio_b
    finite = isfinite.(difference) .& lo_a.valid .& lo_b.valid
    scale = any(finite) ? maximum(abs, difference[finite]) : 0.0
    cutoff = max(atol, rtol * scale)
    valid = finite .& (abs.(difference) .> cutoff)
    V2 = zeros(promote_type(eltype(difference), eltype(lo_a.potential)), size(difference))
    V0 = similar(V2)
    fill!(V0, 0)
    V2[valid] .= (lo_a.potential[valid] .- lo_b.potential[valid]) ./ difference[valid]
    V0[valid] .= lo_a.potential[valid] .- V2[valid] .* ratio_a[valid]
    return (V0=V0, V2=V2, difference=difference, valid=valid, lo_a=lo_a, lo_b=lo_b)
end
