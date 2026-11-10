"""
    luscher_zeta(q2; nmax=7, quadrature_points=4000)

Lüscher's zeta function `Z₀₀(1; q²) = (4π)^(-1/2) Σ_n (n² - q²)^(-1)` for the
cubic box in the rest frame, analytically continued, from the heat-kernel
representation split at `t = 1`:

```text
√(4π) Z₀₀ = Σ_n e^{-(n²-q²)}/(n²-q²) + π^{3/2} ∫₀¹ dt t^{-3/2} (e^{t q²} - 1) - 2π^{3/2}
          + π^{3/2} Σ_{m≠0} ∫₀¹ dt t^{-3/2} e^{t q² - π² m²/t}.
```

The lattice sums run over `|n_i|, |m_i| ≤ nmax`, and the integrals use
Simpson's rule in `u = √t` with `quadrature_points` intervals. Near `q² = 0`,
`√(4π) Z₀₀ ≈ -1/q² - 8.913633`. `q²` must not be the square of an integer
vector.
"""
function luscher_zeta(q2::Real; nmax::Integer=7, quadrature_points::Integer=4000)
    isfinite(q2) || throw(ArgumentError("q2 must be finite"))
    iseven(quadrature_points) || throw(ArgumentError("quadrature_points must be even"))
    q2 = Float64(q2)
    sum_n = 0.0
    sum_m = 0.0
    for nx in -nmax:nmax, ny in -nmax:nmax, nz in -nmax:nmax
        n2 = nx^2 + ny^2 + nz^2
        denominator = n2 - q2
        abs(denominator) > 1e-12 ||
            throw(ArgumentError("q2 = $q2 is on a pole of the zeta function"))
        sum_n += exp(-denominator) / denominator
        n2 == 0 && continue
        # ∫₀¹ dt t^{-3/2} e^{t q² - π² m²/t} = 2 ∫₀¹ du u^{-2} e^{u² q² - π² m²/u²}
        sum_m += _simpson(u -> iszero(u) ? 0.0 : 2 * exp(u^2 * q2 - π^2 * n2 / u^2) / u^2,
            quadrature_points)
    end
    # ∫₀¹ dt t^{-3/2} (e^{t q²} - 1) = 2 ∫₀¹ du (e^{u² q²} - 1)/u², which tends to 2q² at u = 0.
    subtracted = _simpson(u -> iszero(u) ? 2 * q2 : 2 * expm1(u^2 * q2) / u^2, quadrature_points)
    return (sum_n + π^1.5 * (subtracted - 2 + sum_m)) / sqrt(4π)
end

function _simpson(f, intervals)
    h = 1 / intervals
    total = f(0.0) + f(1.0)
    for k in 1:intervals-1
        total += (isodd(k) ? 4 : 2) * f(k * h)
    end
    return total * h / 3
end

"""
    luscher_pcotdelta(p2, L)

S-wave `p cot δ` from Lüscher's formula in a cubic box of side `L`,
`p cot δ = 2 Z₀₀(1; q²) / (√π L)` with `q² = p² L²/(2π)²`. `p2` is the squared
relative momentum, negative below threshold; `p2` and `L` in one unit system.
"""
function luscher_pcotdelta(p2::Real, L::Real)
    L > 0 || throw(ArgumentError("L must be positive"))
    return 2 * luscher_zeta(p2 * L^2 / (2π)^2) / (sqrt(π) * L)
end

"""
    luscher_momentum_squared(energy_shift, mass; dispersion=:continuum)

Squared relative momentum `p²` of two particles of mass `mass` at rest in total
whose energy is `2 mass + energy_shift`. `dispersion=:continuum` uses
`E = 2√(m² + p²)`; `dispersion=:lattice` uses the free lattice boson relation
`cosh(E/2) = cosh m + 2 sin²(p/2)` solved for `p² = (2 asin(√x))²` with
`x = (cosh(E/2) - cosh m)/2`, in lattice units.
"""
function luscher_momentum_squared(energy_shift::Real, mass::Real; dispersion::Symbol=:continuum)
    mass > 0 || throw(ArgumentError("mass must be positive"))
    half = mass + energy_shift / 2
    dispersion == :continuum && return half^2 - mass^2
    dispersion == :lattice || throw(ArgumentError("dispersion must be :continuum or :lattice"))
    x = (cosh(half) - cosh(mass)) / 2
    return x >= 0 ? (2 * asin(sqrt(x)))^2 : -(2 * asinh(sqrt(-x)))^2
end

"""
    luscher_scattering_length(energy_shift, mass, L; dispersion=:continuum)

S-wave scattering length from the two-particle energy shift in a box of side
`L`, at leading order of the effective-range expansion, `a = -1/(p cot δ)`.
The sign convention is that of [`s_wave_scattering_length`](@ref): the
zero-energy wave function behaves as `r - a`, so a repulsive interaction
(positive `energy_shift`, as for `I=2` ππ) gives `a > 0`. For small `a/L` this
reproduces `ΔE = 4πa/(m L³) [1 - c₁ a/L + c₂ (a/L)²]` with `c₁ = -2.837297` and
`c₂ = 6.375183`, with `m` the particle mass (twice the reduced mass).
"""
function luscher_scattering_length(energy_shift::Real, mass::Real, L::Real; dispersion::Symbol=:continuum)
    p2 = luscher_momentum_squared(energy_shift, mass; dispersion)
    return -1 / luscher_pcotdelta(p2, L)
end
