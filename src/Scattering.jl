"""
    ScatteringLengthResult

Result of a zero-energy radial Schrödinger integration.

- `value`: S-wave scattering length inferred in the asymptotic tail;
- `tail_spread`: maximum minus minimum of the pointwise tail estimates;
- `wavefunction`: regular reduced radial wavefunction, with arbitrary scale;
- `tail_indices`: grid indices used for the asymptotic estimate.
"""
struct ScatteringLengthResult{T,V,I}
    value::T
    tail_spread::T
    wavefunction::V
    tail_indices::I
end

"""
    s_wave_scattering_length(radii, potential; reduced_mass, tail_start)

Integrate the zero-energy S-wave radial Schrödinger equation

`u''(r) = 2 reduced_mass * potential(r) * u(r)`

with the Numerov method and regular boundary condition `u(0)=0`. `radii` must
be a uniformly spaced real grid beginning at zero, and `potential` must be
real and finite on that grid.

Where the potential has vanished, `u(r)` is proportional to
`r - scattering_length`. The returned value is the mean of
`r - u(r)/u'(r)` over centered-difference points with `r >= tail_start`;
`tail_spread` measures residual drift over that interval.

All inputs must use one consistent unit system. In particular,
`2 * reduced_mass * potential` must have units of inverse length squared.
"""
function s_wave_scattering_length(
    radii::AbstractVector{TR},
    potential::AbstractVector{TV};
    reduced_mass::Real,
    tail_start::Real,
) where {TR<:Real,TV<:Real}
    length(radii) == length(potential) ||
        throw(DimensionMismatch("radii and potential must have equal length"))
    length(radii) >= 5 ||
        throw(ArgumentError("at least five radial grid points are required"))
    isfinite(reduced_mass) && reduced_mass > 0 ||
        throw(ArgumentError("reduced_mass must be finite and positive"))
    all(isfinite, radii) ||
        throw(ArgumentError("radii must be finite"))
    all(isfinite, potential) ||
        throw(ArgumentError("potential must be finite"))
    iszero(first(radii)) ||
        throw(ArgumentError("the radial grid must begin at zero"))

    h = radii[2] - radii[1]
    isfinite(h) && h > 0 ||
        throw(ArgumentError("radial grid spacing must be finite and positive"))
    spacing_scale = max(one(float(h)), float(maximum(abs, radii)))
    spacing_tolerance = 128eps(spacing_scale)
    all(
        abs((radii[index] - radii[index - 1]) - h) <=
        spacing_tolerance
        for index in 3:length(radii)
    ) || throw(ArgumentError("radii must be uniformly spaced"))

    first_tail = searchsortedfirst(radii, tail_start)
    first_tail = max(first_tail, 2)
    last_tail = length(radii) - 1
    first_tail <= last_tail ||
        throw(ArgumentError("tail_start leaves no centered-difference tail points"))

    T = promote_type(float(TR), float(TV), typeof(float(reduced_mass)))
    wavefunction = zeros(T, length(radii))
    wavefunction[2] = T(h)
    coefficient = T(2 * reduced_mass) .* potential
    h2_over_12 = T(h^2 / 12)

    for index in 2:(length(radii) - 1)
        denominator = one(T) - h2_over_12 * coefficient[index + 1]
        abs(denominator) > 64eps(T) ||
            throw(ArgumentError("Numerov denominator is numerically singular"))
        wavefunction[index + 1] = (
            2wavefunction[index] *
            (one(T) + 5h2_over_12 * coefficient[index]) -
            wavefunction[index - 1] *
            (one(T) - h2_over_12 * coefficient[index - 1])
        ) / denominator

        isfinite(wavefunction[index + 1]) ||
            throw(ArgumentError("radial integration produced a non-finite value"))
    end

    tail_indices = first_tail:last_tail
    estimates = Vector{T}(undef, length(tail_indices))
    for (output_index, radial_index) in enumerate(tail_indices)
        derivative = (
            wavefunction[radial_index + 1] -
            wavefunction[radial_index - 1]
        ) / (2h)
        abs(derivative) > 64eps(T) ||
            throw(ArgumentError("asymptotic wavefunction derivative is too small"))
        estimates[output_index] =
            radii[radial_index] -
            wavefunction[radial_index] / derivative
    end

    value = sum(estimates) / length(estimates)
    spread = maximum(estimates) - minimum(estimates)
    return ScatteringLengthResult(value, spread, wavefunction, tail_indices)
end
