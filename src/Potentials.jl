"""
    PotentialResult

Result of an equal-mass, leading-order potential extraction.

Fields:

- `potential`: extracted pointwise potential, with invalid sites set to zero.
- `valid`: mask identifying sites used in the pointwise division.
- `numerator`: sum of all terms before division by `R`.
- `spatial`: `∇²R/m`.
- `first_time`: `-∂ₜR`.
- `second_time`: `∂ₜ²R/(4m)`.
"""
struct PotentialResult{P,M,N,S,D1,D2}
    potential::P
    valid::M
    numerator::N
    spatial::S
    first_time::D1
    second_time::D2
end

function _finite_scale(values)
    isempty(values) && throw(ArgumentError("time slices must not be empty"))
    return mapreduce(
        value -> begin
            magnitude = abs(value)
            isfinite(magnitude) ? magnitude : zero(magnitude)
        end,
        max,
        values,
    )
end

"""
    lo_potential_equal_mass(rminus, rzero, rplus;
                            mass, spacing=(1,1,1), at=1,
                            rtol=sqrt(eps()), atol=0)

Extract the leading-order time-dependent HAL QCD potential for an equal-mass,
single-channel system at zero external field:

`V = R⁻¹ [(1/(4m))∂ₜ² - ∂ₜ + ∇²/m] R`.

A site is valid only when `R` and the numerator are finite and
`abs(R) > max(atol, rtol * maximum(abs, R))`. The relative cutoff is invariant
under a global rescaling `R → cR`.

This function is a `B=0` API. It must not be interpreted as a completed
background-field HAL QCD formulation.
"""
function lo_potential_equal_mass(
    rminus::AbstractArray{T1,3},
    rzero::AbstractArray{T2,3},
    rplus::AbstractArray{T3,3};
    mass::Real,
    spacing::NTuple{3,<:Real}=(1.0, 1.0, 1.0),
    at::Real=1.0,
    rtol::Real=sqrt(eps(Float64)),
    atol::Real=0,
) where {T1,T2,T3}
    isfinite(mass) && mass > 0 ||
        throw(ArgumentError("mass must be finite and positive"))
    isfinite(rtol) && rtol >= 0 ||
        throw(ArgumentError("rtol must be finite and nonnegative"))
    isfinite(atol) && atol >= 0 ||
        throw(ArgumentError("atol must be finite and nonnegative"))
    size(rminus) == size(rzero) == size(rplus) ||
        throw(DimensionMismatch("all time slices must have identical sizes"))

    lap = laplacian_periodic(rzero, spacing)
    dt, d2t = time_derivatives(rminus, rzero, rplus, at)
    spatial = lap ./ mass
    first_time = .-dt
    second_time = d2t ./ (4 * mass)
    numerator = spatial .+ first_time .+ second_time

    Tout = promote_type(eltype(numerator), eltype(rzero))
    potential = similar(numerator, Tout)
    valid = falses(size(rzero))

    scale = _finite_scale(rzero)
    cutoff = max(atol, rtol * scale)

    @inbounds for i in eachindex(rzero, numerator, potential, valid)
        denominator_magnitude = abs(rzero[i])
        numerator_magnitude = abs(numerator[i])
        if isfinite(denominator_magnitude) &&
           isfinite(numerator_magnitude) &&
           denominator_magnitude > cutoff
            potential[i] = numerator[i] / rzero[i]
            valid[i] = true
        else
            potential[i] = zero(Tout)
        end
    end

    return PotentialResult(
        potential,
        valid,
        numerator,
        spatial,
        first_time,
        second_time,
    )
end

"""
    lo_potential_equal_mass(dataset::NBSCorrelatorDataset, time;
                            source=1, sample=1,
                            normalization_atol=0,
                            spacing=dataset.metadata.lattice_spacing[1:3],
                            at=dataset.metadata.lattice_spacing[4],
                            kwargs...)

Extract the zero-field equal-mass LO potential from three consecutive,
interior time slices of an NBS dataset. The selected `time - 1`, `time`, and
`time + 1` normalizations must all pass [`normalization_mask`](@ref).

The method deliberately does not wrap the temporal boundary and does not
interpolate an unavailable normalization. Spatial and temporal lattice
spacings default to the values stored in `dataset.metadata.lattice_spacing`;
explicit keyword arguments may override them. Remaining keyword arguments
are forwarded to the three-array [`lo_potential_equal_mass`](@ref) method.
"""
function lo_potential_equal_mass(
    dataset::NBSCorrelatorDataset,
    time::Integer;
    source::Integer=1,
    sample::Integer=1,
    normalization_atol::Real=0,
    spacing=Tuple(dataset.metadata.lattice_spacing[1:3]),
    at=dataset.metadata.lattice_spacing[4],
    kwargs...,
)
    nt = size(dataset.four_point, 4)
    2 <= time <= nt - 1 ||
        throw(ArgumentError(
            "time must have two interior neighbors; got time=$time for nt=$nt",
        ))
    checkbounds(dataset.one_hadron_1, time, source, sample)
    available = normalization_mask(dataset; atol=normalization_atol)
    selected_times = (time - 1, time, time + 1)
    unavailable_times = filter(
        selected_time ->
            !available[selected_time, source, sample],
        selected_times,
    )
    isempty(unavailable_times) ||
        throw(
            DomainError(
                unavailable_times,
                "the selected three-slice stencil contains unavailable " *
                "one-hadron normalizations",
            ),
        )

    normalized_slice(selected_time) = r_correlator(
        view(
            dataset.four_point,
            :,
            :,
            :,
            selected_time,
            source,
            sample,
        ),
        dataset.one_hadron_1[selected_time, source, sample],
        dataset.one_hadron_2[selected_time, source, sample];
        atol=normalization_atol,
    )
    return lo_potential_equal_mass(
        normalized_slice(time - 1),
        normalized_slice(time),
        normalized_slice(time + 1);
        spacing,
        at,
        kwargs...,
    )
end

"""
    divided_potential_components(result, denominator)

Return the three contributions to the extracted potential after pointwise
division by the R-correlator:

- `spatial = (∇²R/m)/R`
- `first_time = (-∂ₜR)/R`
- `second_time = (∂ₜ²R/(4m))/R`
- `total = spatial + first_time + second_time`

Invalid sites use zero and are identified by the returned `valid` mask. This
helper makes the distinction between numerator terms stored in
[`PotentialResult`](@ref) and the divided terms printed by common reference
codes explicit.
"""
function divided_potential_components(
    result::PotentialResult,
    denominator::AbstractArray,
)
    size(denominator) == size(result.potential) ||
        throw(DimensionMismatch("denominator and result must have identical sizes"))

    spatial = similar(result.potential)
    first_time = similar(result.potential)
    second_time = similar(result.potential)

    @inbounds for i in eachindex(
        denominator,
        result.valid,
        spatial,
        first_time,
        second_time,
    )
        if result.valid[i]
            spatial[i] = result.spatial[i] / denominator[i]
            first_time[i] = result.first_time[i] / denominator[i]
            second_time[i] = result.second_time[i] / denominator[i]
        else
            spatial[i] = zero(eltype(spatial))
            first_time[i] = zero(eltype(first_time))
            second_time[i] = zero(eltype(second_time))
        end
    end

    return (
        spatial=spatial,
        first_time=first_time,
        second_time=second_time,
        total=result.potential,
        valid=result.valid,
    )
end
