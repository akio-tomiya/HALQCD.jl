"""
    jackknife_estimates(samples, estimator=identity)

Construct delete-one jackknife estimates from scalar or array-like samples.
Nonlinear derived quantities, including potentials, should be supplied through
`estimator` so they are reconstructed independently for every jackknife
sample.
"""
function jackknife_estimates(samples::AbstractVector, estimator=identity)
    n = length(samples)
    n >= 2 || throw(ArgumentError("at least two samples are required"))
    total = reduce(+, samples)
    return [estimator((total - samples[i]) / (n - 1)) for i in eachindex(samples)]
end

"""
    jackknife_summary(estimates)

Return the mean and jackknife standard error for scalar or array-like
delete-one estimates.
"""
function jackknife_summary(estimates::AbstractVector)
    n = length(estimates)
    n >= 2 || throw(ArgumentError("at least two estimates are required"))
    mean_value = reduce(+, estimates) / n
    variance = reduce(+, (abs2.(x .- mean_value) for x in estimates))
    stderr = sqrt.(((n - 1) / n) .* variance)
    return (mean=mean_value, stderr=stderr)
end

"""
    JackknifePotentialResult

End-to-end potential extraction from already constructed delete-one
jackknife R-correlators.

- `estimates`: one [`PotentialResult`](@ref) per jackknife sample.
- `components`: jackknife mean and standard error for the divided `spatial`,
  `first_time`, `second_time`, and `total` potential terms.
- `valid`: sites valid in every jackknife sample.

Component summaries outside `valid` must not be interpreted.
"""
struct JackknifePotentialResult{E,C,M}
    estimates::E
    components::C
    valid::M
end

"""
    lo_potential_from_jackknife_equal_mass(rminus, rzero, rplus; kwargs...)

Apply [`lo_potential_equal_mass`](@ref) independently to every delete-one
jackknife sample. Inputs have shape `(x, y, z, jackknife_sample)` and `kwargs`
are forwarded to the single-sample extractor.

This function does not construct delete-one samples from configurations. Its
name is intentionally explicit so raw configurations cannot be silently
mistaken for jackknife estimates.
"""
function lo_potential_from_jackknife_equal_mass(
    rminus::AbstractArray{T1,4},
    rzero::AbstractArray{T2,4},
    rplus::AbstractArray{T3,4};
    kwargs...,
) where {T1,T2,T3}
    size(rminus) == size(rzero) == size(rplus) ||
        throw(DimensionMismatch("all jackknife arrays must have identical sizes"))
    nsamples = size(rzero, 4)
    nsamples >= 2 || throw(ArgumentError("at least two jackknife samples are required"))

    estimates = [
        lo_potential_equal_mass(
            view(rminus, :, :, :, sample),
            view(rzero, :, :, :, sample),
            view(rplus, :, :, :, sample);
            kwargs...,
        )
        for sample in axes(rzero, 4)
    ]
    divided = [
        divided_potential_components(
            estimates[sample],
            view(rzero, :, :, :, sample),
        )
        for sample in eachindex(estimates)
    ]

    valid = copy(estimates[1].valid)
    for estimate in @view estimates[2:end]
        valid .&= estimate.valid
    end

    summarize(component) = jackknife_summary([
        getproperty(sample, component) for sample in divided
    ])
    components = (
        spatial=summarize(:spatial),
        first_time=summarize(:first_time),
        second_time=summarize(:second_time),
        total=summarize(:total),
    )

    return JackknifePotentialResult(estimates, components, valid)
end
