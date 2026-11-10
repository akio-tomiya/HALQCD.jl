module HALQCDQCDMeasurementsExt

using HALQCD
using Gaugefields
using LatticeDiracOperators
using QCDMeasurements

struct HALNBSMeasurement{K<:NamedTuple} <: QCDMeasurements.AbstractMeasurement
    channel::String
    sample_id::String
    source_positions::Vector{NTuple{4,Int}}
    propagator_options::K
    verbose_level::Int
end

# Keyword names and defaults follow the Wilson branch of
# QCDMeasurements.Pion_correlator_measurement.
function HALQCD.HAL_NBS_measurement(
    U::AbstractVector{<:AbstractGaugefields{NC,4}};
    channel::AbstractString=HALQCD.PIPI_I2_CHANNEL,
    sample_id::AbstractString,
    fermiontype="Wilson",
    κ=1,
    r=1,
    eps_CG=1e-14,
    MaxCGstep=5000,
    method_CG=nothing,
    cSW=nothing,
    BoundaryCondition=nothing,
    source_positions=[(1, 1, 1, 1)],
    verbose_level=2,
    cov_neural_net=nothing,
) where {NC}
    channel == HALQCD.PIPI_I2_CHANNEL ||
        throw(ArgumentError("only the $(HALQCD.PIPI_I2_CHANNEL) channel is implemented"))
    fermiontype in ("Wilson", "WilsonClover") ||
        throw(ArgumentError("only Wilson and WilsonClover fermions are supported so far"))
    isnothing(cov_neural_net) ||
        throw(ArgumentError("smeared links are not supported yet"))
    boundary_condition =
        isnothing(BoundaryCondition) ? QCDMeasurements.BoundaryCondition_4D_default :
        BoundaryCondition
    options = (
        κ=κ,
        r=r,
        fermiontype=fermiontype,
        eps_CG=eps_CG,
        MaxCGstep=MaxCGstep,
        method_CG=method_CG,
        cSW=cSW,
        BoundaryCondition=collect(boundary_condition),
    )
    return HALNBSMeasurement(
        String(channel),
        String(sample_id),
        [Tuple(Int.(position)) for position in source_positions],
        options,
        Int(verbose_level),
    )
end

function QCDMeasurements.measure(
    m::HALNBSMeasurement,
    U::AbstractVector{<:AbstractGaugefields{NC,4}};
    additional_string="",
) where {NC}
    seconds = @elapsed dataset = HALQCD.juliaqcd_pipi_i2_nbs(
        U;
        sample_id=m.sample_id,
        source_positions=m.source_positions,
        verbose_level=max(m.verbose_level - 2, 0),
        m.propagator_options...,
    )
    outputstring =
        "$additional_string HAL_NBS $(m.channel) sample=$(m.sample_id) " *
        "sources=$(length(m.source_positions)) " *
        "max_residual=$(dataset.metadata.maximum_external_residual) " *
        "seconds=$(round(seconds; digits=3))"
    m.verbose_level >= 2 && println(outputstring)
    return QCDMeasurements.Measurement_output(dataset, outputstring)
end

end
