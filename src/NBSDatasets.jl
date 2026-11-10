const NBS_DATASET_SCHEMA_VERSION = v"1.0.0"
const NBS_DATASET_AXIS_ORDER = (
    :x,
    :y,
    :z,
    :time,
    :source,
    :sample,
)

const _NBS_REQUIRED_METADATA = (
    :channel,
    :spin,
    :source_operator,
    :sink_operator,
    :temporal_boundary_condition,
    :lattice_shape,
    :lattice_spacing,
    :magnetic_flux,
    :sample_ids,
    :source_ids,
    :code_commit,
)

function _require_nonempty_string(value, name)
    value isa AbstractString && !isempty(strip(value)) ||
        throw(ArgumentError("$name must be a nonempty string"))
    return value
end

"""
    NBSCorrelatorDataset(four_point, one_hadron_1, one_hadron_2; metadata,
                         sample_kind=:raw_configuration,
                         schema_version=NBS_DATASET_SCHEMA_VERSION)

Versioned in-memory container for a spatially resolved two-hadron correlator
and its matching one-hadron normalizations.

The fixed axis order of `four_point` is
`(x, y, z, time, source, sample)`. The fixed axis order of each one-hadron
correlator is `(time, source, sample)`. Raw configurations and derived
resamples are distinguished by `sample_kind`, rather than calling both kinds
of samples configurations.
"""
struct NBSCorrelatorDataset{
    TF<:AbstractArray,
    TC1<:AbstractArray,
    TC2<:AbstractArray,
    M<:NamedTuple,
}
    four_point::TF
    one_hadron_1::TC1
    one_hadron_2::TC2
    metadata::M
    sample_kind::Symbol
    schema_version::VersionNumber
end

function NBSCorrelatorDataset(
    four_point::AbstractArray{<:Number},
    one_hadron_1::AbstractArray{<:Number},
    one_hadron_2::AbstractArray{<:Number};
    metadata::NamedTuple,
    sample_kind::Symbol=:raw_configuration,
    schema_version::VersionNumber=NBS_DATASET_SCHEMA_VERSION,
)
    Base.require_one_based_indexing(four_point, one_hadron_1, one_hadron_2)
    schema_version == NBS_DATASET_SCHEMA_VERSION ||
        throw(ArgumentError(
            "unsupported NBS dataset schema version $schema_version; " *
            "expected $NBS_DATASET_SCHEMA_VERSION",
        ))
    sample_kind in (:raw_configuration, :derived_resample) ||
        throw(ArgumentError(
            "sample_kind must be :raw_configuration or :derived_resample",
        ))

    ndims(four_point) == 6 ||
        throw(DimensionMismatch(
            "four_point axes must be $(NBS_DATASET_AXIS_ORDER)",
        ))
    ndims(one_hadron_1) == 3 ||
        throw(DimensionMismatch(
            "one_hadron_1 axes must be (time, source, sample)",
        ))
    ndims(one_hadron_2) == 3 ||
        throw(DimensionMismatch(
            "one_hadron_2 axes must be (time, source, sample)",
        ))

    normalization_shape = size(four_point)[4:6]
    size(one_hadron_1) == normalization_shape ||
        throw(DimensionMismatch(
            "one_hadron_1 size $(size(one_hadron_1)) does not match " *
            "four_point time/source/sample size $normalization_shape",
        ))
    size(one_hadron_2) == normalization_shape ||
        throw(DimensionMismatch(
            "one_hadron_2 size $(size(one_hadron_2)) does not match " *
            "four_point time/source/sample size $normalization_shape",
        ))

    for key in _NBS_REQUIRED_METADATA
        haskey(metadata, key) ||
            throw(ArgumentError("NBS dataset metadata is missing :$key"))
    end
    for key in (
        :channel,
        :spin,
        :source_operator,
        :sink_operator,
        :temporal_boundary_condition,
    )
        _require_nonempty_string(getproperty(metadata, key), "metadata.$key")
    end
    if sample_kind == :derived_resample
        haskey(metadata, :resampling_method) ||
            throw(ArgumentError(
                "derived resamples require metadata.resampling_method",
            ))
        metadata.resampling_method isa AbstractString &&
            !isempty(strip(metadata.resampling_method)) ||
            throw(ArgumentError(
                "metadata.resampling_method must be a nonempty string",
            ))
        haskey(metadata, :parent_configuration_ids) ||
            throw(ArgumentError(
                "derived resamples require metadata.parent_configuration_ids",
            ))
        parent_configuration_ids = collect(metadata.parent_configuration_ids)
        isempty(parent_configuration_ids) &&
            throw(ArgumentError(
                "metadata.parent_configuration_ids must not be empty",
            ))
        allunique(parent_configuration_ids) ||
            throw(ArgumentError(
                "metadata.parent_configuration_ids must be unique",
            ))
        all(
            id -> id isa AbstractString && !isempty(strip(id)),
            parent_configuration_ids,
        ) ||
            throw(ArgumentError(
                "metadata.parent_configuration_ids must contain nonempty strings",
            ))
    end

    lattice_shape = Tuple(metadata.lattice_shape)
    length(lattice_shape) == 4 &&
        all(x -> x isa Integer && x > 0, lattice_shape) ||
        throw(ArgumentError(
            "metadata.lattice_shape must contain four positive integers",
        ))
    lattice_shape == size(four_point)[1:4] ||
        throw(DimensionMismatch(
            "metadata.lattice_shape $lattice_shape does not match " *
            "four_point space-time shape $(size(four_point)[1:4])",
        ))

    lattice_spacing = Tuple(metadata.lattice_spacing)
    length(lattice_spacing) == 4 ||
        throw(ArgumentError("metadata.lattice_spacing must have four entries"))
    all(x -> x isa Real && isfinite(x) && x > 0, lattice_spacing) ||
        throw(ArgumentError(
            "metadata.lattice_spacing entries must be finite and positive",
        ))
    metadata.magnetic_flux isa Integer ||
        throw(ArgumentError("metadata.magnetic_flux must be an integer"))

    sample_ids = collect(metadata.sample_ids)
    length(sample_ids) == size(four_point, 6) ||
        throw(DimensionMismatch(
            "metadata.sample_ids length does not match sample axis",
        ))
    allunique(sample_ids) ||
        throw(ArgumentError("metadata.sample_ids must be unique"))
    all(id -> id isa AbstractString && !isempty(strip(id)), sample_ids) ||
        throw(ArgumentError(
            "metadata.sample_ids must contain nonempty strings",
        ))

    source_ids = collect(metadata.source_ids)
    length(source_ids) == size(four_point, 5) ||
        throw(DimensionMismatch(
            "metadata.source_ids length does not match source axis",
        ))
    allunique(source_ids) ||
        throw(ArgumentError("metadata.source_ids must be unique"))
    all(id -> id isa AbstractString && !isempty(strip(id)), source_ids) ||
        throw(ArgumentError(
            "metadata.source_ids must contain nonempty strings",
        ))

    _require_nonempty_string(metadata.code_commit, "metadata.code_commit")

    canonical_metadata = merge(
        metadata,
        (
            lattice_shape=lattice_shape,
            lattice_spacing=lattice_spacing,
            sample_ids=Tuple(sample_ids),
            source_ids=Tuple(source_ids),
        ),
    )
    if sample_kind == :derived_resample
        canonical_metadata = merge(
            canonical_metadata,
            (
                parent_configuration_ids=Tuple(
                    metadata.parent_configuration_ids,
                ),
            ),
        )
    end

    return NBSCorrelatorDataset{
        typeof(four_point),
        typeof(one_hadron_1),
        typeof(one_hadron_2),
        typeof(canonical_metadata),
    }(
        four_point,
        one_hadron_1,
        one_hadron_2,
        canonical_metadata,
        sample_kind,
        schema_version,
    )
end

"""
    normalization_mask(dataset::NBSCorrelatorDataset; atol=0)

Return a Boolean array with axes `(time, source, sample)`. An entry is `true`
only when the matching product of one-hadron correlators is finite and has
magnitude strictly larger than `atol`.

This function makes unavailable normalization points explicit. It does not
drop, interpolate, or replace them.
"""
function normalization_mask(dataset::NBSCorrelatorDataset; atol::Real=0)
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    products = dataset.one_hadron_1 .* dataset.one_hadron_2
    return map(products) do denominator
        magnitude = abs(denominator)
        isfinite(magnitude) && magnitude > atol
    end
end

"""
    r_correlator(dataset::NBSCorrelatorDataset; atol=0)

Normalize every time, source, and sample in an NBS dataset.
The returned array keeps the dataset's fixed six-axis order.

Use [`normalization_mask`](@ref) to inspect availability first when a dataset
may contain an exact or near-zero one-hadron normalization.
"""
function r_correlator(dataset::NBSCorrelatorDataset; atol::Real=0)
    denominator_type = Base.promote_op(
        *,
        eltype(dataset.one_hadron_1),
        eltype(dataset.one_hadron_2),
    )
    output_type = Base.promote_op(
        /,
        eltype(dataset.four_point),
        denominator_type,
    )
    output = similar(dataset.four_point, output_type)

    for sample in axes(dataset.four_point, 6)
        for source in axes(dataset.four_point, 5)
            for time in axes(dataset.four_point, 4)
                r_correlator!(
                    view(output, :, :, :, time, source, sample),
                    view(
                        dataset.four_point,
                        :,
                        :,
                        :,
                        time,
                        source,
                        sample,
                    ),
                    dataset.one_hadron_1[time, source, sample],
                    dataset.one_hadron_2[time, source, sample];
                    atol=atol,
                )
            end
        end
    end
    return output
end

function _delete_one_last_axis(values::AbstractArray)
    sample_axis = ndims(values)
    sample_count = size(values, sample_axis)
    sample_count >= 2 ||
        throw(ArgumentError("at least two raw samples are required"))

    output_type = Base.promote_op(/, eltype(values), Int)
    output = similar(values, output_type)
    total = dropdims(sum(values; dims=sample_axis); dims=sample_axis)
    for sample in axes(values, sample_axis)
        selectdim(output, sample_axis, sample) .=
            (total .- selectdim(values, sample_axis, sample)) ./ (sample_count - 1)
    end
    return output
end

"""
    delete_one_jackknife_dataset(dataset::NBSCorrelatorDataset)

Construct delete-one averages of `four_point`, `one_hadron_1`, and
`one_hadron_2` independently from a raw-configuration dataset. Normalization
must be applied to the returned dataset, so nonlinear `F/(C₁C₂)` is rebuilt
inside every jackknife sample.
"""
function delete_one_jackknife_dataset(dataset::NBSCorrelatorDataset)
    dataset.sample_kind == :raw_configuration ||
        throw(ArgumentError(
            "delete-one input must contain raw configurations",
        ))
    sample_count = size(dataset.four_point, 6)
    sample_count >= 2 ||
        throw(ArgumentError("at least two raw configurations are required"))

    parent_configuration_ids = dataset.metadata.sample_ids
    jackknife_ids = Tuple(
        "delete-one-without:" * string(id) for id in parent_configuration_ids
    )
    metadata = merge(
        dataset.metadata,
        (
            sample_ids=jackknife_ids,
            resampling_method="delete-one jackknife",
            parent_configuration_ids=parent_configuration_ids,
        ),
    )
    return NBSCorrelatorDataset(
        _delete_one_last_axis(dataset.four_point),
        _delete_one_last_axis(dataset.one_hadron_1),
        _delete_one_last_axis(dataset.one_hadron_2);
        metadata=metadata,
        sample_kind=:derived_resample,
    )
end
