function _nbs_fixture_metadata(; kwargs...)
    defaults = (
        channel="XiXi",
        spin="singlet",
        source_operator="local-Xi",
        sink_operator="local-Xi",
        temporal_boundary_condition="antiperiodic-quark",
        lattice_shape=(2, 2, 2, 3),
        lattice_spacing=(1.0, 1.0, 1.0, 1.0),
        magnetic_flux=0,
        sample_ids=["cfg-001", "cfg-002"],
        source_ids=["src-0", "src-1"],
        code_commit="fixture",
    )
    return merge(defaults, (; kwargs...))
end

@testset "Versioned NBS correlator dataset" begin
    four_point = reshape(
        ComplexF64.(1:(2 * 2 * 2 * 3 * 2 * 2)),
        2,
        2,
        2,
        3,
        2,
        2,
    )
    one_hadron_1 = fill(2.0, 3, 2, 2)
    one_hadron_2 = reshape(1.0:12.0, 3, 2, 2)
    metadata = _nbs_fixture_metadata()

    dataset = NBSCorrelatorDataset(
        four_point,
        one_hadron_1,
        one_hadron_2;
        metadata=metadata,
    )
    @test dataset.schema_version == v"1.0.0"
    @test dataset.sample_kind == :raw_configuration
    @test NBS_DATASET_AXIS_ORDER ==
          (:x, :y, :z, :time, :source, :sample)
    @test dataset.metadata.sample_ids == ("cfg-001", "cfg-002")
    @test dataset.metadata.source_ids == ("src-0", "src-1")

    normalized = r_correlator(dataset)
    @test normalization_mask(dataset) == trues(3, 2, 2)
    for sample in 1:2, source in 1:2, time in 1:3
        denominator =
            one_hadron_1[time, source, sample] *
            one_hadron_2[time, source, sample]
        @test normalized[:, :, :, time, source, sample] ==
              four_point[:, :, :, time, source, sample] ./ denominator
    end

    @test_throws ArgumentError NBSCorrelatorDataset(
        four_point,
        one_hadron_1,
        one_hadron_2;
        metadata=metadata,
        schema_version=v"2.0.0",
    )
    @test_throws ArgumentError NBSCorrelatorDataset(
        four_point,
        one_hadron_1,
        one_hadron_2;
        metadata=metadata,
        sample_kind=:jackknife,
    )
    @test_throws DimensionMismatch NBSCorrelatorDataset(
        four_point[:, :, :, :, :, 1],
        one_hadron_1,
        one_hadron_2;
        metadata=metadata,
    )
    @test_throws DimensionMismatch NBSCorrelatorDataset(
        four_point,
        one_hadron_1[:, :, 1:1],
        one_hadron_2;
        metadata=metadata,
    )
    @test_throws ArgumentError NBSCorrelatorDataset(
        four_point,
        one_hadron_1,
        one_hadron_2;
        metadata=Base.structdiff(metadata, (; channel=metadata.channel)),
    )
    @test_throws DimensionMismatch NBSCorrelatorDataset(
        four_point,
        one_hadron_1,
        one_hadron_2;
        metadata=_nbs_fixture_metadata(lattice_shape=(2, 2, 3, 3)),
    )
    @test_throws ArgumentError NBSCorrelatorDataset(
        four_point,
        one_hadron_1,
        one_hadron_2;
        metadata=_nbs_fixture_metadata(
            lattice_shape=(2.0, 2.0, 2.0, 3.0),
        ),
    )
    @test_throws DimensionMismatch NBSCorrelatorDataset(
        four_point,
        one_hadron_1,
        one_hadron_2;
        metadata=_nbs_fixture_metadata(sample_ids=["cfg-001"]),
    )
    @test_throws ArgumentError NBSCorrelatorDataset(
        four_point,
        one_hadron_1,
        one_hadron_2;
        metadata=_nbs_fixture_metadata(source_ids=["same", "same"]),
    )
    @test_throws ArgumentError NBSCorrelatorDataset(
        four_point,
        one_hadron_1,
        one_hadron_2;
        metadata=_nbs_fixture_metadata(channel=" "),
    )
    @test_throws ArgumentError NBSCorrelatorDataset(
        four_point,
        one_hadron_1,
        one_hadron_2;
        metadata=_nbs_fixture_metadata(sample_ids=["cfg-001", " "]),
    )
    @test_throws ArgumentError NBSCorrelatorDataset(
        four_point,
        one_hadron_1,
        one_hadron_2;
        metadata=merge(metadata, (resampling_method="",)),
        sample_kind=:derived_resample,
    )
    @test_throws ArgumentError NBSCorrelatorDataset(
        four_point,
        one_hadron_1,
        one_hadron_2;
        metadata=merge(
            metadata,
            (resampling_method="delete-one jackknife",),
        ),
        sample_kind=:derived_resample,
    )
    derived = NBSCorrelatorDataset(
        four_point,
        one_hadron_1,
        one_hadron_2;
        metadata=merge(
            metadata,
            (
                resampling_method="delete-one jackknife",
                parent_configuration_ids=["cfg-001", "cfg-002"],
            ),
        ),
        sample_kind=:derived_resample,
    )
    @test derived.sample_kind == :derived_resample
    @test derived.metadata.parent_configuration_ids ==
          ("cfg-001", "cfg-002")

    mutable_sample_ids = ["cfg-001", "cfg-002"]
    immutable_metadata = _nbs_fixture_metadata(
        sample_ids=mutable_sample_ids,
    )
    immutable_dataset = NBSCorrelatorDataset(
        four_point,
        one_hadron_1,
        one_hadron_2;
        metadata=immutable_metadata,
    )
    mutable_sample_ids[1] = "changed-after-validation"
    @test immutable_dataset.metadata.sample_ids == ("cfg-001", "cfg-002")

    zero_normalization = copy(one_hadron_1)
    zero_normalization[2, 1, 2] = 0
    invalid = NBSCorrelatorDataset(
        four_point,
        zero_normalization,
        one_hadron_2;
        metadata=metadata,
    )
    expected_mask = trues(3, 2, 2)
    expected_mask[2, 1, 2] = false
    @test normalization_mask(invalid) == expected_mask
    @test normalization_mask(invalid; atol=4) ==
          (abs.(zero_normalization .* one_hadron_2) .> 4)
    @test_throws ArgumentError normalization_mask(invalid; atol=-1)
    @test_throws DomainError r_correlator(invalid)
end

@testset "Delete-one NBS dataset" begin
    four_point = reshape(ComplexF64.(1:24), 2, 2, 1, 1, 1, 6)
    one_hadron_1 = reshape(Float64[1, 2, 4, 8, 16, 32], 1, 1, 6)
    one_hadron_2 = reshape(Float64[2, 3, 5, 7, 11, 13], 1, 1, 6)
    metadata = _nbs_fixture_metadata(
        lattice_shape=(2, 2, 1, 1),
        sample_ids=["cfg-$i" for i in 1:6],
        source_ids=["src-0"],
    )
    raw = NBSCorrelatorDataset(
        four_point,
        one_hadron_1,
        one_hadron_2;
        metadata=metadata,
    )
    jackknife = delete_one_jackknife_dataset(raw)

    @test jackknife.sample_kind == :derived_resample
    @test jackknife.metadata.resampling_method == "delete-one jackknife"
    @test jackknife.metadata.parent_configuration_ids ==
          Tuple(metadata.sample_ids)
    @test eltype(jackknife.four_point) == ComplexF64
    @test jackknife.metadata.sample_ids[3] ==
          "delete-one-without:cfg-3"

    for omitted in 1:6
        kept = [sample for sample in 1:6 if sample != omitted]
        @test jackknife.four_point[:, :, :, :, :, omitted] ==
              dropdims(
                  sum(four_point[:, :, :, :, :, kept]; dims=6);
                  dims=6,
              ) ./ 5
        @test jackknife.one_hadron_1[1, 1, omitted] ==
              sum(one_hadron_1[1, 1, kept]) / 5
        @test jackknife.one_hadron_2[1, 1, omitted] ==
              sum(one_hadron_2[1, 1, kept]) / 5
    end

    rebuilt = r_correlator(jackknife)
    normalized_first = HALQCD._delete_one_last_axis(r_correlator(raw))
    @test !isapprox(rebuilt, normalized_first; rtol=1e-10, atol=1e-12)

    @test_throws ArgumentError delete_one_jackknife_dataset(jackknife)
    one_sample = NBSCorrelatorDataset(
        four_point[:, :, :, :, :, 1:1],
        one_hadron_1[:, :, 1:1],
        one_hadron_2[:, :, 1:1];
        metadata=_nbs_fixture_metadata(
            lattice_shape=(2, 2, 1, 1),
            sample_ids=["cfg-1"],
            source_ids=["src-0"],
        ),
    )
    @test_throws ArgumentError delete_one_jackknife_dataset(one_sample)
end
