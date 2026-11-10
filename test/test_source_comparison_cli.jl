include(joinpath(@__DIR__, "..", "scripts", "compare_spatial_nbs_sources.jl"))

function _source_shift_fixture_pair()
    nx = 2
    nt = 5
    first_four_point =
        Array{ComplexF64}(undef, nx, nx, nx, nt, 1, 1)
    first_hadron_1 = Array{ComplexF64}(undef, nt, 1, 1)
    first_hadron_2 = Array{ComplexF64}(undef, nt, 1, 1)
    for time in 1:nt
        first_hadron_1[time, 1, 1] =
            (2.0 + 0.13time) * exp(-0.17(time - 1))
        first_hadron_2[time, 1, 1] =
            (1.7 - 0.08time) * exp(-0.11(time - 1))
        for z in 1:nx, y in 1:nx, x in 1:nx
            first_four_point[x, y, z, time, 1, 1] = (
                1.0 +
                0.09x +
                0.07y +
                0.05z +
                0.03time +
                im * (0.02x - 0.01y + 0.015z)
            ) * exp(-0.23(time - 1))
        end
    end

    second_four_point = similar(first_four_point)
    second_hadron_1 = similar(first_hadron_1)
    second_hadron_2 = similar(first_hadron_2)
    for second_time in 1:nt
        first_time, hadron_sign =
            temporal_shift_mapping(second_time, 1, nt)
        second_four_point[:, :, :, second_time, 1, 1] .=
            hadron_sign^2 .* first_four_point[:, :, :, first_time, 1, 1]
        second_hadron_1[second_time, 1, 1] =
            hadron_sign * first_hadron_1[first_time, 1, 1]
        second_hadron_2[second_time, 1, 1] =
            hadron_sign * first_hadron_2[first_time, 1, 1]
    end

    shared_metadata = (
        channel="XiXi",
        spin="singlet",
        source_operator="bilocal point Xi",
        sink_operator="local Xi pair with center-of-mass sum",
        temporal_boundary_condition="antiperiodic-quark",
        lattice_shape=(nx, nx, nx, nt),
        lattice_spacing=(0.7, 0.8, 0.9, 1.2),
        magnetic_flux=0,
        sample_ids=["synthetic-cold"],
        source_ids=["bilocal-source"],
        code_commit="fixture",
        gauge_condition="cold",
        kappa_light=0.1,
        kappa_strange=0.09,
        julia_version=string(VERSION),
        solver_squared_residual_threshold=1e-20,
        center_of_mass_reduction="sum",
        six_quark_contraction="full 6x6 determinant",
        relative_coordinate_convention=(
            "array index n stores displacement n-1 modulo spatial extent"
        ),
        one_hadron_spin_reduction="(up-up + down-down) / 2",
        normalization_atol=0.0,
        normalization_policy="retain raw data and mask unavailable times",
    )
    first = NBSCorrelatorDataset(
        first_four_point,
        first_hadron_1,
        first_hadron_2;
        metadata=merge(
            shared_metadata,
            (
                source_time=1,
                source_positions=("(1, 1, 1, 1)", "(2, 1, 1, 1)"),
            ),
        ),
    )
    second = NBSCorrelatorDataset(
        second_four_point,
        second_hadron_1,
        second_hadron_2;
        metadata=merge(
            shared_metadata,
            (
                source_time=2,
                source_positions=("(1, 1, 1, 2)", "(2, 1, 1, 2)"),
            ),
        ),
    )
    return first, second
end

@testset "spatial NBS source-time comparison helpers" begin
    @test temporal_shift_mapping(2, 1, 5) == (1, 1)
    @test temporal_shift_mapping(1, 1, 5) == (5, -1)
    @test temporal_shift_mapping(5, -1, 5) == (1, -1)
    @test temporal_shift_mapping(3, 6, 5) == (2, -1)
    @test_throws ArgumentError temporal_shift_mapping(1, 0, 0)
    @test_throws BoundsError temporal_shift_mapping(0, 0, 5)

    values = ComplexF64[1, 2, 3]
    @test comparison_relative_error(values, values) == 0
    @test comparison_relative_error(values, 2 .* values) == 0.5
    @test_throws DimensionMismatch comparison_relative_error(
        values,
        values[1:2],
    )

    first_components = (1.0 + 0im, -0.999 + 0im, 0.0 + 0im)
    second_components = (1.0 + 1e-6 + 0im, -0.999 + 0im, 0.0 + 0im)
    cancellation = potential_difference_amplification_diagnostic(
        sum(first_components),
        sum(second_components),
        first_components,
        second_components,
    )
    @test cancellation["component_difference_l1_norm"] ≈ 1e-6
    @test cancellation["potential_absolute_difference"] ≈ 1e-6
    @test cancellation["triangle_bound_ratio"] ≈ 1
    @test cancellation["relative_error_amplification"] > 1000
    @test cancellation["relative_error_amplification"] <=
        cancellation["cancellation_upper_bound"]
    @test cancellation[
        "roundoff_aware_relative_error_amplification"
    ] <= cancellation["cancellation_upper_bound"]
    @test cancellation[
        "difference_closure_relative_to_component_scale"
    ] <=
        512eps(Float64)
    @test_throws ErrorException potential_difference_amplification_diagnostic(
        sum(first_components) + 0.1,
        sum(second_components),
        first_components,
        second_components,
    )

    first, second = _source_shift_fixture_pair()
    @test isnothing(validate_source_comparison_metadata(first, second))
    @test recorded_solver_algorithm(first) == "bicg"
    @test recorded_source_positions(first) ==
        [(1, 1, 1, 1), (2, 1, 1, 1)]
    @test recorded_source_time(first) == 1
    @test recorded_source_time(second) == 2
    fallback_first = NBSCorrelatorDataset(
        first.four_point,
        first.one_hadron_1,
        first.one_hadron_2;
        metadata=Base.structdiff(first.metadata, (; source_time=1)),
    )
    @test recorded_source_time(fallback_first) == 1
    @test_throws ErrorException recorded_source_time(
        NBSCorrelatorDataset(
            first.four_point,
            first.one_hadron_1,
            first.one_hadron_2;
            metadata=merge(first.metadata, (source_time=0,)),
        ),
    )
    @test_throws ErrorException recorded_source_time(
        NBSCorrelatorDataset(
            first.four_point,
            first.one_hadron_1,
            first.one_hadron_2;
            metadata=merge(
                first.metadata,
                (source_positions=("(1, 1, 1, 2)", "(2, 1, 1, 2)"),),
            ),
        ),
    )
    @test_throws ErrorException recorded_source_time(
        NBSCorrelatorDataset(
            first.four_point,
            first.one_hadron_1,
            first.one_hadron_2;
            metadata=merge(
                first.metadata,
                (
                    source_time=6,
                    source_positions=("(1, 1, 1, 6)", "(2, 1, 1, 6)"),
                ),
            ),
        ),
    )
    spatially_shifted_source = NBSCorrelatorDataset(
        second.four_point,
        second.one_hadron_1,
        second.one_hadron_2;
        metadata=merge(
            second.metadata,
            (source_positions=("(1, 2, 1, 2)", "(2, 2, 1, 2)"),),
        ),
    )
    @test_throws ErrorException validate_source_comparison_metadata(
        first,
        spatially_shifted_source,
    )
    out_of_bounds_source = NBSCorrelatorDataset(
        first.four_point,
        first.one_hadron_1,
        first.one_hadron_2;
        metadata=merge(
            first.metadata,
            (source_positions=("(0, 1, 1, 1)", "(2, 1, 1, 1)"),),
        ),
    )
    @test_throws ErrorException recorded_source_positions(
        out_of_bounds_source,
    )
    mismatched_solver = NBSCorrelatorDataset(
        second.four_point,
        second.one_hadron_1,
        second.one_hadron_2;
        metadata=merge(
            second.metadata,
            (solver_squared_residual_threshold=1e-18,),
        ),
    )
    @test_throws ErrorException validate_source_comparison_metadata(
        first,
        mismatched_solver,
    )
    @test isnothing(
        validate_source_comparison_metadata(
            first,
            mismatched_solver;
            allow_solver_threshold_mismatch=true,
        ),
    )
    @test_throws ErrorException validate_source_comparison_metadata(
        first,
        second;
        allow_solver_threshold_mismatch=true,
    )
    dense_solver = NBSCorrelatorDataset(
        second.four_point,
        second.one_hadron_1,
        second.one_hadron_2;
        metadata=merge(
            second.metadata,
            (solver_algorithm="dense-lu",),
        ),
    )
    @test_throws ErrorException validate_source_comparison_metadata(
        first,
        dense_solver,
    )
    @test isnothing(
        validate_source_comparison_metadata(
            first,
            dense_solver;
            allow_solver_algorithm_mismatch=true,
        ),
    )
    @test_throws ErrorException validate_source_comparison_metadata(
        first,
        second;
        allow_solver_algorithm_mismatch=true,
    )
    @test_throws ErrorException validate_source_comparison_metadata(
        first,
        dense_solver;
        allow_solver_algorithm_mismatch=true,
        allow_solver_threshold_mismatch=true,
    )
    mismatched_contraction = NBSCorrelatorDataset(
        second.four_point,
        second.one_hadron_1,
        second.one_hadron_2;
        metadata=merge(
            second.metadata,
            (six_quark_contraction="different contraction",),
        ),
    )
    @test_throws ErrorException validate_source_comparison_metadata(
        first,
        mismatched_contraction,
    )
    unrecorded_contraction = NBSCorrelatorDataset(
        second.four_point,
        second.one_hadron_1,
        second.one_hadron_2;
        metadata=Base.structdiff(
            second.metadata,
            (; six_quark_contraction="full 6x6 determinant"),
        ),
    )
    @test_throws ErrorException validate_source_comparison_metadata(
        first,
        unrecorded_contraction,
    )
    @test isnothing(
        validate_source_comparison_metadata(
            first,
            unrecorded_contraction;
            allow_unrecorded_six_quark_contraction=true,
        ),
    )
    @test comparison_environment_boolean(
        "HALQCD_TEST_COMPARISON_BOOLEAN",
    ) == false
    withenv("HALQCD_TEST_COMPARISON_BOOLEAN" => "true") do
        @test comparison_environment_boolean(
            "HALQCD_TEST_COMPARISON_BOOLEAN",
        )
    end
    withenv("HALQCD_TEST_COMPARISON_BOOLEAN" => "yes") do
        @test_throws ErrorException comparison_environment_boolean(
            "HALQCD_TEST_COMPARISON_BOOLEAN",
        )
    end
    mismatched_source = NBSCorrelatorDataset(
        second.four_point,
        second.one_hadron_1,
        second.one_hadron_2;
        metadata=merge(
            second.metadata,
            (source_ids=("different-source",),),
        ),
    )
    @test_throws ErrorException validate_source_comparison_metadata(
        first,
        mismatched_source,
    )
    pure_gauge_first = NBSCorrelatorDataset(
        first.four_point,
        first.one_hadron_1,
        first.one_hadron_2;
        metadata=merge(
            first.metadata,
            (
                gauge_condition="pure-gauge",
                gauge_transformation=(
                    "deterministic local SU(3) transform of cold links"
                ),
                sample_ids=["synthetic-pure-gauge"],
            ),
        ),
    )
    @test_throws ErrorException validate_source_comparison_metadata(
        first,
        pure_gauge_first,
    )
    @test_throws ErrorException validate_source_comparison_metadata(
        first,
        pure_gauge_first;
        allow_solver_threshold_mismatch=true,
        expect_gauge_equivalence=true,
    )
    @test_throws ErrorException validate_source_comparison_metadata(
        first,
        pure_gauge_first;
        allow_solver_algorithm_mismatch=true,
        expect_gauge_equivalence=true,
    )
    @test isnothing(
        validate_source_comparison_metadata(
            first,
            pure_gauge_first;
            expect_gauge_equivalence=true,
        ),
    )
    @test_throws ErrorException validate_source_comparison_metadata(
        first,
        first;
        expect_gauge_equivalence=true,
    )
    unexpected_pure_gauge = NBSCorrelatorDataset(
        pure_gauge_first.four_point,
        pure_gauge_first.one_hadron_1,
        pure_gauge_first.one_hadron_2;
        metadata=merge(
            pure_gauge_first.metadata,
            (gauge_transformation="unrecorded transformation",),
        ),
    )
    @test_throws ErrorException validate_source_comparison_metadata(
        first,
        unexpected_pure_gauge;
        expect_gauge_equivalence=true,
    )
    magnetic_first = NBSCorrelatorDataset(
        first.four_point,
        first.one_hadron_1,
        first.one_hadron_2;
        metadata=merge(first.metadata, (magnetic_flux=1,)),
    )
    magnetic_second = NBSCorrelatorDataset(
        second.four_point,
        second.one_hadron_1,
        second.one_hadron_2;
        metadata=merge(second.metadata, (magnetic_flux=1,)),
    )
    @test_throws ErrorException validate_source_comparison_metadata(
        magnetic_first,
        magnetic_second,
    )
end

@testset "spatial NBS solver-threshold diagnostic end to end" begin
    standard, _ = _source_shift_fixture_pair()
    strict = NBSCorrelatorDataset(
        standard.four_point,
        standard.one_hadron_1,
        standard.one_hadron_2;
        metadata=merge(
            standard.metadata,
            (solver_squared_residual_threshold=1e-24,),
        ),
    )
    mktempdir() do directory
        standard_path = joinpath(directory, "standard")
        strict_path = joinpath(directory, "strict")
        output_path = joinpath(directory, "comparison.toml")
        write_nbs_dataset(standard_path, standard)
        write_nbs_dataset(strict_path, strict)

        original_args = copy(ARGS)
        empty!(ARGS)
        append!(ARGS, (standard_path, "1", strict_path, "1", output_path))
        try
            withenv(
                "HALQCD_ALLOW_SOLVER_THRESHOLD_MISMATCH" => "true",
                "HALQCD_SOURCE_COMPARISON_RTOL" => "1e-13",
                "HALQCD_NBS_NORMALIZATION_ATOL" => "0",
                "HALQCD_POTENTIAL_RTOL" => "0",
            ) do
                compare_spatial_nbs_sources_main()
            end
        finally
            empty!(ARGS)
            append!(ARGS, original_args)
        end

        summary = TOML.parsefile(output_path)
        @test summary["solver_threshold_mismatch_comparison"]
        @test !summary["gauge_equivalence_comparison"]
        @test summary["source_shift"] == 0
        @test summary["first_solver_squared_residual_threshold"] == 1e-20
        @test summary["second_solver_squared_residual_threshold"] == 1e-24
        @test all(iszero, values(summary["maxima"]))
    end
end

@testset "spatial NBS gauge-equivalence comparison end to end" begin
    cold, _ = _source_shift_fixture_pair()
    pure_gauge = NBSCorrelatorDataset(
        cold.four_point,
        cold.one_hadron_1,
        cold.one_hadron_2;
        metadata=merge(
            cold.metadata,
            (
                gauge_condition="pure-gauge",
                gauge_transformation=(
                    "deterministic local SU(3) transform of cold links"
                ),
                sample_ids=["synthetic-pure-gauge"],
            ),
        ),
    )
    mktempdir() do directory
        cold_path = joinpath(directory, "cold")
        pure_gauge_path = joinpath(directory, "pure-gauge")
        output_path = joinpath(directory, "comparison.toml")
        write_nbs_dataset(cold_path, cold)
        write_nbs_dataset(pure_gauge_path, pure_gauge)

        original_args = copy(ARGS)
        empty!(ARGS)
        append!(ARGS, (cold_path, "1", pure_gauge_path, "1", output_path))
        try
            withenv(
                "HALQCD_EXPECT_GAUGE_EQUIVALENCE" => "true",
                "HALQCD_SOURCE_COMPARISON_RTOL" => "1e-13",
                "HALQCD_NBS_NORMALIZATION_ATOL" => "0",
                "HALQCD_POTENTIAL_RTOL" => "0",
            ) do
                compare_spatial_nbs_sources_main()
            end
        finally
            empty!(ARGS)
            append!(ARGS, original_args)
        end

        summary = TOML.parsefile(output_path)
        @test summary["gauge_equivalence_comparison"]
        @test summary["first_gauge_condition"] == "cold"
        @test summary["second_gauge_condition"] == "pure-gauge"
        @test summary["source_shift"] == 0
        @test all(iszero, values(summary["maxima"]))
        gauge_diagnostic = summary["per_time"]["sink_time_2"][
            "potential_comparison"
        ]["maximum_difference_diagnostic"]
        @test gauge_diagnostic["maximum_absolute_difference"] == 0
        @test length(gauge_diagnostic["zero_based_displacement"]) == 3
        @test gauge_diagnostic["first_potential"] ==
            gauge_diagnostic["second_potential"]
    end
end

@testset "spatial NBS source-time comparison end to end" begin
    first, second = _source_shift_fixture_pair()
    mktempdir() do directory
        first_path = joinpath(directory, "first")
        second_path = joinpath(directory, "second")
        output_path = joinpath(directory, "comparison.toml")
        write_nbs_dataset(first_path, first)
        write_nbs_dataset(second_path, second)

        original_args = copy(ARGS)
        empty!(ARGS)
        append!(
            ARGS,
            (first_path, "1", second_path, "2", output_path),
        )
        try
            withenv(
                "HALQCD_SOURCE_COMPARISON_RTOL" => "1e-13",
                "HALQCD_NBS_NORMALIZATION_ATOL" => "0",
                "HALQCD_POTENTIAL_RTOL" => "0",
                "HALQCD_ALLOW_UNRECORDED_SIX_QUARK_CONTRACTION" => nothing,
                "HALQCD_EXPECT_GAUGE_EQUIVALENCE" => nothing,
            ) do
                compare_spatial_nbs_sources_main()
            end
        finally
            empty!(ARGS)
            append!(ARGS, original_args)
        end

        summary = TOML.parsefile(output_path)
        @test summary["format_version"] == "1.1.0"
        @test summary["source_shift"] == 1
        @test summary["first_six_quark_contraction"] ==
            "full 6x6 determinant"
        @test summary["second_six_quark_contraction"] ==
            "full 6x6 determinant"
        @test !summary["allowed_unrecorded_six_quark_contraction"]
        @test !summary["gauge_equivalence_comparison"]
        @test summary["first_gauge_condition"] == "cold"
        @test summary["second_gauge_condition"] == "cold"
        @test summary["potential_comparison_pairs"] == [[2, 3], [3, 4]]
        @test all(value <= 1e-14 for value in values(summary["maxima"]))
        source_diagnostic = summary["per_time"]["sink_time_3"][
            "potential_comparison"
        ]["maximum_difference_diagnostic"]
        @test source_diagnostic["maximum_absolute_difference"] == 0
        @test source_diagnostic["first_components"] ==
            source_diagnostic["second_components"]
        source_cancellation =
            source_diagnostic["cancellation_diagnostic"]
        @test source_cancellation["component_difference_l1_norm"] == 0
        @test source_cancellation["potential_absolute_difference"] == 0
        @test source_cancellation["relative_error_amplification"] == 0
        @test source_cancellation[
            "roundoff_aware_relative_error_amplification"
        ] == 0
        @test source_cancellation["triangle_bound_ratio"] == 0
        @test !ispath(output_path * ".tmp-$(getpid())")
    end
end
