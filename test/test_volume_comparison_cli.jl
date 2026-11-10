include(joinpath(@__DIR__, "..", "scripts", "compare_spatial_nbs_volumes.jl"))

function _volume_comparison_fixture(nx; amplitude=1.0)
    nt = 5
    time_dependence = exp.(-0.2 .* (0:(nt - 1)))
    four_point = Array{ComplexF64}(undef, nx, nx, nx, nt, 1, 1)
    for time in 1:nt
        four_point[:, :, :, time, 1, 1] .=
            amplitude * time_dependence[time]
    end
    one_hadron = ones(Float64, nt, 1, 1)
    metadata = (
        channel="XiXi",
        spin="singlet",
        source_operator="local-Xi",
        sink_operator="local-Xi",
        temporal_boundary_condition="antiperiodic-quark",
        lattice_shape=(nx, nx, nx, nt),
        lattice_spacing=(1.0, 1.0, 1.0, 1.0),
        magnetic_flux=0,
        sample_ids=["cold"],
        source_ids=["src-1"],
        code_commit="fixture",
        gauge_condition="cold",
        kappa_light=0.1,
        kappa_strange=0.09,
        solver_squared_residual_threshold=1e-20,
        source_positions=("(1, 1, 1, 1)", "(2, 1, 1, 1)"),
        center_of_mass_reduction="sum",
        relative_coordinate_convention=(
            "array index n stores displacement n-1 modulo spatial extent"
        ),
        one_hadron_spin_reduction="(up-up + down-down) / 2",
    )
    return NBSCorrelatorDataset(
        four_point,
        one_hadron,
        one_hadron;
        metadata,
    )
end

@testset "spatial NBS volume comparison helpers" begin
    small = _volume_comparison_fixture(2)
    large = _volume_comparison_fixture(3)
    diagnostics = spatial_volume_diagnostics(
        small,
        large;
        mass=2.0,
        potential_rtol=0.0,
    )

    @test diagnostics["first_spatial_shape"] == [2, 2, 2]
    @test diagnostics["second_spatial_shape"] == [3, 3, 3]
    @test diagnostics["common_stencil_times"] == [2, 3, 4]
    @test diagnostics["maximum_origin_r_relative_error"] ≈ 0 atol = 1e-15
    @test diagnostics["maximum_origin_potential_relative_error"] ≈ 0 atol =
        1e-14
    @test diagnostics["maximum_common_cube_r_relative_error"] ≈ 0 atol =
        1e-15
    @test diagnostics["maximum_common_cube_potential_relative_error"] ≈ 0 atol =
        1e-14
    @test diagnostics["maximum_common_cube_r_shape_relative_error"] ≈ 0 atol =
        1e-15
    @test all(
        entry["first_maximum_abs_r"] ≈ entry["second_maximum_abs_r"]
        for entry in values(diagnostics["per_time"])
    )
    @test all(
        entry["first_maximum_abs_potential"] ≈
        entry["second_maximum_abs_potential"]
        for entry in values(diagnostics["per_time"])
    )
    @test all(
        isapprox(
            entry["common_cube_r_relative_error"],
            0;
            atol=1e-15,
        )
        for entry in values(diagnostics["per_time"])
    )
    @test all(
        isapprox(
            entry["common_cube_potential_relative_error"],
            0;
            atol=1e-14,
        )
        for entry in values(diagnostics["per_time"])
    )
    @test all(
        entry["common_cube_valid_sites"] == 8
        for entry in values(diagnostics["per_time"])
    )

    volume_scaled_small = _volume_comparison_fixture(2; amplitude=1 / 8)
    volume_scaled_large = _volume_comparison_fixture(3; amplitude=1 / 27)
    volume_scaled_diagnostics = spatial_volume_diagnostics(
        volume_scaled_small,
        volume_scaled_large;
        mass=2.0,
        potential_rtol=0.0,
    )
    @test volume_scaled_diagnostics["maximum_origin_r_relative_error"] ≈
        19 / 27
    @test volume_scaled_diagnostics[
        "maximum_volume_rescaled_origin_r_relative_error"
    ] ≈ 0 atol = 1e-15
    @test volume_scaled_diagnostics[
        "maximum_volume_rescaled_common_cube_r_relative_error"
    ] ≈ 0 atol = 1e-15
    @test volume_scaled_diagnostics[
        "maximum_common_cube_r_shape_relative_error"
    ] ≈ 0 atol = 1e-15
    @test volume_scaled_diagnostics[
        "maximum_common_cube_potential_relative_error"
    ] ≈ 0 atol = 1e-14

    @test scalar_symmetric_relative_error(1.0, 2.0) ==
        scalar_symmetric_relative_error(2.0, 1.0)
    @test field_symmetric_relative_error([1.0, 2.0], [1.0, 2.0]) == 0
    @test_throws DimensionMismatch field_symmetric_relative_error(
        [1.0, 2.0],
        [1.0],
    )
    @test_throws ArgumentError field_symmetric_relative_error(
        Float64[],
        Float64[],
    )
    @test_throws ErrorException spatial_volume_diagnostics(
        large,
        small;
        mass=2.0,
    )
    @test_throws ArgumentError spatial_volume_diagnostics(
        small,
        large;
        mass=0.0,
    )

    mismatched_source = NBSCorrelatorDataset(
        large.four_point,
        large.one_hadron_1,
        large.one_hadron_2;
        metadata=merge(large.metadata, (source_ids=("different-source",),)),
    )
    @test_throws ErrorException spatial_volume_diagnostics(
        small,
        mismatched_source;
        mass=2.0,
    )

    mismatched_source_positions = NBSCorrelatorDataset(
        large.four_point,
        large.one_hadron_1,
        large.one_hadron_2;
        metadata=merge(
            large.metadata,
            (source_positions=("(1, 1, 1, 1)", "(3, 1, 1, 1)"),),
        ),
    )
    @test_throws ErrorException spatial_volume_diagnostics(
        small,
        mismatched_source_positions;
        mass=2.0,
    )

    magnetic_small = NBSCorrelatorDataset(
        small.four_point,
        small.one_hadron_1,
        small.one_hadron_2;
        metadata=merge(small.metadata, (magnetic_flux=1,)),
    )
    magnetic_large = NBSCorrelatorDataset(
        large.four_point,
        large.one_hadron_1,
        large.one_hadron_2;
        metadata=merge(large.metadata, (magnetic_flux=1,)),
    )
    @test_throws ErrorException spatial_volume_diagnostics(
        magnetic_small,
        magnetic_large;
        mass=2.0,
    )
end
