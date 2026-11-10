#!/usr/bin/env julia

using HALQCD
using Printf
using TOML

function scalar_symmetric_relative_error(first, second)
    numerator = abs(first - second)
    denominator = max(abs(first), abs(second), eps(Float64))
    return numerator / denominator
end

function field_symmetric_relative_error(first, second)
    size(first) == size(second) ||
        throw(DimensionMismatch("comparison fields must have the same shape"))
    isempty(first) &&
        throw(ArgumentError("comparison fields must not be empty"))
    numerator = maximum(abs, first .- second)
    denominator = max(
        maximum(abs, first),
        maximum(abs, second),
        eps(Float64),
    )
    return numerator / denominator
end

function validate_volume_comparison_metadata(first, second)
    first.sample_kind == second.sample_kind ||
        error("dataset sample kinds differ")
    for key in (
        :channel,
        :spin,
        :source_operator,
        :sink_operator,
        :temporal_boundary_condition,
        :lattice_spacing,
        :magnetic_flux,
        :gauge_condition,
        :kappa_light,
        :kappa_strange,
        :solver_squared_residual_threshold,
        :source_positions,
        :center_of_mass_reduction,
        :relative_coordinate_convention,
        :one_hadron_spin_reduction,
        :sample_ids,
        :source_ids,
    )
        getproperty(first.metadata, key) == getproperty(second.metadata, key) ||
            error("dataset metadata differs at $key")
    end
    first.metadata.magnetic_flux == 0 ||
        error("ordinary spatial volume diagnostics require zero magnetic flux")
    size(first.four_point)[4:6] == size(second.four_point)[4:6] ||
        error("time, source, and sample axes must agree")
    first_spatial_shape = size(first.four_point)[1:3]
    second_spatial_shape = size(second.four_point)[1:3]
    all(==(first_spatial_shape[1]), first_spatial_shape) ||
        error("first spatial volume must be cubic")
    all(==(second_spatial_shape[1]), second_spatial_shape) ||
        error("second spatial volume must be cubic")
    prod(first_spatial_shape) < prod(second_spatial_shape) ||
        error("the first dataset must have the smaller spatial volume")
    return first_spatial_shape, second_spatial_shape
end

function normalized_volume_slice(dataset, time, source, sample; atol)
    return r_correlator(
        view(dataset.four_point, :, :, :, time, source, sample),
        dataset.one_hadron_1[time, source, sample],
        dataset.one_hadron_2[time, source, sample];
        atol,
    )
end

function spatial_volume_diagnostics(
    first,
    second;
    mass,
    source=1,
    sample=1,
    normalization_atol=0.0,
    potential_rtol=sqrt(eps(Float64)),
)
    isfinite(mass) && mass > 0 ||
        throw(ArgumentError("mass must be finite and positive"))
    normalization_atol >= 0 ||
        throw(ArgumentError("normalization_atol must be nonnegative"))
    isfinite(potential_rtol) && potential_rtol >= 0 ||
        throw(ArgumentError("potential_rtol must be finite and nonnegative"))
    first_shape, second_shape =
        validate_volume_comparison_metadata(first, second)
    checkbounds(first.one_hadron_1, 1, source, sample)
    checkbounds(second.one_hadron_1, 1, source, sample)

    first_mask = normalization_mask(first; atol=normalization_atol)
    second_mask = normalization_mask(second; atol=normalization_atol)
    nt = size(first.four_point, 4)
    common_stencil_times = [
        time for time in 2:(nt - 1) if
        all(first_mask[t, source, sample] for t in (time - 1):(time + 1)) &&
        all(second_mask[t, source, sample] for t in (time - 1):(time + 1))
    ]
    isempty(common_stencil_times) &&
        error("no common interior time has a normalizable three-slice stencil")

    per_time = Dict{String,Any}()
    first_volume = prod(first_shape)
    second_volume = prod(second_shape)
    origin_r_errors = Float64[]
    origin_potential_errors = Float64[]
    common_r_errors = Float64[]
    volume_rescaled_origin_r_errors = Float64[]
    volume_rescaled_common_r_errors = Float64[]
    common_r_shape_errors = Float64[]
    common_potential_errors = Float64[]
    for time in common_stencil_times
        first_r = normalized_volume_slice(
            first,
            time,
            source,
            sample;
            atol=normalization_atol,
        )
        second_r = normalized_volume_slice(
            second,
            time,
            source,
            sample;
            atol=normalization_atol,
        )
        first_result = lo_potential_equal_mass(
            first,
            time;
            source,
            sample,
            normalization_atol,
            mass,
            spacing=Tuple(first.metadata.lattice_spacing[1:3]),
            at=first.metadata.lattice_spacing[4],
            rtol=potential_rtol,
        )
        second_result = lo_potential_equal_mass(
            second,
            time;
            source,
            sample,
            normalization_atol,
            mass,
            spacing=Tuple(second.metadata.lattice_spacing[1:3]),
            at=second.metadata.lattice_spacing[4],
            rtol=potential_rtol,
        )
        origin = CartesianIndex(1, 1, 1)
        first_result.valid[origin] && second_result.valid[origin] ||
            error("origin is not valid at time $time")
        origin_r_error =
            scalar_symmetric_relative_error(first_r[origin], second_r[origin])
        origin_potential_error = scalar_symmetric_relative_error(
            first_result.potential[origin],
            second_result.potential[origin],
        )
        push!(origin_r_errors, origin_r_error)
        push!(origin_potential_errors, origin_potential_error)
        first_valid = first_result.potential[first_result.valid]
        second_valid = second_result.potential[second_result.valid]
        common_ranges = ntuple(axis -> axes(first_r, axis), 3)
        second_common_r = view(second_r, common_ranges...)
        common_r_error =
            field_symmetric_relative_error(first_r, second_common_r)
        volume_rescaled_origin_r_error =
            scalar_symmetric_relative_error(
                first_volume * first_r[origin],
                second_volume * second_r[origin],
            )
        volume_rescaled_common_r_error =
            field_symmetric_relative_error(
                first_volume .* first_r,
                second_volume .* second_common_r,
            )
        first_common_scale = maximum(abs, first_r)
        second_common_scale = maximum(abs, second_common_r)
        first_common_scale > eps(Float64) ||
            error("first common R cube is numerically zero at time $time")
        second_common_scale > eps(Float64) ||
            error("second common R cube is numerically zero at time $time")
        common_r_shape_error = field_symmetric_relative_error(
            first_r ./ first_common_scale,
            second_common_r ./ second_common_scale,
        )
        second_common_valid =
            view(second_result.valid, common_ranges...)
        common_valid = first_result.valid .& second_common_valid
        any(common_valid) ||
            error("no common spatial site has a valid potential at time $time")
        first_common_potential = first_result.potential[common_valid]
        second_common_potential =
            view(second_result.potential, common_ranges...)[common_valid]
        common_potential_error = field_symmetric_relative_error(
            first_common_potential,
            second_common_potential,
        )
        push!(common_r_errors, common_r_error)
        push!(
            volume_rescaled_origin_r_errors,
            volume_rescaled_origin_r_error,
        )
        push!(
            volume_rescaled_common_r_errors,
            volume_rescaled_common_r_error,
        )
        push!(common_r_shape_errors, common_r_shape_error)
        push!(common_potential_errors, common_potential_error)
        per_time["time_$time"] = Dict(
            "first_origin_r_real" => real(first_r[origin]),
            "first_origin_r_imag" => imag(first_r[origin]),
            "second_origin_r_real" => real(second_r[origin]),
            "second_origin_r_imag" => imag(second_r[origin]),
            "origin_r_relative_error" => origin_r_error,
            "first_origin_potential_real" =>
                real(first_result.potential[origin]),
            "first_origin_potential_imag" =>
                imag(first_result.potential[origin]),
            "second_origin_potential_real" =>
                real(second_result.potential[origin]),
            "second_origin_potential_imag" =>
                imag(second_result.potential[origin]),
            "origin_potential_relative_error" => origin_potential_error,
            "common_cube_r_relative_error" => common_r_error,
            "volume_rescaled_origin_r_relative_error" =>
                volume_rescaled_origin_r_error,
            "volume_rescaled_common_cube_r_relative_error" =>
                volume_rescaled_common_r_error,
            "common_cube_r_shape_relative_error" =>
                common_r_shape_error,
            "common_cube_potential_relative_error" =>
                common_potential_error,
            "common_cube_valid_sites" => count(common_valid),
            "first_maximum_abs_r" => maximum(abs, first_r),
            "second_maximum_abs_r" => maximum(abs, second_r),
            "first_r_root_mean_square" =>
                sqrt(sum(abs2, first_r) / length(first_r)),
            "second_r_root_mean_square" =>
                sqrt(sum(abs2, second_r) / length(second_r)),
            "first_maximum_abs_potential" => maximum(abs, first_valid),
            "second_maximum_abs_potential" => maximum(abs, second_valid),
            "first_valid_sites" => count(first_result.valid),
            "second_valid_sites" => count(second_result.valid),
        )
    end

    return Dict(
        "first_spatial_shape" => collect(first_shape),
        "second_spatial_shape" => collect(second_shape),
        "common_stencil_times" => common_stencil_times,
        "maximum_origin_r_relative_error" => maximum(origin_r_errors),
        "maximum_origin_potential_relative_error" =>
            maximum(origin_potential_errors),
        "maximum_common_cube_r_relative_error" =>
            maximum(common_r_errors),
        "maximum_volume_rescaled_origin_r_relative_error" =>
            maximum(volume_rescaled_origin_r_errors),
        "maximum_volume_rescaled_common_cube_r_relative_error" =>
            maximum(volume_rescaled_common_r_errors),
        "maximum_common_cube_r_shape_relative_error" =>
            maximum(common_r_shape_errors),
        "maximum_common_cube_potential_relative_error" =>
            maximum(common_potential_errors),
        "per_time" => per_time,
    )
end

function compare_spatial_nbs_volumes_main()
    length(ARGS) == 3 ||
        error(
            "usage: julia compare_spatial_nbs_volumes.jl " *
            "SMALL_DATASET LARGE_DATASET OUTPUT_TOML",
        )
    first_path = abspath(ARGS[1])
    second_path = abspath(ARGS[2])
    output_path = abspath(ARGS[3])
    !ispath(output_path) ||
        error("refusing to overwrite output path: $output_path")

    first = read_nbs_dataset(first_path)
    second = read_nbs_dataset(second_path)
    normalization_atol = parse(
        Float64,
        get(ENV, "HALQCD_NBS_NORMALIZATION_ATOL", "0"),
    )
    potential_rtol = parse(
        Float64,
        get(ENV, "HALQCD_POTENTIAL_RTOL", string(sqrt(eps(Float64)))),
    )
    default_mass =
        free_wilson_quark_pole_mass(first.metadata.kappa_light) +
        2 * free_wilson_quark_pole_mass(first.metadata.kappa_strange)
    mass = parse(
        Float64,
        get(ENV, "HALQCD_BARYON_MASS", string(default_mass)),
    )
    source = parse(Int, get(ENV, "HALQCD_POTENTIAL_SOURCE", "1"))
    sample = parse(Int, get(ENV, "HALQCD_POTENTIAL_SAMPLE", "1"))
    diagnostics = spatial_volume_diagnostics(
        first,
        second;
        mass,
        source,
        sample,
        normalization_atol,
        potential_rtol,
    )
    repository_root = normpath(joinpath(@__DIR__, ".."))
    diagnostics["format"] = "HALQCD.jl spatial NBS volume diagnostic"
    diagnostics["format_version"] = "1.0.0"
    diagnostics["first_dataset"] = first_path
    diagnostics["second_dataset"] = second_path
    diagnostics["first_manifest_sha256"] =
        strip(read(joinpath(first_path, "manifest.sha256"), String))
    diagnostics["second_manifest_sha256"] =
        strip(read(joinpath(second_path, "manifest.sha256"), String))
    diagnostics["first_dataset_commit"] = first.metadata.code_commit
    diagnostics["second_dataset_commit"] = second.metadata.code_commit
    diagnostics["analyzer_commit"] =
        readchomp(`git -C $repository_root rev-parse HEAD`)
    diagnostics["mass"] = mass
    diagnostics["source"] = source
    diagnostics["sample"] = sample
    diagnostics["normalization_atol"] = normalization_atol
    diagnostics["potential_rtol"] = potential_rtol
    temporary_output = output_path * ".tmp-$(getpid())"
    !ispath(temporary_output) ||
        error("temporary output path already exists: $temporary_output")
    open(temporary_output, "w") do io
        TOML.print(io, diagnostics; sorted=true)
    end
    mv(temporary_output, output_path)

    @printf("first_spatial_shape=%s\n", repr(Tuple(diagnostics["first_spatial_shape"])))
    @printf("second_spatial_shape=%s\n", repr(Tuple(diagnostics["second_spatial_shape"])))
    @printf(
        "maximum_origin_r_relative_error=%.17e\n",
        diagnostics["maximum_origin_r_relative_error"],
    )
    @printf(
        "maximum_origin_potential_relative_error=%.17e\n",
        diagnostics["maximum_origin_potential_relative_error"],
    )
    @printf(
        "maximum_common_cube_r_relative_error=%.17e\n",
        diagnostics["maximum_common_cube_r_relative_error"],
    )
    @printf(
        "maximum_volume_rescaled_origin_r_relative_error=%.17e\n",
        diagnostics["maximum_volume_rescaled_origin_r_relative_error"],
    )
    @printf(
        "maximum_volume_rescaled_common_cube_r_relative_error=%.17e\n",
        diagnostics[
            "maximum_volume_rescaled_common_cube_r_relative_error"
        ],
    )
    @printf(
        "maximum_common_cube_r_shape_relative_error=%.17e\n",
        diagnostics["maximum_common_cube_r_shape_relative_error"],
    )
    @printf(
        "maximum_common_cube_potential_relative_error=%.17e\n",
        diagnostics["maximum_common_cube_potential_relative_error"],
    )
    # This marker means that the diagnostic completed without an input or
    # arithmetic error.  It deliberately does not claim volume stability.
    println("volume_diagnostic_completed=true")
end

if abspath(PROGRAM_FILE) == @__FILE__
    compare_spatial_nbs_volumes_main()
end
