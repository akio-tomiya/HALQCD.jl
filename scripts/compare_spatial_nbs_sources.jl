#!/usr/bin/env julia

using HALQCD
using Printf
using TOML

function temporal_shift_mapping(sink_time, source_shift, nt)
    nt > 0 || throw(ArgumentError("nt must be positive"))
    1 <= sink_time <= nt ||
        throw(BoundsError(1:nt, sink_time))
    raw_time = sink_time - source_shift
    winding = fld(raw_time - 1, nt)
    mapped_time = mod1(raw_time, nt)
    three_quark_boundary_sign = isodd(winding) ? -1 : 1
    return mapped_time, three_quark_boundary_sign
end

function comparison_relative_error(first, second)
    size(first) == size(second) ||
        throw(DimensionMismatch("comparison operands must have the same shape"))
    return maximum(abs, first .- second) /
        max(maximum(abs, first), maximum(abs, second), eps())
end

function complex_value_summary(value)
    return Dict(
        "real" => Float64(real(value)),
        "imaginary" => Float64(imag(value)),
        "absolute_value" => Float64(abs(value)),
    )
end

function potential_component_vector(components, index)
    return (
        components.spatial[index],
        components.first_time[index],
        components.second_time[index],
    )
end

function potential_decomposition_diagnostic(total, component_values)
    component_sum = sum(component_values)
    component_l1 = sum(abs, component_values)
    closure_absolute_error = abs(total - component_sum)
    closure_scale = max(abs(total), component_l1, eps(Float64))
    closure_relative_error = closure_absolute_error / closure_scale
    closure_tolerance = 256 * eps(Float64)
    closure_relative_error <= closure_tolerance ||
        error(
            "potential component decomposition does not close: " *
            "relative error $closure_relative_error exceeds " *
            "$closure_tolerance",
        )
    cancellation_factor = if iszero(abs(total))
        iszero(component_l1) ? 1.0 : Inf
    else
        component_l1 / abs(total)
    end
    return Dict(
        "component_l1_norm" => component_l1,
        "component_sum" => complex_value_summary(component_sum),
        "closure_absolute_error" => closure_absolute_error,
        "closure_relative_error" => closure_relative_error,
        "cancellation_factor" => cancellation_factor,
    )
end

function potential_difference_amplification_diagnostic(
    first_total,
    second_total,
    first_component_values,
    second_component_values,
)
    first_decomposition = potential_decomposition_diagnostic(
        first_total,
        first_component_values,
    )
    second_decomposition = potential_decomposition_diagnostic(
        second_total,
        second_component_values,
    )
    component_difference_l1 = sum(
        abs(first - second) for
        (first, second) in zip(
            first_component_values,
            second_component_values,
        )
    )
    potential_absolute_difference = abs(first_total - second_total)
    component_scale = max(
        first_decomposition["component_l1_norm"],
        second_decomposition["component_l1_norm"],
        eps(Float64),
    )
    difference_closure_absolute_error = abs(
        (first_total - second_total) -
        sum(
            first - second for
            (first, second) in zip(
                first_component_values,
                second_component_values,
            )
        ),
    )
    difference_scale = max(
        potential_absolute_difference,
        component_difference_l1,
        eps(Float64),
    )
    difference_closure_relative_to_difference_scale =
        difference_closure_absolute_error / difference_scale
    difference_closure_relative_to_component_scale =
        difference_closure_absolute_error / component_scale
    closure_tolerance = 512 * eps(Float64)
    difference_closure_relative_to_component_scale <= closure_tolerance ||
        error(
            "potential difference does not close through component " *
            "differences at the scale of the original components: " *
            "relative error " *
            "$difference_closure_relative_to_component_scale exceeds " *
            "$closure_tolerance",
        )

    potential_scale = max(abs(first_total), abs(second_total), eps(Float64))
    component_relative_perturbation = component_difference_l1 / component_scale
    roundoff_relative_perturbation =
        difference_closure_absolute_error / component_scale
    effective_relative_perturbation =
        component_relative_perturbation + roundoff_relative_perturbation
    potential_relative_perturbation =
        potential_absolute_difference / potential_scale
    relative_error_amplification = if iszero(component_relative_perturbation)
        iszero(potential_relative_perturbation) ? 0.0 : Inf
    else
        potential_relative_perturbation / component_relative_perturbation
    end
    roundoff_aware_relative_error_amplification = if iszero(
        effective_relative_perturbation,
    )
        iszero(potential_relative_perturbation) ? 0.0 : Inf
    else
        potential_relative_perturbation / effective_relative_perturbation
    end
    raw_triangle_bound_ratio = if iszero(component_difference_l1)
        iszero(potential_absolute_difference) ? 0.0 : Inf
    else
        potential_absolute_difference / component_difference_l1
    end
    roundoff_aware_triangle_bound =
        component_difference_l1 + difference_closure_absolute_error
    triangle_bound_ratio = if iszero(roundoff_aware_triangle_bound)
        iszero(potential_absolute_difference) ? 0.0 : Inf
    else
        potential_absolute_difference / roundoff_aware_triangle_bound
    end
    triangle_tolerance = 512 * eps(Float64)
    triangle_bound_ratio <= 1 + triangle_tolerance ||
        error(
            "potential difference exceeds the component-difference " *
            "triangle bound: ratio $triangle_bound_ratio",
        )

    cancellation_upper_bound = max(
        first_decomposition["cancellation_factor"],
        second_decomposition["cancellation_factor"],
    )
    roundoff_aware_relative_error_amplification <=
        cancellation_upper_bound * (1 + triangle_tolerance) ||
        error(
            "roundoff-aware relative-error amplification exceeds the " *
            "cancellation bound",
        )

    return Dict(
        "first_decomposition" => first_decomposition,
        "second_decomposition" => second_decomposition,
        "component_difference_l1_norm" => component_difference_l1,
        "potential_absolute_difference" => potential_absolute_difference,
        "difference_closure_absolute_error" =>
            difference_closure_absolute_error,
        "difference_closure_relative_to_difference_scale" =>
            difference_closure_relative_to_difference_scale,
        "difference_closure_relative_to_component_scale" =>
            difference_closure_relative_to_component_scale,
        "component_relative_perturbation" =>
            component_relative_perturbation,
        "roundoff_relative_perturbation" =>
            roundoff_relative_perturbation,
        "effective_relative_perturbation" =>
            effective_relative_perturbation,
        "potential_relative_perturbation" =>
            potential_relative_perturbation,
        "relative_error_amplification" => relative_error_amplification,
        "roundoff_aware_relative_error_amplification" =>
            roundoff_aware_relative_error_amplification,
        "cancellation_upper_bound" => cancellation_upper_bound,
        "raw_triangle_bound_ratio" => raw_triangle_bound_ratio,
        "triangle_bound_ratio" => triangle_bound_ratio,
    )
end

function maximum_potential_difference_diagnostic(
    first_potential,
    second_potential,
    first_components,
    second_components,
)
    first_potential.valid == second_potential.valid ||
        error("potential valid masks differ")
    valid_indices = findall(first_potential.valid)
    isempty(valid_indices) && error("potential comparison has no valid site")
    absolute_differences = [
        abs(
            first_potential.potential[index] -
            second_potential.potential[index],
        ) for index in valid_indices
    ]
    maximum_position = argmax(absolute_differences)
    index = valid_indices[maximum_position]
    first_component_values = potential_component_vector(
        first_components,
        index,
    )
    second_component_values = potential_component_vector(
        second_components,
        index,
    )
    return Dict(
        "zero_based_displacement" => collect(Tuple(index)) .- 1,
        "maximum_absolute_difference" => absolute_differences[maximum_position],
        "first_potential" =>
            complex_value_summary(first_potential.potential[index]),
        "second_potential" =>
            complex_value_summary(second_potential.potential[index]),
        "first_components" => Dict(
            "spatial" => complex_value_summary(first_components.spatial[index]),
            "first_time" =>
                complex_value_summary(first_components.first_time[index]),
            "second_time" =>
                complex_value_summary(first_components.second_time[index]),
        ),
        "second_components" => Dict(
            "spatial" => complex_value_summary(second_components.spatial[index]),
            "first_time" =>
                complex_value_summary(second_components.first_time[index]),
            "second_time" =>
                complex_value_summary(second_components.second_time[index]),
        ),
        "cancellation_diagnostic" =>
            potential_difference_amplification_diagnostic(
                first_potential.potential[index],
                second_potential.potential[index],
                first_component_values,
                second_component_values,
            ),
    )
end

function validate_source_comparison_metadata(
    first,
    second;
    allow_unrecorded_six_quark_contraction=false,
    allow_solver_threshold_mismatch=false,
    allow_solver_algorithm_mismatch=false,
    expect_gauge_equivalence=false,
)
    allow_solver_threshold_mismatch && expect_gauge_equivalence &&
        error(
            "solver-threshold and gauge-equivalence diagnostics must be isolated",
        )
    allow_solver_algorithm_mismatch && expect_gauge_equivalence &&
        error(
            "solver-algorithm and gauge-equivalence diagnostics must be isolated",
        )
    allow_solver_algorithm_mismatch && allow_solver_threshold_mismatch &&
        error(
            "solver-algorithm and solver-threshold diagnostics must be isolated",
        )
    size(first.four_point) == size(second.four_point) ||
        error("dataset shapes differ")
    first.sample_kind == second.sample_kind ||
        error("dataset sample kinds differ")
    for key in (
        :channel,
        :spin,
        :source_operator,
        :sink_operator,
        :temporal_boundary_condition,
        :lattice_shape,
        :lattice_spacing,
        :magnetic_flux,
        :kappa_light,
        :kappa_strange,
        :source_ids,
        :julia_version,
        :center_of_mass_reduction,
        :relative_coordinate_convention,
        :one_hadron_spin_reduction,
        :normalization_atol,
        :normalization_policy,
    )
        first_value = getproperty(first.metadata, key)
        second_value = getproperty(second.metadata, key)
        first_value == second_value ||
            error("dataset metadata differs at $key")
    end
    first_solver_threshold =
        first.metadata.solver_squared_residual_threshold
    second_solver_threshold =
        second.metadata.solver_squared_residual_threshold
    if first_solver_threshold != second_solver_threshold
        allow_solver_threshold_mismatch ||
            error(
                "dataset metadata differs at " *
                "solver_squared_residual_threshold",
            )
    elseif allow_solver_threshold_mismatch
        error(
            "solver-threshold mismatch diagnostic requires different thresholds",
        )
    end
    first_solver_algorithm = recorded_solver_algorithm(first)
    second_solver_algorithm = recorded_solver_algorithm(second)
    if first_solver_algorithm != second_solver_algorithm
        allow_solver_algorithm_mismatch ||
            error("dataset metadata differs at solver_algorithm")
    elseif allow_solver_algorithm_mismatch
        error(
            "solver-algorithm mismatch diagnostic requires different algorithms",
        )
    end
    first_gauge = first.metadata.gauge_condition
    second_gauge = second.metadata.gauge_condition
    if first_gauge != second_gauge
        expect_gauge_equivalence ||
            error("dataset metadata differs at gauge_condition")
        Set((first_gauge, second_gauge)) == Set(("cold", "pure-gauge")) ||
            error(
                "gauge-equivalence comparison requires cold and pure-gauge datasets",
            )
        pure_gauge_metadata = first_gauge == "pure-gauge" ?
            first.metadata : second.metadata
        haskey(pure_gauge_metadata, :gauge_transformation) ||
            error("pure-gauge dataset does not record gauge_transformation")
        pure_gauge_metadata.gauge_transformation ==
            "deterministic local SU(3) transform of cold links" ||
            error("unexpected pure-gauge transformation metadata")
    elseif expect_gauge_equivalence
        error("gauge-equivalence comparison requires different gauge conditions")
    end
    if first_gauge == second_gauge
        first.metadata.sample_ids == second.metadata.sample_ids ||
            error("dataset metadata differs at sample_ids")
    end
    first.metadata.magnetic_flux == 0 ||
        error("ordinary source-time diagnostics require zero magnetic flux")
    first_positions = recorded_source_positions(first)
    second_positions = recorded_source_positions(second)
    first_spatial_positions =
        map(position -> position[1:3], first_positions)
    second_spatial_positions =
        map(position -> position[1:3], second_positions)
    first_spatial_positions == second_spatial_positions ||
        error("dataset spatial source positions differ")
    first_records_contraction =
        haskey(first.metadata, :six_quark_contraction)
    second_records_contraction =
        haskey(second.metadata, :six_quark_contraction)
    if first_records_contraction && second_records_contraction
        first.metadata.six_quark_contraction ==
            second.metadata.six_quark_contraction ||
            error("dataset metadata differs at six_quark_contraction")
    elseif !allow_unrecorded_six_quark_contraction
        error(
            "six_quark_contraction is unrecorded in at least one dataset; " *
            "set HALQCD_ALLOW_UNRECORDED_SIX_QUARK_CONTRACTION=true only " *
            "after an external provenance audit",
        )
    end
    return nothing
end

function comparison_environment_boolean(name, default=false)
    raw_value = lowercase(strip(get(ENV, name, string(default))))
    raw_value == "true" && return true
    raw_value == "false" && return false
    error("$name must be true or false")
end

function recorded_solver_algorithm(dataset)
    algorithm = get(dataset.metadata, :solver_algorithm, "bicg")
    algorithm isa AbstractString && !isempty(strip(algorithm)) ||
        error("metadata.solver_algorithm must be a nonempty string")
    normalized = lowercase(strip(algorithm))
    normalized in ("bicg", "dense-lu") ||
        error("unsupported metadata.solver_algorithm $algorithm")
    return normalized
end

function recorded_source_positions(dataset)
    haskey(dataset.metadata, :source_positions) ||
        error("dataset metadata does not record source_positions")
    positions = NTuple{4,Int}[]
    for position in dataset.metadata.source_positions
        position isa AbstractString ||
            error("metadata.source_positions entries must be strings")
        fields = split(strip(position, ['(', ')']), ',')
        length(fields) == 4 ||
            error("cannot parse source position $position")
        coordinates = Tuple(parse(Int, strip(field)) for field in fields)
        push!(positions, coordinates)
    end
    isempty(positions) &&
        error("metadata.source_positions must not be empty")
    lattice_shape = size(dataset.four_point)[1:4]
    all(
        all(
            1 <= position[dimension] <= lattice_shape[dimension] for
            dimension in 1:4
        )
        for position in positions
    ) || error(
        "metadata.source_positions contains a coordinate outside the lattice",
    )
    return positions
end

function source_time_from_positions(positions)
    source_times = map(position -> position[4], positions)
    all(==(first(source_times)), source_times) ||
        error("all source positions must have the same source time")
    return first(source_times)
end

function recorded_source_time(dataset)
    has_explicit_time = haskey(dataset.metadata, :source_time)
    has_positions = haskey(dataset.metadata, :source_positions)
    has_explicit_time || has_positions ||
        error("dataset metadata does not record source_time or source_positions")

    explicit_time = if has_explicit_time
        value = dataset.metadata.source_time
        value isa Integer && value > 0 ||
            error("metadata.source_time must be a positive integer")
        Int(value)
    else
        nothing
    end
    positions_time = has_positions ?
        source_time_from_positions(recorded_source_positions(dataset)) :
        nothing
    if explicit_time !== nothing && positions_time !== nothing
        explicit_time == positions_time ||
            error("metadata.source_time disagrees with source_positions")
    end
    source_time =
        explicit_time === nothing ? something(positions_time) : explicit_time
    source_time <= size(dataset.four_point, 4) ||
        error("recorded source time lies outside the temporal lattice")
    return source_time
end

function compare_spatial_nbs_sources_main()
    length(ARGS) == 5 ||
        error(
            "usage: julia compare_spatial_nbs_sources.jl " *
            "FIRST_DATASET FIRST_SOURCE_TIME SECOND_DATASET " *
            "SECOND_SOURCE_TIME OUTPUT_TOML",
        )
    first_path = abspath(ARGS[1])
    first_source_time = parse(Int, ARGS[2])
    second_path = abspath(ARGS[3])
    second_source_time = parse(Int, ARGS[4])
    output_path = abspath(ARGS[5])
    !ispath(output_path) ||
        error("refusing to overwrite output path: $output_path")

    first = read_nbs_dataset(first_path)
    second = read_nbs_dataset(second_path)
    allow_unrecorded_six_quark_contraction =
        comparison_environment_boolean(
            "HALQCD_ALLOW_UNRECORDED_SIX_QUARK_CONTRACTION",
        )
    expect_gauge_equivalence = comparison_environment_boolean(
        "HALQCD_EXPECT_GAUGE_EQUIVALENCE",
    )
    allow_solver_threshold_mismatch = comparison_environment_boolean(
        "HALQCD_ALLOW_SOLVER_THRESHOLD_MISMATCH",
    )
    allow_solver_algorithm_mismatch = comparison_environment_boolean(
        "HALQCD_ALLOW_SOLVER_ALGORITHM_MISMATCH",
    )
    validate_source_comparison_metadata(
        first,
        second;
        allow_unrecorded_six_quark_contraction,
        allow_solver_threshold_mismatch,
        allow_solver_algorithm_mismatch,
        expect_gauge_equivalence,
    )
    recorded_source_time(first) == first_source_time ||
        error("FIRST_SOURCE_TIME disagrees with dataset metadata")
    recorded_source_time(second) == second_source_time ||
        error("SECOND_SOURCE_TIME disagrees with dataset metadata")

    nt = size(first.four_point, 4)
    all(time -> 1 <= time <= nt, (first_source_time, second_source_time)) ||
        error("source times must lie inside the temporal lattice")
    source_shift = second_source_time - first_source_time
    expect_gauge_equivalence && source_shift != 0 &&
        error("gauge-equivalence comparison requires equal source times")
    allow_solver_threshold_mismatch && source_shift != 0 &&
        error("solver-threshold diagnostic requires equal source times")
    allow_solver_algorithm_mismatch && source_shift != 0 &&
        error("solver-algorithm diagnostic requires equal source times")
    tolerance = parse(
        Float64,
        get(ENV, "HALQCD_SOURCE_COMPARISON_RTOL", "5e-11"),
    )
    isfinite(tolerance) && tolerance >= 0 ||
        error("HALQCD_SOURCE_COMPARISON_RTOL must be finite and nonnegative")
    normalization_atol = parse(
        Float64,
        get(ENV, "HALQCD_NBS_NORMALIZATION_ATOL", "0"),
    )
    normalization_atol >= 0 ||
        error("HALQCD_NBS_NORMALIZATION_ATOL must be nonnegative")
    potential_rtol = parse(
        Float64,
        get(ENV, "HALQCD_POTENTIAL_RTOL", string(sqrt(eps(Float64)))),
    )
    isfinite(potential_rtol) && potential_rtol >= 0 ||
        error("HALQCD_POTENTIAL_RTOL must be finite and nonnegative")
    default_mass =
        free_wilson_quark_pole_mass(first.metadata.kappa_light) +
        2 * free_wilson_quark_pole_mass(first.metadata.kappa_strange)
    baryon_mass = parse(
        Float64,
        get(ENV, "HALQCD_BARYON_MASS", string(default_mass)),
    )
    isfinite(baryon_mass) && baryon_mass > 0 ||
        error("HALQCD_BARYON_MASS must be finite and positive")

    first_mask = normalization_mask(first; atol=normalization_atol)
    second_mask = normalization_mask(second; atol=normalization_atol)
    per_time = Dict{String,Any}()
    four_point_errors = Float64[]
    first_hadron_errors = Float64[]
    second_hadron_errors = Float64[]
    normalized_errors = Float64[]

    for sink_time in 1:nt
        mapped_time, hadron_sign = temporal_shift_mapping(
            sink_time,
            source_shift,
            nt,
        )
        two_hadron_sign = hadron_sign^2
        first_four_point =
            view(first.four_point, :, :, :, mapped_time, 1, 1)
        second_four_point =
            view(second.four_point, :, :, :, sink_time, 1, 1)
        four_point_error = comparison_relative_error(
            two_hadron_sign .* first_four_point,
            second_four_point,
        )
        first_hadron_error = comparison_relative_error(
            fill(
                hadron_sign * first.one_hadron_1[mapped_time, 1, 1],
                1,
            ),
            fill(second.one_hadron_1[sink_time, 1, 1], 1),
        )
        second_hadron_error = comparison_relative_error(
            fill(
                hadron_sign * first.one_hadron_2[mapped_time, 1, 1],
                1,
            ),
            fill(second.one_hadron_2[sink_time, 1, 1], 1),
        )
        mask_agrees =
            first_mask[mapped_time, 1, 1] ==
            second_mask[sink_time, 1, 1]
        mask_agrees ||
            error("normalization availability does not shift at time $sink_time")

        normalized_error = if first_mask[mapped_time, 1, 1]
            first_r = r_correlator(
                first_four_point,
                first.one_hadron_1[mapped_time, 1, 1],
                first.one_hadron_2[mapped_time, 1, 1];
                atol=normalization_atol,
            )
            second_r = r_correlator(
                second_four_point,
                second.one_hadron_1[sink_time, 1, 1],
                second.one_hadron_2[sink_time, 1, 1];
                atol=normalization_atol,
            )
            comparison_relative_error(first_r, second_r)
        else
            NaN
        end

        push!(four_point_errors, four_point_error)
        push!(first_hadron_errors, first_hadron_error)
        push!(second_hadron_errors, second_hadron_error)
        isfinite(normalized_error) && push!(normalized_errors, normalized_error)
        per_time["sink_time_$sink_time"] = Dict{String,Any}(
            "mapped_first_sink_time" => mapped_time,
            "three_quark_boundary_sign" => hadron_sign,
            "two_hadron_boundary_sign" => two_hadron_sign,
            "four_point_relative_error" => four_point_error,
            "one_hadron_1_relative_error" => first_hadron_error,
            "one_hadron_2_relative_error" => second_hadron_error,
            "normalization_available" =>
                second_mask[sink_time, 1, 1],
            "normalized_r_relative_error" => normalized_error,
        )
    end

    isempty(normalized_errors) &&
        error("no normalized time slice is available for comparison")
    potential_errors = Float64[]
    potential_spatial_errors = Float64[]
    potential_first_time_errors = Float64[]
    potential_second_time_errors = Float64[]
    potential_pairs = Vector{Vector{Int}}()
    for second_time in 2:(nt - 1)
        first_time, _ = temporal_shift_mapping(
            second_time,
            source_shift,
            nt,
        )
        2 <= first_time <= nt - 1 || continue
        first_stencil_available = all(
            first_mask[time, 1, 1] for time in (first_time - 1):(first_time + 1)
        )
        second_stencil = (second_time - 1):(second_time + 1)
        second_stencil_available = all(
            second_mask[time, 1, 1] for time in second_stencil
        )
        first_stencil_available && second_stencil_available || continue

        first_potential = lo_potential_equal_mass(
            first,
            first_time;
            mass=baryon_mass,
            rtol=potential_rtol,
            normalization_atol,
            spacing=Tuple(first.metadata.lattice_spacing[1:3]),
            at=first.metadata.lattice_spacing[4],
        )
        second_potential = lo_potential_equal_mass(
            second,
            second_time;
            mass=baryon_mass,
            rtol=potential_rtol,
            normalization_atol,
            spacing=Tuple(second.metadata.lattice_spacing[1:3]),
            at=second.metadata.lattice_spacing[4],
        )
        first_potential.valid == second_potential.valid ||
            error(
                "potential valid masks differ for first time $first_time " *
                "and second time $second_time",
            )
        first_r = r_correlator(
            view(first.four_point, :, :, :, first_time, 1, 1),
            first.one_hadron_1[first_time, 1, 1],
            first.one_hadron_2[first_time, 1, 1];
            atol=normalization_atol,
        )
        second_r = r_correlator(
            view(second.four_point, :, :, :, second_time, 1, 1),
            second.one_hadron_1[second_time, 1, 1],
            second.one_hadron_2[second_time, 1, 1];
            atol=normalization_atol,
        )
        first_components = divided_potential_components(
            first_potential,
            first_r,
        )
        second_components = divided_potential_components(
            second_potential,
            second_r,
        )
        potential_error = comparison_relative_error(
            first_potential.potential,
            second_potential.potential,
        )
        spatial_error = comparison_relative_error(
            first_components.spatial,
            second_components.spatial,
        )
        first_time_error = comparison_relative_error(
            first_components.first_time,
            second_components.first_time,
        )
        second_time_error = comparison_relative_error(
            first_components.second_time,
            second_components.second_time,
        )
        maximum_difference_diagnostic =
            maximum_potential_difference_diagnostic(
                first_potential,
                second_potential,
                first_components,
                second_components,
            )
        push!(potential_errors, potential_error)
        push!(potential_spatial_errors, spatial_error)
        push!(potential_first_time_errors, first_time_error)
        push!(potential_second_time_errors, second_time_error)
        push!(potential_pairs, [first_time, second_time])
        per_time["sink_time_$second_time"]["potential_comparison"] = Dict(
            "mapped_first_center_time" => first_time,
            "potential_relative_error" => potential_error,
            "spatial_component_relative_error" => spatial_error,
            "first_time_component_relative_error" => first_time_error,
            "second_time_component_relative_error" => second_time_error,
            "maximum_difference_diagnostic" => maximum_difference_diagnostic,
        )
    end
    isempty(potential_errors) &&
        error("no interior source-shifted potential stencil is comparable")
    maxima = Dict(
        "four_point_relative_error" => maximum(four_point_errors),
        "one_hadron_1_relative_error" => maximum(first_hadron_errors),
        "one_hadron_2_relative_error" => maximum(second_hadron_errors),
        "normalized_r_relative_error" => maximum(normalized_errors),
        "potential_relative_error" => maximum(potential_errors),
        "potential_spatial_component_relative_error" =>
            maximum(potential_spatial_errors),
        "potential_first_time_component_relative_error" =>
            maximum(potential_first_time_errors),
        "potential_second_time_component_relative_error" =>
            maximum(potential_second_time_errors),
    )
    repository_root = normpath(joinpath(@__DIR__, ".."))
    summary = Dict(
        "format" => "HALQCD.jl spatial NBS source-time comparison",
        "format_version" => "1.1.0",
        "first_dataset" => first_path,
        "first_manifest_sha256" =>
            strip(read(joinpath(first_path, "manifest.sha256"), String)),
        "first_dataset_commit" => first.metadata.code_commit,
        "first_source_time" => first_source_time,
        "second_dataset" => second_path,
        "second_manifest_sha256" =>
            strip(read(joinpath(second_path, "manifest.sha256"), String)),
        "second_dataset_commit" => second.metadata.code_commit,
        "second_source_time" => second_source_time,
        "source_shift" => source_shift,
        "gauge_equivalence_comparison" => expect_gauge_equivalence,
        "solver_threshold_mismatch_comparison" =>
            allow_solver_threshold_mismatch,
        "solver_algorithm_mismatch_comparison" =>
            allow_solver_algorithm_mismatch,
        "first_solver_algorithm" => recorded_solver_algorithm(first),
        "second_solver_algorithm" => recorded_solver_algorithm(second),
        "first_solver_squared_residual_threshold" =>
            first.metadata.solver_squared_residual_threshold,
        "second_solver_squared_residual_threshold" =>
            second.metadata.solver_squared_residual_threshold,
        "first_gauge_condition" => first.metadata.gauge_condition,
        "second_gauge_condition" => second.metadata.gauge_condition,
        "first_six_quark_contraction" => (
            haskey(first.metadata, :six_quark_contraction) ?
            first.metadata.six_quark_contraction : "unrecorded"
        ),
        "second_six_quark_contraction" => (
            haskey(second.metadata, :six_quark_contraction) ?
            second.metadata.six_quark_contraction : "unrecorded"
        ),
        "allowed_unrecorded_six_quark_contraction" =>
            allow_unrecorded_six_quark_contraction,
        "comparison_tolerance" => tolerance,
        "normalization_atol" => normalization_atol,
        "potential_rtol" => potential_rtol,
        "baryon_mass" => baryon_mass,
        "potential_comparison_pairs" => potential_pairs,
        "analyzer_commit" =>
            readchomp(`git -C $repository_root rev-parse HEAD`),
        "maxima" => maxima,
        "per_time" => per_time,
    )
    temporary_output = output_path * ".tmp-$(getpid())"
    !ispath(temporary_output) ||
        error("temporary output path already exists: $temporary_output")
    open(temporary_output, "w") do io
        TOML.print(io, summary; sorted=true)
    end
    mv(temporary_output, output_path)

    @printf("source_shift=%d\n", source_shift)
    for key in sort(collect(keys(maxima)))
        @printf("%s=%.17e\n", key, maxima[key])
    end
    all(value <= tolerance for value in values(maxima)) ||
        error("source-time comparison exceeds tolerance $tolerance")
    println("comparison_pass=true")
end

if abspath(PROGRAM_FILE) == @__FILE__
    compare_spatial_nbs_sources_main()
end
