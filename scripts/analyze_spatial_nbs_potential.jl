#!/usr/bin/env julia

using HALQCD
using Printf
using TOML

function symmetric_relative_error(first, second)
    size(first) == size(second) ||
        throw(DimensionMismatch("symmetry operands must have the same shape"))
    numerator = maximum(abs, first .- second)
    denominator = max(maximum(abs, first), maximum(abs, second), eps())
    return numerator / denominator
end

function normalized_nbs_slice(dataset, time, source, sample; atol)
    return r_correlator(
        view(dataset.four_point, :, :, :, time, source, sample),
        dataset.one_hadron_1[time, source, sample],
        dataset.one_hadron_2[time, source, sample];
        atol,
    )
end

function parse_selected_times(text, available_times)
    strip(text) == "" && return collect(available_times)
    selected = parse.(Int, split(text, ","))
    allunique(selected) ||
        throw(ArgumentError("HALQCD_POTENTIAL_TIMES must be unique"))
    all(time -> time in available_times, selected) ||
        throw(
            ArgumentError(
                "every requested potential time must have a fully " *
                "normalizable three-slice stencil",
            ),
        )
    return selected
end

function analyze_spatial_nbs_potential_main()
    length(ARGS) == 2 ||
        error(
            "usage: julia analyze_spatial_nbs_potential.jl " *
            "INPUT_DATASET OUTPUT_DIRECTORY",
        )
    input_dataset = abspath(ARGS[1])
    output_directory = abspath(ARGS[2])
    isdir(input_dataset) ||
        error("input dataset is not a directory: $input_dataset")
    !ispath(output_directory) ||
        error("refusing to overwrite output path: $output_directory")

    dataset = read_nbs_dataset(input_dataset)
    dataset.metadata.magnetic_flux == 0 ||
        error("this analyzer is restricted to zero magnetic flux")
    source = parse(Int, get(ENV, "HALQCD_POTENTIAL_SOURCE", "1"))
    sample = parse(Int, get(ENV, "HALQCD_POTENTIAL_SAMPLE", "1"))
    normalization_atol = parse(
        Float64,
        get(ENV, "HALQCD_NBS_NORMALIZATION_ATOL", "0"),
    )
    potential_rtol = parse(
        Float64,
        get(ENV, "HALQCD_POTENTIAL_RTOL", string(sqrt(eps(Float64)))),
    )
    normalization_atol >= 0 ||
        error("HALQCD_NBS_NORMALIZATION_ATOL must be nonnegative")
    isfinite(potential_rtol) && potential_rtol >= 0 ||
        error("HALQCD_POTENTIAL_RTOL must be finite and nonnegative")
    checkbounds(dataset.one_hadron_1, 1, source, sample)

    haskey(dataset.metadata, :kappa_light) ||
        error("dataset metadata does not contain kappa_light")
    haskey(dataset.metadata, :kappa_strange) ||
        error("dataset metadata does not contain kappa_strange")
    default_mass =
        free_wilson_quark_pole_mass(dataset.metadata.kappa_light) +
        2 * free_wilson_quark_pole_mass(dataset.metadata.kappa_strange)
    mass_was_overridden = haskey(ENV, "HALQCD_BARYON_MASS")
    baryon_mass = parse(
        Float64,
        get(ENV, "HALQCD_BARYON_MASS", string(default_mass)),
    )
    isfinite(baryon_mass) && baryon_mass > 0 ||
        error("HALQCD_BARYON_MASS must be finite and positive")

    nt = size(dataset.four_point, 4)
    available = normalization_mask(dataset; atol=normalization_atol)
    stencil_times = [
        time for time in 2:(nt - 1) if
        all(available[selected, source, sample] for selected in (time - 1):(time + 1))
    ]
    isempty(stencil_times) &&
        error("no interior time has a fully normalizable three-slice stencil")
    selected_times = parse_selected_times(
        get(ENV, "HALQCD_POTENTIAL_TIMES", ""),
        stencil_times,
    )

    temporary_directory = output_directory * ".tmp-$(getpid())"
    !ispath(temporary_directory) ||
        error("temporary output path already exists: $temporary_directory")
    mkpath(temporary_directory)
    table_path = joinpath(temporary_directory, "potential.tsv")
    summary_by_time = Dict{String,Any}()

    open(table_path, "w") do io
        println(
            io,
            join(
                (
                    "time",
                    "dx",
                    "dy",
                    "dz",
                    "r_squared",
                    "valid",
                    "r_real",
                    "r_imag",
                    "potential_real",
                    "potential_imag",
                    "spatial_real",
                    "spatial_imag",
                    "first_time_real",
                    "first_time_imag",
                    "second_time_real",
                    "second_time_imag",
                ),
                '\t',
            ),
        )
        for time in selected_times
            result = lo_potential_equal_mass(
                dataset,
                time;
                source,
                sample,
                normalization_atol,
                mass=baryon_mass,
                spacing=Tuple(dataset.metadata.lattice_spacing[1:3]),
                at=dataset.metadata.lattice_spacing[4],
                rtol=potential_rtol,
            )
            rzero = normalized_nbs_slice(
                dataset,
                time,
                source,
                sample;
                atol=normalization_atol,
            )
            components = divided_potential_components(result, rzero)
            nx, ny, nz = size(rzero)
            yz_error = if ny == nz
                symmetric_relative_error(
                    rzero,
                    permutedims(rzero, (1, 3, 2)),
                )
            else
                NaN
            end
            maximum_r_real = maximum(abs, real.(rzero))
            imaginary_to_real = maximum(abs, imag.(rzero)) /
                max(maximum_r_real, eps())
            valid_values = result.potential[result.valid]
            isempty(valid_values) &&
                error("potential time $time has no valid spatial sites")
            summary_by_time["time_$time"] = Dict(
                "valid_sites" => count(result.valid),
                "total_sites" => length(result.valid),
                "maximum_abs_potential" => maximum(abs, valid_values),
                "maximum_abs_spatial_component" =>
                    maximum(abs, components.spatial[result.valid]),
                "maximum_abs_first_time_component" =>
                    maximum(abs, components.first_time[result.valid]),
                "maximum_abs_second_time_component" =>
                    maximum(abs, components.second_time[result.valid]),
                "r_imaginary_to_real_maximum_ratio" => imaginary_to_real,
                "r_yz_exchange_symmetric_relative_error" => yz_error,
            )

            for z in 1:nz, y in 1:ny, x in 1:nx
                dx, dy, dz = x - 1, y - 1, z - 1
                @printf(
                    io,
                    "%d\t%d\t%d\t%d\t%d\t%s\t%.17e\t%.17e\t%.17e\t%.17e\t%.17e\t%.17e\t%.17e\t%.17e\t%.17e\t%.17e\n",
                    time,
                    dx,
                    dy,
                    dz,
                    dx^2 + dy^2 + dz^2,
                    string(result.valid[x, y, z]),
                    real(rzero[x, y, z]),
                    imag(rzero[x, y, z]),
                    real(result.potential[x, y, z]),
                    imag(result.potential[x, y, z]),
                    real(components.spatial[x, y, z]),
                    imag(components.spatial[x, y, z]),
                    real(components.first_time[x, y, z]),
                    imag(components.first_time[x, y, z]),
                    real(components.second_time[x, y, z]),
                    imag(components.second_time[x, y, z]),
                )
            end
        end
    end

    repository_root = normpath(joinpath(@__DIR__, ".."))
    analyzer_commit = readchomp(`git -C $repository_root rev-parse HEAD`)
    summary = Dict(
        "format" => "HALQCD.jl spatial NBS potential diagnostic",
        "format_version" => "1.0.0",
        "input_dataset" => input_dataset,
        "input_manifest_sha256" =>
            strip(read(joinpath(input_dataset, "manifest.sha256"), String)),
        "dataset_code_commit" => dataset.metadata.code_commit,
        "analyzer_code_commit" => analyzer_commit,
        "sample_kind" => string(dataset.sample_kind),
        "source" => source,
        "sample" => sample,
        "normalization_atol" => normalization_atol,
        "potential_rtol" => potential_rtol,
        "baryon_mass" => baryon_mass,
        "baryon_mass_source" => mass_was_overridden ?
            "HALQCD_BARYON_MASS override" :
            "sum of free Wilson quark pole masses: light + 2 strange",
        "selected_times" => selected_times,
        "fully_normalizable_stencil_times" => stencil_times,
        "results" => summary_by_time,
    )
    open(joinpath(temporary_directory, "summary.toml"), "w") do io
        TOML.print(io, summary; sorted=true)
    end
    mv(temporary_directory, output_directory)

    @printf("input_dataset=%s\n", input_dataset)
    @printf("output_directory=%s\n", output_directory)
    @printf("baryon_mass=%.17e\n", baryon_mass)
    @printf("baryon_mass_source=%s\n", summary["baryon_mass_source"])
    @printf("selected_times=%s\n", repr(Tuple(selected_times)))
    for time in selected_times
        result_summary = summary_by_time["time_$time"]
        @printf(
            "time=%d valid_sites=%d/%d max_abs_potential=%.17e yz_relative_error=%.17e imag_to_real=%.17e\n",
            time,
            result_summary["valid_sites"],
            result_summary["total_sites"],
            result_summary["maximum_abs_potential"],
            result_summary["r_yz_exchange_symmetric_relative_error"],
            result_summary["r_imaginary_to_real_maximum_ratio"],
        )
    end
    # Completion is not a physics pass/fail criterion.
    println("analysis_completed=true")
end

if abspath(PROGRAM_FILE) == @__FILE__
    analyze_spatial_nbs_potential_main()
end
