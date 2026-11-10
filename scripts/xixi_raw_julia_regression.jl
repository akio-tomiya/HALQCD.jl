#!/usr/bin/env julia

using HALQCD

function load_numeric_column(path, column)
    values = Float64[]
    for line in eachline(path)
        stripped = strip(line)
        (isempty(stripped) || startswith(stripped, '#')) && continue
        fields = split(stripped)
        length(fields) >= column ||
            error("missing column $column in $path")
        push!(values, parse(Float64, fields[column]))
    end
    return values
end

function configuration_directories(root)
    return sort([
        joinpath(root, entry)
        for entry in readdir(root)
        if isdir(joinpath(root, entry))
    ])
end

function unique_prefix_file(directory, prefix)
    matches = [
        joinpath(directory, entry)
        for entry in readdir(directory)
        if startswith(entry, prefix)
    ]
    length(matches) == 1 ||
        error("expected one '$prefix*' file in $directory, found $(length(matches))")
    return only(matches)
end

function bin_means(samples, bin_size)
    length(samples) % bin_size == 0 ||
        error("sample count is not divisible by bin size")
    return [
        reduce(+, @view samples[start:(start + bin_size - 1)]) / bin_size
        for start in 1:bin_size:length(samples)
    ]
end

function load_xi_jackknife_correlator(data_root; bin_size=2)
    root = joinpath(data_root, "correlator.PS.dir")
    directories = configuration_directories(root)
    forward = [
        load_numeric_column(
            unique_prefix_file(directory, "Xi_CG05_CG05_"),
            2,
        )
        for directory in directories
    ]
    backward = [
        load_numeric_column(
            unique_prefix_file(directory, "antiXi_CG05_CG05_"),
            2,
        )
        for directory in directories
    ]
    forward_bins = bin_means(forward, bin_size)
    backward_bins = bin_means(backward, bin_size)
    nt = length(first(forward_bins))
    reverse_times = [mod(nt - time, nt) + 1 for time in 0:(nt - 1)]
    folded_bins = [
        forward_bins[bin] .+ backward_bins[bin][reverse_times]
        for bin in eachindex(forward_bins)
    ]
    return jackknife_estimates(folded_bins)
end

function load_wave_jackknife(data_root, time; bin_size=2)
    root = joinpath(data_root, "BBwave.dir.S4.00")
    directories = configuration_directories(root)
    prefix = "NBSwave.+$(lpad(time, 3, '0'))+B00"
    configurations = [
        load_ph1_compressed_a1plus_singlet(
            unique_prefix_file(directory, prefix),
        )
        for directory in directories
    ]
    return jackknife_estimates(bin_means(configurations, bin_size))
end

function stack_samples(samples)
    output = Array{eltype(first(samples))}(
        undef,
        size(first(samples))...,
        length(samples),
    )
    for sample in eachindex(samples)
        output[:, :, :, sample] .= samples[sample]
    end
    return output
end

function table_coordinates(lattice_size)
    return [
        # NumPy C-order flattening maps the reference linear index to (z,y,x).
        (z + 1, y + 1, x + 1)
        for z in 0:(lattice_size ÷ 2)
        for y in z:(lattice_size ÷ 2)
        for x in y:(lattice_size ÷ 2)
    ]
end

function report_difference(name, summary, reference, radii, coordinates)
    means = [real(summary.mean[index...]) for index in coordinates]
    errors = [summary.stderr[index...] for index in coordinates]
    mean_difference = abs.(means .- reference.mean)
    error_difference = abs.(errors .- reference.stderr)
    short_distance = radii .<= 12
    println(name, "_mean_max_abs=", maximum(mean_difference))
    println(name, "_stderr_max_abs=", maximum(error_difference))
    println(
        name,
        "_mean_rms=",
        sqrt(sum(abs2, mean_difference) / length(mean_difference)),
    )
    println(
        name,
        "_stderr_rms=",
        sqrt(sum(abs2, error_difference) / length(error_difference)),
    )
    println(
        name,
        "_mean_max_abs_r_le_12=",
        maximum(mean_difference[short_distance]),
    )
    println(
        name,
        "_stderr_max_abs_r_le_12=",
        maximum(error_difference[short_distance]),
    )
    return (
        mean_max_abs=maximum(mean_difference),
        stderr_max_abs=maximum(error_difference),
    )
end

function write_on_axis_table(path, result, reference, coordinates)
    open(path, "w") do io
        println(
            io,
            "r\tjulia_total\tjulia_stderr\treference_total\treference_stderr",
        )
        for index in eachindex(coordinates)
            coordinate = coordinates[index]
            coordinate[1] == 1 && coordinate[2] == 1 || continue
            println(
                io,
                join(
                    (
                        reference.radius[index],
                        real(result.components.total.mean[coordinate...]),
                        result.components.total.stderr[coordinate...],
                        reference.total.mean[index],
                        reference.total.stderr[index],
                    ),
                    '\t',
                ),
            )
        end
    end
    return path
end

function interpolate_shells(shells, radius)
    shell_radii = [shell.radius for shell in shells]
    index = searchsortedlast(shell_radii, radius)
    index == 0 && return real(first(shells).mean)
    index == length(shells) && return real(last(shells).mean)
    left = shells[index]
    right = shells[index + 1]
    weight = (radius - left.radius) / (right.radius - left.radius)
    return (1 - weight) * real(left.mean) + weight * real(right.mean)
end

function radial_potential_grid(
    potential,
    valid,
    cutoff;
    radial_spacing=0.02,
)
    shells = shell_average(potential; valid=valid)
    radii = collect(0.0:radial_spacing:24.0)
    radial_potential = [
        radius <= cutoff ? interpolate_shells(shells, radius) : 0.0
        for radius in radii
    ]
    return radii, radial_potential
end

function scattering_from_potential(
    potential,
    valid,
    cutoff;
    mass=0.665,
    radial_spacing=0.02,
)
    radii, radial_potential = radial_potential_grid(
        potential,
        valid,
        cutoff;
        radial_spacing=radial_spacing,
    )
    return s_wave_scattering_length(
        radii,
        radial_potential;
        reduced_mass=mass / 2,
        tail_start=cutoff + 2,
    )
end

function born_scattering_length(
    potential,
    valid,
    cutoff;
    mass=0.665,
    radial_spacing=0.02,
)
    radii, radial_potential = radial_potential_grid(
        potential,
        valid,
        cutoff;
        radial_spacing=radial_spacing,
    )
    integrand = radii .^ 2 .* radial_potential
    integral = radial_spacing * (
        sum(integrand) -
        (first(integrand) + last(integrand)) / 2
    )
    return mass * integral # 2 * reduced_mass, with reduced_mass = mass / 2
end

function write_scattering_cutoff_table(path, result)
    open(path, "w") do io
        println(
            io,
            "cutoff\tjackknife_mean\tjackknife_stderr\t" *
            "mean_potential_value\tborn_mean_potential\ttail_spread",
        )
        for cutoff in (8.0, 10.0, 12.0, 14.0, 16.0, 18.0)
            sample_values = [
                scattering_from_potential(
                    estimate.potential,
                    estimate.valid,
                    cutoff,
                ).value
                for estimate in result.estimates
            ]
            summary = jackknife_summary(sample_values)
            mean_potential_result = scattering_from_potential(
                result.components.total.mean,
                result.valid,
                cutoff,
            )
            born_value = born_scattering_length(
                result.components.total.mean,
                result.valid,
                cutoff,
            )
            println(
                io,
                join(
                    (
                        cutoff,
                        summary.mean,
                        summary.stderr,
                        mean_potential_result.value,
                        born_value,
                        mean_potential_result.tail_spread,
                    ),
                    '\t',
                ),
            )
        end
    end
    return path
end

function write_scattering_grid_table(path, result)
    open(path, "w") do io
        println(io, "cutoff\tradial_spacing\tscattering_length\ttail_spread")
        for cutoff in (12.0, 16.0), radial_spacing in (0.1, 0.05, 0.02, 0.01)
            scattering = scattering_from_potential(
                result.components.total.mean,
                result.valid,
                cutoff;
                radial_spacing=radial_spacing,
            )
            println(
                io,
                join(
                    (
                        cutoff,
                        radial_spacing,
                        scattering.value,
                        scattering.tail_spread,
                    ),
                    '\t',
                ),
            )
        end
    end
    return path
end

function write_shell_potential_table(path, result)
    sample_shells = [
        shell_average(estimate.potential; valid=estimate.valid)
        for estimate in result.estimates
    ]
    shell_count = length(first(sample_shells))
    all(length(shells) == shell_count for shells in sample_shells) ||
        error("jackknife shell grids do not match")

    open(path, "w") do io
        println(io, "r2\tradius\tjackknife_mean\tjackknife_stderr\tsite_count")
        for index in 1:shell_count
            r2 = first(sample_shells)[index].r2
            radius = first(sample_shells)[index].radius
            count = first(sample_shells)[index].count
            all(
                shells[index].r2 == r2 && shells[index].count == count
                for shells in sample_shells
            ) || error("jackknife shell membership does not match")
            summary = jackknife_summary([
                real(shells[index].mean)
                for shells in sample_shells
            ])
            println(
                io,
                join(
                    (r2, radius, summary.mean, summary.stderr, count),
                    '\t',
                ),
            )
        end
    end
    return path
end

function main(arguments)
    length(arguments) == 1 ||
        error("usage: xixi_raw_julia_regression.jl REFERENCE_REPOSITORY")
    reference_repository = only(arguments)
    data_root = joinpath(reference_repository, "data", "sample_data")
    correlator = load_xi_jackknife_correlator(data_root)

    normalized = Dict{Int,Vector}()
    for time in (9, 10, 11)
        wave = load_wave_jackknife(data_root, time)
        normalized[time] = [
            real.(wave[sample]) ./ correlator[sample][time + 1]^2
            for sample in eachindex(wave)
        ]
    end

    result = lo_potential_from_jackknife_equal_mass(
        stack_samples(normalized[9]),
        stack_samples(normalized[10]),
        stack_samples(normalized[11]);
        mass=0.665,
        rtol=0,
        atol=0,
    )
    table_path = joinpath(
        reference_repository,
        "notes",
        "results.pot",
        "pot_1s0_cen_xixi_t010_004conf_002bin.dat",
    )
    reference = load_hal_pot_table(table_path)
    coordinates = table_coordinates(48)

    println("julia_version=", VERSION)
    println("reference_commit=3446378916df74eca887cb36c380ed0ab151192e")
    println("lattice_size=48")
    println("configuration_count=4")
    println("bin_size=2")
    println("jackknife_samples=", length(correlator))
    println("valid_sites=", count(result.valid))
    println("total_sites=", length(result.valid))
    spatial_difference = report_difference(
        "spatial",
        result.components.spatial,
        reference.spatial,
        reference.radius,
        coordinates,
    )
    first_time_difference = report_difference(
        "first_time",
        result.components.first_time,
        reference.first_time,
        reference.radius,
        coordinates,
    )
    second_time_difference = report_difference(
        "second_time",
        result.components.second_time,
        reference.second_time,
        reference.radius,
        coordinates,
    )
    total_difference = report_difference(
        "total",
        result.components.total,
        reference.total,
        reference.radius,
        coordinates,
    )

    checks = (
        spatial_difference.mean_max_abs <= 5e-7,
        spatial_difference.stderr_max_abs <= 5e-9,
        first_time_difference.mean_max_abs <= 5e-9,
        first_time_difference.stderr_max_abs <= 5e-9,
        second_time_difference.mean_max_abs <= 5e-9,
        second_time_difference.stderr_max_abs <= 5e-9,
        total_difference.mean_max_abs <= 5e-7,
        total_difference.stderr_max_abs <= 5e-9,
        count(result.valid) == length(result.valid),
    )
    println("regression_pass=", all(checks))
    all(checks) ||
        error("XiXi regression exceeded the published-table precision thresholds")

    report_table = get(ENV, "HALQCD_REPORT_TSV", "")
    isempty(report_table) ||
        println(
            "on_axis_table=",
            write_on_axis_table(
                report_table,
                result,
                reference,
                coordinates,
            ),
        )

    scattering_table = get(ENV, "HALQCD_SCATTERING_TSV", "")
    isempty(scattering_table) ||
        println(
            "scattering_cutoff_table=",
            write_scattering_cutoff_table(scattering_table, result),
        )

    shell_table = get(ENV, "HALQCD_SHELL_TSV", "")
    isempty(shell_table) ||
        println(
            "shell_potential_table=",
            write_shell_potential_table(shell_table, result),
        )

    scattering_grid_table = get(ENV, "HALQCD_SCATTERING_GRID_TSV", "")
    isempty(scattering_grid_table) ||
        println(
            "scattering_grid_table=",
            write_scattering_grid_table(scattering_grid_table, result),
        )
end

main(ARGS)
