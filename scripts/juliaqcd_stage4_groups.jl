#!/usr/bin/env julia

# Stage-4 check: does the JuliaQCD Wilson pipeline run for gauge groups other
# than SU(3)?
#
# For each group in HALQCD_STAGE4_GROUPS (SU2, SU3, SU4):
#   1. fields:     cold and hot gauge fields build; gauge_group(U) is right.
#   2. free:       on cold links the solver agrees with the momentum-space free
#                  Wilson propagator in the LatticeDiracOperators.jl γ basis
#                  (entry by entry, pion and I=2 four-point function).
#   3. covariance: after a random gauge transformation of hot links,
#                  S'(x) = g(x) S(x) g(source)†, and the pion and four-point
#                  functions do not change.
#   4. reference:  the one-pion correlator agrees with
#                  QCDMeasurements.Pion_correlator_measurement.
#   5. measure:    HAL_NBS_measurement / measure gives a dataset that survives
#                  a disk round trip and a finite LO potential.
# A failing check prints the error and the top stack frames, so the function
# where a group stops is recorded. Small lattices only; run on a lab machine.

using Gaugefields
using LatticeDiracOperators
using QCDMeasurements
using HALQCD
using LinearAlgebra
using Printf
using Random

const NX = parse(Int, get(ENV, "HALQCD_STAGE4_NX", "4"))
const NT = parse(Int, get(ENV, "HALQCD_STAGE4_NT", "8"))
const KAPPA = parse(Float64, get(ENV, "HALQCD_STAGE4_KAPPA", "0.12"))
const EPS_CG = parse(Float64, get(ENV, "HALQCD_STAGE4_EPS_CG", "1e-20"))
const MAXCGSTEP = 10_000
const BOUNDARY = [1, 1, 1, -1]
const RTOL = parse(Float64, get(ENV, "HALQCD_STAGE4_RTOL", "1e-8"))
const GROUPS = split(get(ENV, "HALQCD_STAGE4_GROUPS", "SU2,SU3,SU4"), ",")
const OUTPUT = get(ENV, "HALQCD_STAGE4_OUTPUT", joinpath(pwd(), "stage4_group_datasets"))
const SOURCE = (2, 1, 3, 2)
# γ_1..γ_4 of the LatticeDiracOperators.jl Wilson operator, for comparing the
# free propagator entry by entry.
const LDO_GAMMAS =
    Tuple(LatticeDiracOperators.Dirac_operators.mk_gamma(1)[1][:, :, μ] for μ in 1:4)

maximum_relative_difference(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

ncolor_of(group) = parse(Int, group[3:end])
expected_group(group) = SU{ncolor_of(group)}()

group_element(group, rng) = random_group_element(SU{ncolor_of(group)}(), rng)

function gauge_field(group, condition)
    ncolor = ncolor_of(group)
    condition == "cold" &&
        return Initialize_Gaugefields(ncolor, 0, NX, NX, NX, NT; condition="cold")
    return Initialize_Gaugefields(
        ncolor, 0, NX, NX, NX, NT;
        condition="hot", randomnumber="Reproducible",
    )
end

link(Uμ, n, site) = [Uμ[a, b, site...] for a in 1:n, b in 1:n]

function set_link!(Uμ, n, site, matrix)
    for b in 1:n, a in 1:n
        Uμ[a, b, site...] = matrix[a, b]
    end
end

# U_μ(x) -> g(x) U_μ(x) g(x+μ)†, with periodic links.
function gauge_transform(U, rotations, n)
    transformed = deepcopy(U)
    shape = size(rotations)
    for μ in 1:4, index in CartesianIndices(shape)
        site = Tuple(index)
        forward = ntuple(ν -> ν == μ ? mod1(site[ν] + 1, shape[ν]) : site[ν], 4)
        set_link!(
            transformed[μ], n, site,
            rotations[site...] * link(U[μ], n, site) * rotations[forward...]',
        )
    end
    Gaugefields.set_wing_U!(transformed)
    return transformed
end

function solve(U; source_position=(1, 1, 1, 1))
    return juliaqcd_point_propagator(
        U; κ=KAPPA, eps_CG=EPS_CG, MaxCGstep=MAXCGSTEP, BoundaryCondition=BOUNDARY,
        source_position,
    )
end

function fields_check(group)
    cold = gauge_field(group, "cold")
    hot = gauge_field(group, "hot")
    println("gauge_field_type=", nameof(typeof(hot[1])))
    println("gauge_group=", gauge_group(hot))
    n = ncolor_of(group)
    first_link = link(hot[2], n, (1, 2, 1, 3))
    @printf("hot_link_unitarity_defect=%.3e\n", opnorm(first_link' * first_link - I))
    return gauge_group(hot) == expected_group(group) && gauge_group(cold) == expected_group(group)
end

function free_check(group)
    n = ncolor_of(group)
    solved = solve(gauge_field(group, "cold"); source_position=SOURCE)
    @printf("maximum_external_residual=%.6e\n", solved.metadata["maximum_external_residual"])
    free = free_wilson_point_propagator(
        (NX, NX, NX, NT); kappa=KAPPA, ncolor=n, boundary_condition=Tuple(BOUNDARY),
        source_position=SOURCE, gammas=LDO_GAMMAS,
    )
    propagator_difference = maximum_relative_difference(solved.data, free)
    pion_difference = maximum_relative_difference(
        pion_correlator_point_source(solved.data), pion_correlator_point_source(free),
    )
    four_point_difference = maximum_relative_difference(
        pipi_i2_four_point(solved.data), pipi_i2_four_point(free),
    )
    @printf("free_propagator_max_relative_difference=%.6e\n", propagator_difference)
    @printf("free_pion_max_relative_difference=%.6e\n", pion_difference)
    @printf("free_four_point_max_relative_difference=%.6e\n", four_point_difference)
    return max(propagator_difference, pion_difference, four_point_difference) <= RTOL
end

function covariance_check(group)
    n = ncolor_of(group)
    rng = MersenneTwister(20261104)
    U = gauge_field(group, "hot")
    rotations = [group_element(group, rng) for x in 1:NX, y in 1:NX, z in 1:NX, t in 1:NT]
    transformed_U = gauge_transform(U, rotations, n)
    S = solve(U; source_position=SOURCE).data
    transformed = solve(transformed_U; source_position=SOURCE)
    @printf("maximum_external_residual=%.6e\n", transformed.metadata["maximum_external_residual"])
    S_prime = transformed.data
    expected = similar(S)
    g_source = rotations[SOURCE...]
    for index in CartesianIndices(size(S)[5:8]), α in 1:4, β in 1:4
        expected[:, α, :, β, index] = rotations[index] * S[:, α, :, β, index] * g_source'
    end
    propagator_difference = maximum_relative_difference(S_prime, expected)
    pion_difference = maximum_relative_difference(
        pion_correlator_point_source(S_prime), pion_correlator_point_source(S),
    )
    four_point_difference = maximum_relative_difference(
        pipi_i2_four_point(S_prime), pipi_i2_four_point(S),
    )
    @printf("covariant_propagator_max_relative_difference=%.6e\n", propagator_difference)
    @printf("invariant_pion_max_relative_difference=%.6e\n", pion_difference)
    @printf("invariant_four_point_max_relative_difference=%.6e\n", four_point_difference)
    return max(propagator_difference, pion_difference, four_point_difference) <= RTOL
end

function reference_check(group)
    U = gauge_field(group, "hot")
    pion = pion_correlator_point_source(solve(U).data)
    reference = Pion_correlator_measurement(
        U;
        fermiontype="Wilson",
        κ=KAPPA,
        eps_CG=EPS_CG,
        MaxCGstep=MAXCGSTEP,
        BoundaryCondition=BOUNDARY,
        verbose_level=1,
    )
    reference_pion = get_value(measure(reference, U))
    difference = maximum_relative_difference(pion, reference_pion)
    @printf("pion_vs_qcdmeasurements_max_relative_difference=%.6e\n", difference)
    return difference <= RTOL
end

function measure_check(group)
    U = gauge_field(group, "hot")
    sample_id = "stage4-hot-$(lowercase(group))-$(NX)x$(NT)"
    m = HAL_NBS_measurement(
        U;
        sample_id,
        κ=KAPPA,
        eps_CG=EPS_CG,
        MaxCGstep=MAXCGSTEP,
        BoundaryCondition=BOUNDARY,
        source_positions=[(1, 1, 1, 1), SOURCE],
        verbose_level=1,
    )
    dataset = get_value(measure(m, U))
    println("dataset_gauge_group=", dataset.metadata.gauge_group)
    println("dataset_ncolor=", dataset.metadata.ncolor)
    directory = joinpath(OUTPUT, sample_id)
    isdir(directory) && rm(directory; recursive=true)
    loaded = read_nbs_dataset(write_nbs_dataset(directory, dataset))
    round_trip = loaded.four_point == dataset.four_point &&
        loaded.one_hadron_1 == dataset.one_hadron_1
    println("dataset_round_trip=$round_trip")
    pion = vec(sum(loaded.one_hadron_1[:, :, 1]; dims=2))
    potentials_finite = true
    for t in 2:NT÷2
        mass = log(pion[t] / pion[t+1])
        result = lo_potential_equal_mass(loaded, t; mass)
        potentials_finite &= all(isfinite, result.potential)
        @printf("t=%d effective_mass=%.6f potential_origin=%.6e\n",
            t, mass, real(result.potential[1, 1, 1]))
    end
    println("potentials_finite=$potentials_finite")
    return dataset.metadata.gauge_group == string(expected_group(group)) &&
        round_trip && potentials_finite
end

const CHECKS = (
    fields=fields_check,
    free=free_check,
    covariance=covariance_check,
    reference=reference_check,
    measure=measure_check,
)

function run_check(group, name, check)
    println("== $group $name")
    try
        seconds = @elapsed passed = check(group)
        @printf("%s_%s_seconds=%.2f\n", group, name, seconds)
        println("$(group)_$(name)_pass=$passed")
        return passed ? "pass" : "fail"
    catch error
        println("$(group)_$(name)_error=", sprint(showerror, error))
        for frame in first(stacktrace(catch_backtrace()), 12)
            println("    at ", frame)
        end
        return "error"
    end
end

function stage4_main()
    for extension in (:HALQCDJuliaQCDExt, :HALQCDQCDMeasurementsExt)
        loaded = !isnothing(Base.get_extension(HALQCD, extension))
        println("$(extension)_loaded=$loaded")
        loaded || error("$extension did not load")
    end
    @printf("kappa=%.6f eps_CG=%.1e lattice=%dx%dx%dx%d source=%s\n",
        KAPPA, EPS_CG, NX, NX, NX, NT, SOURCE)
    summary = Pair{String,String}[]
    for group in GROUPS
        for (name, check) in pairs(CHECKS)
            push!(summary, "$group $name" => run_check(group, name, check))
        end
    end
    println("== summary")
    for (case, status) in summary
        println(rpad(case, 20), status)
    end
    all(status -> status == "pass" || startswith(status, "skipped"), last.(summary)) ||
        error("stage-4 check failed")
    println("stage4_check_pass=true")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && stage4_main()
