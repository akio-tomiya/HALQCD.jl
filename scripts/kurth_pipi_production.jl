#!/usr/bin/env julia

# Stage 7: reproduction of Kurth, Ishii, Doi, Aoki, Hatsuda, JHEP 12 (2013) 015
# (arXiv:1305.4462), I=2 ππ from Lüscher's method and the HAL QCD potential.
# Everything runs on the JuliaQCD v1 path (Gaugefields.gauge_configuration,
# LatticeMatrices.jl MPILattice fields, one process).
#
# Paper setup: quenched SU(3), Wilson plaquette action at β = 5.8726 (a ≈ 0.115 fm),
# valence tree-level clover (c_SW = 1) Wilson quarks on 2HEX-smeared links,
# m_π ≈ 940 MeV, wall source in Coulomb gauge, temporal Dirichlet wall T/2 away from
# the source, periodic space, N_t = 128. κ is not given in the paper; it is tuned to
# m_π a ≈ 940 × 0.115 / 197.33 ≈ 0.548 (mode tune). The HEX parameters are those of
# the BMW papers the paper cites (arXiv:1011.2711: α = 0.95, 0.76, 0.38), an assumption.
#
#   KURTH_MODE       generate, measure or tune
#   KURTH_L, KURTH_T lattice (default 16, 128)
#   KURTH_BETA       5.8726
#   KURTH_KAPPA      valence κ (measure); KURTH_KAPPAS="k1,k2,..." for tune
#   KURTH_CSW        1.0
#   KURTH_THERM      thermalization sweeps (default 500)
#   KURTH_SEPARATION sweeps between saved configurations (default 50)
#   KURTH_OR         overrelaxation steps per heatbath sweep (default 4)
#   KURTH_CONFIGS    last configuration number; KURTH_FIRST the first (default 1)
#   KURTH_SEED       seed of the chain (default 20261107); use a new one per chain
#   KURTH_OUTPUT     output folder (configs/ and nbs/ inside)
#   KURTH_EPS_CG     squared relative residual of the solver (default 1e-24)
#   KURTH_GF_VERBOSE Gaugefields verbose level for the gauge fixing (default 1)
#   KURTH_SOURCE_SHIFTS time shifts of the source (default "0"; the paper's four
#                    sources 32 slices apart are "0,32,64,96")
#
# measure prints the time of each phase (Coulomb gauge fixing, HEX, each of the
# 12 solves, the end) so that a long configuration can be followed in the log.
#
# generate writes configs/NNNN.jld2 (Gaugefields.save_configuration, thin links).
# measure takes every configuration without a dataset or lock, fixes Coulomb gauge
# on the thin links, applies 2HEX, solves the wall source at t = T/2 + 1 with the
# Dirichlet wall between t = T and 1 (on the links shifted by each of
# KURTH_SOURCE_SHIFTS), and writes the I=2 NBS dataset to nbs/, one source per shift.
# Several measure workers may run side by side (locks as in stage 6).
import JACC
JACC.@init_backend
using Gaugefields
using LatticeDiracOperators
using HALQCD
using LinearAlgebra
using Printf

const MODE = get(ENV, "KURTH_MODE", "measure")
const L = parse(Int, get(ENV, "KURTH_L", "16"))
const T = parse(Int, get(ENV, "KURTH_T", "128"))
const BETA = parse(Float64, get(ENV, "KURTH_BETA", "5.8726"))
const KAPPA = parse(Float64, get(ENV, "KURTH_KAPPA", "0.1225"))
const CSW = parse(Float64, get(ENV, "KURTH_CSW", "1.0"))
const THERM = parse(Int, get(ENV, "KURTH_THERM", "500"))
const SEPARATION = parse(Int, get(ENV, "KURTH_SEPARATION", "50"))
const OR_STEPS = parse(Int, get(ENV, "KURTH_OR", "4"))
const FIRST = parse(Int, get(ENV, "KURTH_FIRST", "1"))
const CONFIGS = parse(Int, get(ENV, "KURTH_CONFIGS", "100"))
const SEED = parse(Int, get(ENV, "KURTH_SEED", "20261107"))
const OUTPUT = get(ENV, "KURTH_OUTPUT", joinpath(pwd(), "kurth_pipi"))
const EPS_CG = parse(Float64, get(ENV, "KURTH_EPS_CG", "1e-24"))
const GF_VERBOSE = parse(Int, get(ENV, "KURTH_GF_VERBOSE", "1"))
const HEX = HEXSmearing(; alpha_outer=0.95, alpha_middle=0.76, alpha_inner=0.38, iterations=2)
const SOURCE_TIME = T ÷ 2 + 1
const SOURCE_SHIFTS = parse.(Int, split(get(ENV, "KURTH_SOURCE_SHIFTS", "0"), ","))

config_path(n) = joinpath(OUTPUT, "configs", @sprintf("su3_b%.4f_%dx%d_%04d.jld2", BETA, L, T, n))
dataset_path(n) = joinpath(OUTPUT, "nbs", @sprintf("nbs_su3_b%.4f_k%.5f_%dx%d_%04d", BETA, KAPPA, L, T, n))
lock_path(n) = joinpath(OUTPUT, "nbs", @sprintf(".lock_%04d", n))

function sweep!(U, heatbath, count)
    for _ in 1:count
        heatbath!(U, heatbath)
        for _ in 1:OR_STEPS
            overrelaxation!(U, heatbath)
        end
    end
    return U
end

function save_config(n, U)
    temporary = config_path(n) * ".partial.jld2"
    save_configuration(temporary, U)
    mv(temporary, config_path(n))
end

function generate_main()
    mkpath(joinpath(OUTPUT, "configs"))
    @printf("generate lattice=%dx%d beta=%.4f therm=%d separation=%d or=%d configs=%d..%d seed=%d\n",
        L, T, BETA, THERM, SEPARATION, OR_STEPS, FIRST, CONFIGS, SEED)
    flush(stdout)
    U = gauge_configuration((L, L, L, T); colors=3, start=:hot, seed=SEED, process_grid=(1, 1, 1, 1))
    heatbath = heatbath_updater(U; beta=BETA, seed=SEED)
    done = findlast(n -> isfile(config_path(n)), FIRST:CONFIGS)
    if done === nothing
        for block in 1:10
            seconds = @elapsed sweep!(U, heatbath, cld(THERM, 10))
            @printf("therm sweeps=%d seconds=%.1f plaquette=%.10f\n", block * cld(THERM, 10), seconds,
                measure_plaquette(U))
            flush(stdout)
        end
        first_config = FIRST
    else
        n = FIRST + done - 1
        # The chain continues from the last saved configuration with a new seed.
        U = load_configuration(config_path(n))
        heatbath = heatbath_updater(U; beta=BETA, seed=SEED + n)
        println("resumed from config ", n)
        first_config = n + 1
    end
    for n in first_config:CONFIGS
        seconds = @elapsed sweep!(U, heatbath, n == FIRST && done === nothing ? 0 : SEPARATION)
        save_config(n, U)
        @printf("config=%d sweep_seconds=%.1f plaquette=%.10f\n", n, seconds, measure_plaquette(U))
        flush(stdout)
    end
    println("kurth_generation_done=true")
end

function coulomb_fix!(U)
    g_transform = similar(U[1])
    work = [similar(U[1]) for _ in 1:6]
    before = measure_plaquette(U)
    seconds = @elapsed gaugefixing!(U, g_transform, 1.0, 200, 1.99, 20_000, 1e-14, 0, work;
        D_fix=3, min_iterations=500)
    residual = coulomb_gauge_residual(juliaqcd_links(U))
    abs(measure_plaquette(U) - before) <= 1e-10 || error("the plaquette changed under gauge fixing")
    return (seconds=seconds, residual=residual.maximum)
end

# Sources at several times on one configuration (the paper uses four, 32 slices
# apart). The Dirichlet wall sits between t = T and 1, T/2 away from the source at
# SOURCE_TIME, so a source at SOURCE_TIME + shift is measured on the links shifted
# in time, V'(t) = V(t + shift), with the wall moved with it. Coulomb gauge and
# 2HEX commute with the shift (the gauge field is periodic in time), so both are
# done once per configuration.
function time_shifted(V, shift)
    shift == 0 && return V
    W = copy_configuration(V)
    for μ in 1:4
        f = W[μ].U
        core = view(f.A, :, :, ntuple(d -> (f.nw+1):(f.nw+f.PN[d]), 4)...)
        core .= circshift(Array(core), (0, 0, 0, 0, 0, -shift))
        Gaugefields.LatticeMatricesCompat.mark_lattice_dirty!(f)
        Gaugefields.set_wing_U!(W[μ])
    end
    return W
end

# One dataset with the source axis running over the shifts. The time axis of each
# is already measured from its own source.
function merge_sources(datasets, shifts)
    length(datasets) == 1 && return only(datasets)
    times = [mod(SOURCE_TIME + shift - 1, T) + 1 for shift in shifts]
    metadata = merge(first(datasets).metadata, (
        source_ids=Tuple("wall-1-1-1-$t" for t in times),
        source_positions=Tuple("1,1,1,$t" for t in times),
        source_time_shifts=Tuple(shifts),
        source_layout="links shifted in time by source_time_shifts; Dirichlet wall T/2 from each source",
        maximum_external_residual=maximum(d.metadata.maximum_external_residual for d in datasets),
    ))
    return NBSCorrelatorDataset(
        cat((d.four_point for d in datasets)...; dims=5),
        cat((d.one_hadron_1 for d in datasets)...; dims=2),
        cat((d.one_hadron_2 for d in datasets)...; dims=2);
        metadata,
    )
end

function stamp(n, phase)
    @printf("%s config=%d %s\n", Libc.strftime("%H:%M:%S", time()), n, phase)
    flush(stdout)
end

function measure(n)
    # GF_VERBOSE is the Gaugefields verbose level of the loaded links: 1 prints the
    # start and the end of each gauge-fixing stage, 3 every iteration (run_meas.sh
    # keeps every 500th). It changes only what is printed.
    U = load_configuration(config_path(n); verbose=GF_VERBOSE)
    stamp(n, "coulomb_start")
    fixed = coulomb_fix!(U)
    stamp(n, @sprintf("coulomb_done seconds=%.1f residual=%.2e", fixed.seconds, fixed.residual))
    hex_seconds = @elapsed V = smear(U, HEX)
    stamp(n, @sprintf("hex_done seconds=%.1f", hex_seconds))
    source = (1, 1, 1, SOURCE_TIME)
    sample_id = @sprintf("kurth-su3-b%.4f-k%.5f-%dx%d-%04d", BETA, KAPPA, L, T, n)
    seconds = 0.0
    datasets = map(SOURCE_SHIFTS) do shift
        stamp(n, @sprintf("source_start shift=%d", shift))
        seconds += @elapsed shifted = juliaqcd_pipi_i2_nbs(time_shifted(V, shift); κ=KAPPA, sample_id,
            source_positions=[source], BoundaryCondition=[1, 1, 1, 0], fermiontype="WilsonClover", cSW=CSW,
            source_type=:wall, method_CG="bicgstab", eps_CG=EPS_CG, MaxCGstep=20_000, progress=stdout)
        shifted
    end
    dataset = merge_sources(datasets, SOURCE_SHIFTS)
    stamp(n, @sprintf("solve_and_contraction_done seconds=%.1f", seconds))
    write_nbs_dataset(dataset_path(n), dataset)
    @printf("config=%d coulomb_seconds=%.1f coulomb_residual=%.2e hex_seconds=%.1f solve_seconds=%.1f max_residual=%.2e\n",
        n, fixed.seconds, fixed.residual, hex_seconds, seconds, get(dataset.metadata, :maximum_external_residual, NaN))
end

function take_lock(n)
    try
        mkdir(lock_path(n))
    catch
        return false
    end
    write(joinpath(lock_path(n), "pid"), string(getpid()))
    return true
end

function measure_main()
    BLAS.set_num_threads(1)
    mkpath(joinpath(OUTPUT, "nbs"))
    @printf("measure pid=%d threads=%d lattice=%dx%d kappa=%.5f csw=%.3f source_time=%d output=%s\n",
        getpid(), Threads.nthreads(), L, T, KAPPA, CSW, SOURCE_TIME, OUTPUT)
    flush(stdout)
    numbers = FIRST:CONFIGS
    while true
        index = findfirst(n -> isfile(config_path(n)) && !isdir(dataset_path(n)) && !isdir(lock_path(n)), numbers)
        if index === nothing
            # A configuration may be gone once its dataset is written (a worker that
            # fetches configurations from another machine deletes its copies).
            if all(n -> isfile(config_path(n)) || isdir(dataset_path(n)), numbers)
                println("measure_worker_done=true still_locked=", filter(n -> !isdir(dataset_path(n)), numbers))
                return
            end
            sleep(30)
            continue
        end
        candidate = numbers[index]
        take_lock(candidate) || continue
        try
            isdir(dataset_path(candidate)) || measure(candidate)
        finally
            rm(lock_path(candidate); recursive=true, force=true)
        end
        flush(stdout)
    end
end

# κ tuning: point-source pion effective masses on the saved configurations, with
# antiperiodic time and no gauge fixing.
# KURTH_TUNE_CONFIGS="5,15,25" picks configurations (default: every one in
# KURTH_FIRST..KURTH_CONFIGS). With KURTH_TUNE_OUT=path each correlator is appended
# to path as "kappa config C(0) C(1) ... C(T-1)" (time from the source), for a
# jackknife over configurations later (scripts/kurth_kappa_fit.jl).
function tune_main()
    kappas = parse.(Float64, split(get(ENV, "KURTH_KAPPAS", "0.122,0.1225,0.123"), ","))
    chosen = get(ENV, "KURTH_TUNE_CONFIGS", "")
    candidates = isempty(chosen) ? (FIRST:CONFIGS) : parse.(Int, split(chosen, ","))
    numbers = filter(n -> isfile(config_path(n)), candidates)
    out = get(ENV, "KURTH_TUNE_OUT", "")
    @printf("tune lattice=%dx%d kappas=%s configs=%s\n", L, T, kappas, collect(numbers))
    flush(stdout)
    for κ in kappas
        correlators = map(numbers) do n
            V = smear(load_configuration(config_path(n)), HEX)
            seconds = @elapsed S = juliaqcd_point_propagator(V; κ, fermiontype="WilsonClover", cSW=CSW,
                BoundaryCondition=[1, 1, 1, -1], method_CG="bicgstab", eps_CG=1e-20, MaxCGstep=20_000,
                source_position=(1, 1, 1, 1)).data
            C = pion_correlator_point_source(S)
            if !isempty(out)
                open(out, "a") do io
                    println(io, κ, " ", n, " ", join((@sprintf("%.16e", c) for c in C), " "))
                end
            end
            @printf("%s kappa=%.5f config=%d seconds=%.1f\n", Libc.strftime("%H:%M:%S", time()), κ, n, seconds)
            flush(stdout)
            C
        end
        C = sum(correlators) / length(correlators)
        # cosh effective mass from C(t-1) + C(t+1) = 2 cosh(m) C(t)
        meff = [acosh((C[t-1] + C[t+1]) / (2C[t])) for t in 2:(T÷2)]
        @printf("kappa=%.5f meff(t=2..%d): %s\n", κ, T ÷ 2, join((@sprintf("%.4f", m) for m in meff), " "))
        flush(stdout)
    end
end

# κ tuning with a smeared operator basis
# (point and Jacobi-smeared sources and sinks, then GEVP): sources P (point) and
# Wuppertal-smeared points W<steps> (KURTH_SMEARINGS="alpha:steps,...", default
# "0.5:10,0.5:30"), the same smearings at the sink, antiperiodic time, no gauge fixing.
# The quark source and sink are smeared with HALQCD.wuppertal_smear on the 2HEX links.
# The sink smearing is applied column by column to the propagator, so the matrix
# C[sink, source](t) = Σ_x tr[(K_sink S K_source)(K_sink S K_source)†] costs one set
# of 12 solves per source. Lines "kappa config sink source C(0) ... C(T-1)" are
# appended to KURTH_TUNE_OUT (scripts/kurth_kappa_gevp.jl reads them).
function tune_gevp_main()
    kappas = parse.(Float64, split(get(ENV, "KURTH_KAPPAS", "0.1218,0.1221,0.1224"), ","))
    numbers = filter(n -> isfile(config_path(n)), parse.(Int, split(ENV["KURTH_TUNE_CONFIGS"], ",")))
    smearings = [(alpha=parse(Float64, a), steps=parse(Int, s))
                 for (a, s) in split.(split(get(ENV, "KURTH_SMEARINGS", "0.5:10,0.5:30"), ","), ":")]
    labels = ["P"; ["W$(s.steps)" for s in smearings]]
    out = ENV["KURTH_TUNE_OUT"]
    @printf("tune_gevp lattice=%dx%d kappas=%s configs=%s smearings=%s\n", L, T, kappas, numbers, smearings)
    flush(stdout)
    # (κ, configuration, source) already written with every sink, kept on a restart
    written = Dict{Tuple{Float64,Int,String},Int}()
    isfile(out) && for line in eachline(out)
        f = split(line)
        k = (parse(Float64, f[1]), parse(Int, f[2]), String(f[4]))
        written[k] = get(written, k, 0) + 1
    end
    done(κ, n, source) = get(written, (κ, n, source), 0) >= length(labels)
    for n in numbers
        all(done(κ, n, l) for κ in kappas, l in labels) && continue
        V = smear(load_configuration(config_path(n)), HEX)
        links = juliaqcd_links(V)
        for κ in kappas
            for (j, source_smearing) in enumerate([nothing; smearings])
                done(κ, n, labels[j]) && continue
                seconds = @elapsed S = juliaqcd_point_propagator(V; κ, fermiontype="WilsonClover", cSW=CSW,
                    BoundaryCondition=[1, 1, 1, -1], method_CG="bicgstab", eps_CG=1e-20, MaxCGstep=20_000,
                    source_position=(1, 1, 1, 1), source_smearing).data
                correlators = [zeros(T) for _ in labels]
                for c in axes(S, 3), s in axes(S, 4)
                    column = S[:, :, c, s, :, :, :, :]
                    for (i, sink_smearing) in enumerate([nothing; smearings])
                        smeared = isnothing(sink_smearing) ? column :
                                  wuppertal_smear(links, column; sink_smearing...)
                        for t in 1:T
                            correlators[i][t] += sum(abs2, view(smeared, :, :, :, :, :, t))
                        end
                    end
                end
                S = nothing
                open(out, "a") do io
                    for (i, C) in enumerate(correlators)
                        println(io, κ, " ", n, " ", labels[i], " ", labels[j], " ",
                            join((@sprintf("%.16e", c) for c in C), " "))
                    end
                end
                @printf("%s kappa=%.5f config=%d source=%s solve_seconds=%.1f\n",
                    Libc.strftime("%H:%M:%S", time()), κ, n, labels[j], seconds)
                flush(stdout)
                GC.gc()
            end
        end
    end
    println("tune_gevp_done=true")
end

function main()
    MODE == "generate" && return generate_main()
    MODE == "measure" && return measure_main()
    MODE == "tune" && return tune_main()
    MODE == "tune_gevp" && return tune_gevp_main()
    error("KURTH_MODE must be generate, measure, tune or tune_gevp")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
