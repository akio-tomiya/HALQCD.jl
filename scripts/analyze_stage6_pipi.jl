#!/usr/bin/env julia

# I=2 two-pseudoscalar scattering from HAL QCD and from Lüscher's formula on
# the same configurations, without fits.
#
#   julia analyze_stage6_pipi.jl DATASET_DIR...   (one NBS dataset per configuration)
#
#   HALQCD_STAGE6_MASS_WINDOW    t range (0-based from the source) of the
#                                effective pion mass log(C(t)/C(t+1)), e.g. "8:14"
#   HALQCD_STAGE6_SHIFT_WINDOW   t range of the effective energy shift
#                                log(R(t)/R(t+1)), R = C_ππ/C_π²
#   HALQCD_STAGE6_POTENTIAL_TIME t of the LO potential (needs t-1 to t+2)
#   HALQCD_STAGE6_POTENTIAL_CSV  if set, write V at every relative position
#                                with its jackknife error to this file
#   HALQCD_STAGE6_BIN            average this many consecutive datasets (in the
#                                order given) before the jackknife, default 1;
#                                choose it to divide each Markov chain's length
#   HALQCD_STAGE6_THERMAL        "1" removes the wrap-around of the periodic time
#                                direction (below), default "0"
#   HALQCD_STAGE6_A1             "1" projects R onto the A1+ representation of the
#                                cubic group about r = 0 (the S wave up to l = 4, ...)
#                                before taking V, default "0"
#   HALQCD_STAGE6_NORM           "pion" (default) divides F by C_π(t)²; "exp" by
#                                e^{-2mt} with m from the mass window, the cross-check of
#                                Akahoshi & Aoki arXiv:2301.06038 App. A (pion excited
#                                states then stay in R instead of cancelling)
#   HALQCD_STAGE6_POTENTIAL_SAMPLES  if set, write m and V on one site per cubic orbit
#                                for the central value and every jackknife sample
#                                (input of scripts/fit_stage6_potential.jl)
#
# Every estimate is a delete-one jackknife over configurations (or bins):
#   m      plateau average of the effective pion mass;
#   ΔE     plateau average of the effective energy shift of the zero-momentum
#          two-pion correlator C_ππ(t) = Σ_r F(r, t);
#   a_L    luscher_scattering_length(ΔE, m, L);
#   V(r)   time-dependent LO potential from R(r, t) = F(r, t)/C_π(t)²;
#   ΔE_V   from the lowest eigenvalue E₀ of H = -∇²/m + V on the same periodic
#          lattice (p² = m E₀, ΔE_V = 2√(m² + p²) - 2m), and a_V from ΔE_V.
# With the same Laplacian in the potential and in H, ΔE_V reproduces ΔE when
# the potential describes the correlator; a_V and a_L then agree as well.
#
# Wrap-around (HALQCD_STAGE6_THERMAL=1). With period T, C_π(t) ∝ e^{-mt} + e^{-m(T-t)},
# and F(r, t) gains a t-independent term from one pion going forward and one
# backward across the boundary. Then m comes from the cosh effective mass, C_π
# is replaced by its forward part C_π(t)/(1 + e^{-m(T-2t)}), and F(r, t) by
# F(r, t) - F(r, t+1). The difference keeps every forward two-pion state, with
# its weight times (1 - e^{-E_n}), so R is still a sum of solutions of the
# time-dependent HAL equation. Both pions going backward, of relative size
# e^{-2m(T-2t)}, is left in. scripts/check_stage6_analysis_excited_thermal.jl
# checks this on a model.

using HALQCD
using LinearAlgebra
using Printf

parse_window(text) = (parts = parse.(Int, split(text, ":")); parts[1]:parts[2])

const MASS_WINDOW = parse_window(get(ENV, "HALQCD_STAGE6_MASS_WINDOW", "8:14"))
const SHIFT_WINDOW = parse_window(get(ENV, "HALQCD_STAGE6_SHIFT_WINDOW", "8:14"))
const POTENTIAL_TIME = parse(Int, get(ENV, "HALQCD_STAGE6_POTENTIAL_TIME", "10"))
const BIN = parse(Int, get(ENV, "HALQCD_STAGE6_BIN", "1"))
const THERMAL = get(ENV, "HALQCD_STAGE6_THERMAL", "0") == "1"
const A1 = get(ENV, "HALQCD_STAGE6_A1", "0") == "1"
const NORM = get(ENV, "HALQCD_STAGE6_NORM", "pion")
NORM in ("pion", "exp") || error("HALQCD_STAGE6_NORM must be pion or exp")

# Per-configuration observables (real parts), with t counted from the source and
# averaged over the sources of the configuration (the source axis).
function observables(dataset)
    nt = size(dataset.four_point, 4)
    ns = size(dataset.four_point, 5)
    pion = [sum(real(dataset.one_hadron_1[t, k, 1]) for k in 1:ns) / ns for t in 1:nt]
    pipi = [sum(real(sum(view(dataset.four_point, :, :, :, t, k, 1))) for k in 1:ns) / ns for t in 1:nt]
    stencil = [sum(real.(dataset.four_point[:, :, :, POTENTIAL_TIME + 1 + δ, k, 1]) for k in 1:ns) / ns
               for δ in -1:2]
    return (pion=pion, pipi=pipi, stencil=stencil)
end

average(samples, field) = sum(getfield(s, field) for s in samples) / length(samples)

# Every estimate is built from averages of these observables, so a bin is the
# average of its members.
function binned(samples, size)
    length(samples) % size == 0 || error("$(length(samples)) datasets do not split into bins of $size")
    return [(pion=average(group, :pion), pipi=average(group, :pipi),
            stencil=[sum(s.stencil[k] for s in group) / size for k in eachindex(group[1].stencil)])
            for group in Iterators.partition(samples, size)]
end

plateau(values, window) = sum(log(values[t+1] / values[t+2]) for t in window) / length(window)

# m with cosh(m(t - T/2))/cosh(m(t + 1 - T/2)) = C(t)/C(t+1), by bisection (t + 1 <= T/2).
function cosh_mass(values, t)
    T = length(values)
    target = values[t+1] / values[t+2]
    f(m) = cosh(m * (t - T / 2)) / cosh(m * (t + 1 - T / 2)) - target
    low, high = 1e-8, 10.0
    f(low) < 0 < f(high) || error("no cosh mass at t=$t")
    for _ in 1:200
        middle = (low + high) / 2
        f(middle) < 0 ? (low = middle) : (high = middle)
    end
    return (low + high) / 2
end

cosh_plateau(values, window) = sum(cosh_mass(values, t) for t in window) / length(window)

# Lowest eigenvalue of the real symmetric operator `apply` by Lanczos with full
# reorthogonalization, started from the constant vector.
function lowest_eigenvalue(apply, shape; steps=80)
    basis = [fill(1 / sqrt(prod(shape)), shape)]
    α, β = Float64[], Float64[]
    for k in 1:steps
        w = apply(basis[k])
        push!(α, dot(basis[k], w))
        for v in basis
            w .-= dot(v, w) .* v
        end
        b = norm(w)
        (b < 1e-12 || k == steps) && break
        push!(β, b)
        push!(basis, w ./ b)
    end
    return eigmin(SymTridiagonal(α, β[1:length(α)-1]))
end

# The stencil of every sample must hold F at potential_time - 1 to potential_time + 2.
function estimate(samples, L; mass_window=MASS_WINDOW, shift_window=SHIFT_WINDOW, potential_time=POTENTIAL_TIME,
        thermal=THERMAL, a1=A1, norm=NORM)
    pion = average(samples, :pion)
    pipi = average(samples, :pipi)
    stencil = [sum(s.stencil[k] for s in samples) / length(samples) for k in 1:4]
    if thermal
        T = length(pion)
        m = cosh_plateau(pion, mass_window)
        pion = pion ./ (1 .+ exp.(-m .* (T .- 2 .* (0:T-1))))
        pipi = pipi[1:end-1] .- pipi[2:end]
        stencil = [stencil[k] .- stencil[k+1] for k in 1:3]
    else
        m = plateau(pion, mass_window)
    end
    norm == "exp" && (pion = exp.(-m .* (0:length(pion)-1)))
    ratio = pipi ./ pion[eachindex(pipi)] .^ 2
    ΔE = plateau(ratio, shift_window)
    a_L = luscher_scattering_length(ΔE, m, L)
    a1 && (stencil = a1plus_project.(stencil[1:3]))
    t = potential_time + 1
    R = [stencil[k] ./ pion[t-2+k]^2 for k in 1:3]
    result = lo_potential_equal_mass(R[1], R[2], R[3]; mass=m)
    all(result.valid) || error("R vanishes on some sites; the potential is not defined there")
    V = real.(result.potential)
    H(ψ) = -laplacian_periodic(ψ, (1.0, 1.0, 1.0)) ./ m .+ V .* ψ
    E0 = lowest_eigenvalue(H, size(V))
    p2 = m * E0
    ΔE_V = 2 * sqrt(m^2 + p2) - 2m
    a_V = luscher_scattering_length(ΔE_V, m, L)
    return vcat([m, ΔE, a_L, ΔE_V, a_V, V[1, 1, 1], V[2, 1, 1], V[3, 1, 1]], vec(V))
end

# Sites (0 ≤ x ≤ y ≤ z ≤ L/2, coordinates from r = 0) representing the cubic orbits.
orbit_sites(L) = [(x, y, z) for z in 0:L÷2 for y in 0:z for x in 0:y]

function write_potential_samples(path, central, estimates, L)
    sites = orbit_sites(L)
    open(path, "w") do io
        println(io, "# sample 0 is the central value, 1.. the delete-one jackknife samples; columns: sample, m, V at the sites")
        println(io, "sites ", join(("$(x),$(y),$(z)" for (x, y, z) in sites), " "))
        for (k, e) in enumerate([[central]; estimates])
            V = reshape(e[9:end], L, L, L)
            println(io, k - 1, " ", e[1], " ", join((V[x+1, y+1, z+1] for (x, y, z) in sites), " "))
        end
    end
end

# V(r) at every relative position, with r the periodic minimal-image distance.
function write_potential_csv(path, central, stderr, L)
    V = reshape(central[9:end], L, L, L)
    E = reshape(stderr[9:end], L, L, L)
    open(path, "w") do io
        println(io, "x,y,z,r,V,V_err")
        for index in CartesianIndices(V)
            d = [min(i - 1, L - i + 1) for i in Tuple(index)]
            @printf(io, "%d,%d,%d,%.10f,%.10e,%.10e\n", d..., norm(d), V[index], E[index])
        end
    end
end

function stage6_analysis_main()
    isempty(ARGS) && error("usage: analyze_stage6_pipi.jl DATASET_DIR...")
    datasets = read_nbs_dataset.(ARGS)
    shape = size(first(datasets).four_point)[1:3]
    allequal(size(d.four_point) for d in datasets) || error("datasets have different shapes")
    shape[1] == shape[2] == shape[3] || error("expected a cubic box")
    L = shape[1]
    samples = binned(observables.(datasets), BIN)
    @printf("configurations=%d bin=%d L=%d nt=%d mass_window=%s shift_window=%s potential_time=%d thermal=%d a1=%d\n",
        length(datasets), BIN, L, size(first(datasets).four_point, 4), MASS_WINDOW, SHIFT_WINDOW, POTENTIAL_TIME,
        THERMAL, A1)
    NORM == "pion" || println("norm=", NORM)
    names = ["m_pi", "dE_correlator", "a_luscher", "dE_potential", "a_potential",
        "V(0)", "V(1,0,0)", "V(2,0,0)"]
    central = estimate(samples, L)
    if length(samples) >= 2
        estimates = [estimate(samples[setdiff(eachindex(samples), [i])], L) for i in eachindex(samples)]
        summary = jackknife_summary(estimates)
        for (k, name) in enumerate(names)
            @printf("%-14s %.8e +- %.2e\n", name, central[k], summary.stderr[k])
        end
        # The two shifts share the configurations, so their difference has its own jackknife error.
        difference = jackknife_summary([e[2] - e[4] for e in estimates])
        @printf("%-14s %.8e +- %.2e\n", "dE_corr-dE_V", central[2] - central[4], difference.stderr)
        csv = get(ENV, "HALQCD_STAGE6_POTENTIAL_CSV", "")
        isempty(csv) || write_potential_csv(csv, central, summary.stderr, L)
        samples_path = get(ENV, "HALQCD_STAGE6_POTENTIAL_SAMPLES", "")
        isempty(samples_path) || write_potential_samples(samples_path, central, estimates, L)
    else
        for (k, name) in enumerate(names)
            @printf("%-14s %.8e (one configuration, no error)\n", name, central[k])
        end
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && stage6_analysis_main()
