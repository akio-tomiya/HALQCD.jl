#!/usr/bin/env julia

# Pion mass by GEVP for the κ tuning of the Kurth reproduction
# (multi-basis GEVP with stability gates). Input: the lines
# "kappa config sink source C(0) ... C(T-1)" of the tune_gevp mode of
# scripts/kurth_pipi_production.jl (antiperiodic quarks, so C(t) = C(T - t)).
#
#   julia scripts/kurth_kappa_gevp.jl FILE... [--target 0.548]
#
# Per κ, every configuration's matrix is symmetrized in time
# ((C(t) + C(T-t))/2) and in the operators ((C_ij + C_ji)/2); the ensemble average is
# normalized by its diagonal at t0 and the GEVP C(t) v = λ C(t0) v gives the principal
# correlator λ₀(t) (largest eigenvalue). m comes from a fit of λ₀(t) to
# A cosh(m (t - T/2)) over a window. Every delete-one jackknife sample repeats the
# whole pipeline. The quoted mass uses the primary basis, t0 and window; the others
# are the stability checks (gates):
#   basis agreement: every candidate basis within BASIS_LIMIT of the primary,
#   t0 stability and window spread: within the same limit, in units of the error.
# Finally m² = c₁ + c₂/κ is fitted on every jackknife sample and solved for m = target.
using LinearAlgebra
using Printf

# GEVP_BASIS="P,W10,W30" is the primary basis; the others are P with each smeared
# operator, and P alone.
const PRIMARY_BASIS = String.(split(get(ENV, "GEVP_BASIS", "P,W10,W30"), ","))
const BASES = [[PRIMARY_BASIS]; [["P", l] for l in PRIMARY_BASIS if l != "P"]; [["P"]]]
# GEVP_T0S="3,5", GEVP_WINDOWS="10:40,15:40,15:50,20:50"; the first of each is primary.
window(text) = (p = parse.(Int, split(text, ":")); p[1]:p[2])
const T0S = parse.(Int, split(get(ENV, "GEVP_T0S", "3,5"), ","))
const PRIMARY_T0 = first(T0S)
const WINDOWS = window.(split(get(ENV, "GEVP_WINDOWS", "15:40,10:40,15:50,20:50"), ","))
const PRIMARY_WINDOW = first(WINDOWS)
const LIMIT_SIGMA = 1.0   # gates: differences below one jackknife error

function read_matrices(paths)
    data = Dict{Float64,Dict{Int,Dict{Tuple{String,String},Vector{Float64}}}}()
    for path in paths, line in eachline(path)
        f = split(line)
        κ, n = parse(Float64, f[1]), parse(Int, f[2])
        get!(get!(data, κ, Dict{Int,Dict{Tuple{String,String},Vector{Float64}}}()), n,
            Dict{Tuple{String,String},Vector{Float64}}())[(f[3], f[4])] = parse.(Float64, f[5:end])
    end
    return data
end

# Time- and operator-symmetrized matrix of one configuration, t = 0 .. T/2.
function symmetrized(entries, labels)
    T = length(first(values(entries)))
    reflect(C) = [(C[t+1] + C[mod(T - t, T)+1]) / 2 for t in 0:T÷2]
    M = [zeros(length(labels), length(labels)) for _ in 0:T÷2]
    for (a, i) in enumerate(labels), (b, j) in enumerate(labels)
        c = (reflect(entries[(i, j)]) .+ reflect(entries[(j, i)])) ./ 2
        for t in eachindex(M)
            M[t][a, b] = c[t]
        end
    end
    return M, T
end

function principal(M, t0)
    d = [1 / sqrt(M[t0+1][k, k]) for k in axes(M[1], 1)]
    N = [Diagonal(d) * m * Diagonal(d) for m in M]
    B = Symmetric(N[t0+1])
    return [t <= t0 ? NaN : maximum(eigvals(Symmetric(N[t+1]), B)) for t in 0:length(M)-1]
end

# Least squares of λ(t)/(A cosh(m(t - T/2))) - 1 over the window; A is solved for each m.
function cosh_fit(λ, window, T)
    ts = collect(window)
    function objective(m)
        f = [cosh(m * (t - T / 2)) for t in ts]
        y = [λ[t+1] for t in ts]
        A = sum(f[k] / y[k] for k in eachindex(ts)) / sum((f[k] / y[k])^2 for k in eachindex(ts))
        return sum((A * f[k] / y[k] - 1)^2 for k in eachindex(ts))
    end
    low, high = 0.05, 3.0
    for _ in 1:200   # golden section
        a = high - 0.618034 * (high - low)
        b = low + 0.618034 * (high - low)
        objective(a) < objective(b) ? (high = b) : (low = a)
    end
    return (low + high) / 2
end

function masses(per_config, labels_all, subset)
    T = per_config[1][2]
    average = [sum(c[1][t] for c in per_config[subset]) / length(subset) for t in eachindex(per_config[1][1])]
    result = Dict{Tuple{Vector{String},Int,UnitRange{Int}},Float64}()
    for basis in BASES
        idx = [findfirst(==(l), labels_all) for l in basis]
        M = [m[idx, idx] for m in average]
        for t0 in T0S
            λ = principal(M, t0)
            for w in WINDOWS
                first(w) > t0 || continue
                result[(basis, t0, w)] = cosh_fit(λ, w, T)
            end
        end
    end
    return result
end

jackknife_error(values, centre) =
    sqrt((length(values) - 1) / length(values) * sum((v - centre)^2 for v in values))

function main(arguments)
    target, paths = 0.548, String[]
    i = 1
    while i <= length(arguments)
        arguments[i] == "--target" ? (target = parse(Float64, arguments[i+1]); i += 2) :
            (push!(paths, arguments[i]); i += 1)
    end
    data = read_matrices(paths)
    kappas = sort(collect(keys(data)))
    configs = sort(collect(intersect((Set(keys(data[κ])) for κ in kappas)...)))
    labels_all = PRIMARY_BASIS
    @printf("kappas=%s configs=%s primary basis=%s t0=%d window=%s\n", kappas, configs, PRIMARY_BASIS,
        PRIMARY_T0, PRIMARY_WINDOW)
    key = (PRIMARY_BASIS, PRIMARY_T0, PRIMARY_WINDOW)
    central_m, sample_m = Float64[], Vector{Float64}[]
    for κ in kappas
        per_config = [symmetrized(data[κ][n], labels_all) for n in configs]
        all_idx = collect(eachindex(configs))
        central = masses(per_config, labels_all, all_idx)
        samples = [masses(per_config, labels_all, setdiff(all_idx, [k])) for k in all_idx]
        err(k) = jackknife_error([s[k] for s in samples], central[k])
        e = err(key)
        @printf("\nkappa=%.5f m_pi=%.5f +- %.5f (primary)\n", κ, central[key], e)
        gates = true
        for basis in BASES, t0 in T0S, w in WINDOWS
            first(w) > t0 || continue
            k = (basis, t0, w)
            d = central[k] - central[key]
            # error of the difference, from the same jackknife samples
            de = jackknife_error([s[k] - s[key] for s in samples], d)
            pass = abs(d) <= LIMIT_SIGMA * max(e, de) || basis == ["P"]
            gates &= pass
            @printf("  basis=%-14s t0=%d window=%-5s m=%.5f +- %.5f diff=%+.5f %s\n", join(basis, "_"), t0,
                "$(first(w)):$(last(w))", central[k], err(k), d, pass ? "" : "GATE_FAIL")
        end
        @printf("  gates_pass=%s (the point-only row is shown for comparison and not gated)\n", gates)
        push!(central_m, central[key])
        push!(sample_m, [s[key] for s in samples])
    end
    if length(kappas) >= 2
        fit(m) = (X = hcat(ones(length(kappas)), 1 ./ kappas); c = X \ (m .^ 2);
                  (c[2] / (target^2 - c[1]), -c[2] / c[1]))
        κs, κc = fit(central_m)
        stars = [fit([sample_m[k][j] for k in eachindex(kappas)]) for j in eachindex(configs)]
        @printf("\nkappa(m_pi a=%.4f)=%.6f +- %.6f kappa_c=%.6f +- %.6f\n", target, κs,
            jackknife_error(first.(stars), κs), κc, jackknife_error(last.(stars), κc))
    end
end

main(ARGS)
