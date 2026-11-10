#!/usr/bin/env julia

# κ for a target pion mass from the point-source correlators written by the tune mode
# of scripts/kurth_pipi_production.jl (KURTH_TUNE_OUT, antiperiodic time).
#
#   julia scripts/kurth_kappa_fit.jl FILE... [--window 15:50] [--target 0.548]
#
# For each κ, m is the average of the cosh effective mass over the window, with a
# delete-one jackknife over configurations (the same configurations for every κ).
# m² is then fitted linearly in 1/κ by least squares, m² = A (1/κ - 1/κ_c), on every
# jackknife sample, and solved for the κ with m = target.
using HALQCD
using LinearAlgebra
using Printf

function read_correlators(paths)
    data = Dict{Float64,Dict{Int,Vector{Float64}}}()
    for path in paths, line in eachline(path)
        fields = split(line)
        κ, n = parse(Float64, fields[1]), parse(Int, fields[2])
        get!(data, κ, Dict{Int,Vector{Float64}}())[n] = parse.(Float64, fields[3:end])
    end
    return data
end

# m with cosh(m(t - T/2))/cosh(m(t + 1 - T/2)) = C(t)/C(t+1), t counted from 0.
function cosh_mass(C, t)
    T = length(C)
    target = C[t+1] / C[t+2]
    f(m) = cosh(m * (t - T / 2)) / cosh(m * (t + 1 - T / 2)) - target
    low, high = 1e-8, 10.0
    f(low) < 0 < f(high) || return NaN
    for _ in 1:200
        middle = (low + high) / 2
        f(middle) < 0 ? (low = middle) : (high = middle)
    end
    return (low + high) / 2
end

plateau(C, window) = sum(cosh_mass(C, t) for t in window) / length(window)

function main(arguments)
    window, target, paths = 15:50, 0.548, String[]
    i = 1
    while i <= length(arguments)
        if arguments[i] == "--window"
            a, b = parse.(Int, split(arguments[i+1], ":")); window = a:b; i += 2
        elseif arguments[i] == "--target"
            target = parse(Float64, arguments[i+1]); i += 2
        else
            push!(paths, arguments[i]); i += 1
        end
    end
    data = read_correlators(paths)
    kappas = sort(collect(keys(data)))
    configs = sort(collect(intersect((Set(keys(data[κ])) for κ in kappas)...)))
    length(configs) >= 2 || error("need at least two configurations measured at every κ")
    @printf("kappas=%s configs=%s window=%s target=%.4f\n", kappas, configs, window, target)
    masses(subset) = [plateau(sum(data[κ][n] for n in subset) / length(subset), window) for κ in kappas]
    function kappa_star(m)
        x = 1 ./ kappas
        X = hcat(ones(length(x)), x)
        c = X \ (m .^ 2)          # m² = c₁ + c₂ / κ
        return (c[2] / (target^2 - c[1]), -c[2] / c[1])  # (κ at the target, κ_c)
    end
    central = masses(configs)
    samples = [masses(setdiff(configs, [n])) for n in configs]
    error_of(values, centre) = sqrt((length(values) - 1) / length(values) * sum((v - centre)^2 for v in values))
    for (k, κ) in enumerate(kappas)
        @printf("kappa=%.5f m_pi=%.5f +- %.5f\n", κ, central[k], error_of([s[k] for s in samples], central[k]))
    end
    if length(kappas) >= 2
        κs, κc = kappa_star(central)
        stars = [kappa_star(s) for s in samples]
        @printf("kappa(m_pi=%.4f)=%.6f +- %.6f kappa_c=%.6f +- %.6f\n", target, κs,
            error_of(first.(stars), κs), κc, error_of(last.(stars), κc))
    end
end

main(ARGS)
