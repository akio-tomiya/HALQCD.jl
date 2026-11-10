#!/usr/bin/env julia

# Pion effective mass log(C(t)/C(t+1)) of Kurth NBS datasets (wall source,
# Dirichlet wall), t counted from the source. With several datasets it prints the
# effective mass of the average correlator; the sources of a dataset are averaged. It also prints the effective energy
# shift log(R(t)/R(t+1)) of R = C_ππ/C_π², without errors (they come from
# scripts/analyze_stage6_pipi.jl).
#
#   julia scripts/kurth_meff.jl DATASET... [--window 10:40]
using HALQCD
using Printf

function main(arguments)
    window = 10:40
    paths = String[]
    i = 1
    while i <= length(arguments)
        if arguments[i] == "--window"
            a, b = parse.(Int, split(arguments[i+1], ":"))
            window = a:b
            i += 2
        else
            push!(paths, arguments[i])
            i += 1
        end
    end
    isempty(paths) && error("give at least one dataset")
    correlators = map(paths) do path
        dataset = read_nbs_dataset(path)
        ns = size(dataset.one_hadron_1, 2)
        [sum(real(dataset.one_hadron_1[t, k, 1]) for k in 1:ns) / ns for t in axes(dataset.one_hadron_1, 1)]
    end
    C = sum(correlators) / length(correlators)
    meff = [log(C[t+1] / C[t+2]) for t in 0:(length(C)-2)]
    # Effective energy shift of R(t) = C_ππ(t)/C_π(t)², C_ππ the zero-momentum
    # two-pion correlator (sum of the four-point function over r).
    pipi = sum(map(paths) do path
        dataset = read_nbs_dataset(path)
        ns = size(dataset.four_point, 5)
        [sum(real(sum(view(dataset.four_point, :, :, :, t, k, 1))) for k in 1:ns) / ns for t in axes(dataset.four_point, 4)]
    end) / length(paths)
    R = pipi ./ C[eachindex(pipi)] .^ 2
    dE = [log(R[t+1] / R[t+2]) for t in 0:(length(R)-2)]
    @printf("datasets=%d\n", length(paths))
    for t in 0:min(length(meff) - 1, last(window) + 5)
        @printf("t=%3d meff=%.4f dEeff=%.6f\n", t, meff[t+1], dE[t+1])
    end
    @printf("mean meff t=%d..%d: %.4f\n", first(window), last(window), sum(meff[t+1] for t in window) / length(window))
end

main(ARGS)
