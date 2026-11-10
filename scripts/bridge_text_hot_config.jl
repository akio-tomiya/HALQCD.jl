#!/usr/bin/env julia

# julia bridge_text_hot_config.jl NC NX NT OUTPUT
# Write a reproducible hot SU(NC) configuration on NX³×NT in the Bridge++ text format
# (Gaugefields.save_textdata), for comparing Wilson-clover implementations with Bridge++
# on the same links (scripts/bridge_clover_compare.jl). Prints the largest |U†U - 1|.
using Gaugefields
using LinearAlgebra
using Printf

function main()
    NC, NX, NT, output = parse(Int, ARGS[1]), parse(Int, ARGS[2]), parse(Int, ARGS[3]), ARGS[4]
    U = Initialize_Gaugefields(NC, 0, NX, NX, NX, NT; condition="hot", randomnumber="Reproducible")
    save_textdata(U, output)
    V = Initialize_Gaugefields(NC, 0, NX, NX, NX, NT; condition="cold")
    load_BridgeText!(output, V, [NX, NX, NX, NT], NC)
    sites = Iterators.product(1:NX, 1:NX, 1:NX, 1:NT)
    link(W, μ, x) = [W[μ][a, b, x...] for a in 1:NC, b in 1:NC]
    unitarity = maximum(norm(link(U, μ, x)' * link(U, μ, x) - I) for μ in 1:4, x in sites)
    roundtrip = maximum(norm(link(U, μ, x) - link(V, μ, x)) for μ in 1:4, x in sites)
    @printf("NC=%d lattice=%dx%dx%dx%d output=%s max|U†U-1|=%.2e max|reread-U|=%.2e\n", NC, NX, NX, NX, NT, output,
        unitarity, roundtrip)
end

main()
