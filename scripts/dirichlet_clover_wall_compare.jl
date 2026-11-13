# Compare the two ways of putting a temporal Dirichlet wall into the clover Wilson
# operator, with the reference implementation of HALQCD.jl:
#   phase0 : boundary phase 0 (hops across the wall cut, F_μν from the unchanged
#            links), the LatticeMatrices.jl 1.2.10 behavior used from now on;
#   zeroed : periodic operator on links with U_4(x, t=T) = 0, used by the
#            extension before (the clover plaquettes across the wall vanish too).
# Prints the relative difference of the pion correlator C(t) at each time.
#
#   julia --project=. scripts/dirichlet_clover_wall_compare.jl [L T kappa]
using HALQCD
using LinearAlgebra
using Random

L = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
T = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 16
kappa = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.12
csw = 1.0
shape = (L, L, L, T)
rng = MersenneTwister(11)

function random_su3(rng, scale)
    a = scale * randn(rng, ComplexF64, 3, 3)
    h = (a - a') / 2
    h -= tr(h) / 3 * I
    return exp(h)
end

links = zeros(ComplexF64, 3, 3, 4, shape...)
for index in CartesianIndices(shape), μ in 1:4
    links[:, :, μ, Tuple(index)...] = random_su3(rng, 0.5)
end
zeroed = copy(links)
zeroed[:, :, 4, :, :, :, T] .= 0

source = (1, 1, 1, T ÷ 2 + 1)   # the wall is between t = T and 1, T/2 away
phase0 = wilson_clover_point_propagator(SU{3}(), links; kappa, csw,
    boundary_condition=(1, 1, 1, 0), source_position=source)
cut = wilson_clover_point_propagator(SU{3}(), zeroed; kappa, csw,
    boundary_condition=(1, 1, 1, 1), source_position=source)
pion(S) = [sum(abs2, view(S, :, :, :, :, :, :, :, t)) for t in 1:T]
a, b = pion(phase0.data), pion(cut.data)
println("shape=", shape, " kappa=", kappa, " csw=", csw, " source=", source,
    " residuals=", maximum(phase0.true_residuals), ", ", maximum(cut.true_residuals))
println("t  distance_to_wall  C_phase0  relative_difference")
for t in 1:T
    distance = min(t - 0.5, T + 0.5 - t)   # the wall sits between T and 1
    println(t, "  ", distance, "  ", a[t], "  ", abs(a[t] - b[t]) / a[t])
end
