#!/usr/bin/env julia

# julia check_stage6_analysis_excited_thermal.jl scripts/analyze_stage6_pipi.jl POTENTIAL_CSV
# Synthetic check of scripts/analyze_stage6_pipi.jl under two effects: elastic
# excited states and the wrap-around of the periodic time direction. V is a measured
# potential (a potential_t*.csv of the analysis, averaged over cubic orbits), H = -∇²/m + V
# with eigenpairs (E_n, ψ_n), and W_n + W_n²/(4m) = E_n.
#   ground:   F(r, τ) = ψ_0(r) ψ_0(0) e^{-(2m + W_0) τ},  C_π(τ) = e^{-mτ}
#   elastic:  F(r, τ) = Σ_n ψ_n(r) ψ_n(0) e^{-(2m + W_n) τ}  (relative motion started at r = 0)
#   thermal:  F(r, τ) + F(r, T - τ) + 2 e^{-mT}/L³,  C_π(τ) = e^{-mτ} + e^{-m(T-τ)}
#             (the constant is one pion forward and one backward across the boundary;
#             with V = 0 it makes Σ_r F = C_π² exactly)
# The time-dependent HAL method must return V and ΔE_V = W_0 at every t in the ground and
# elastic cases; the direct shift log(R(t)/R(t+1)) only in the ground case. Every case is
# analysed as it is and with the wrap-around removal of the analysis (thermal=true), which
# must bring the thermal cases back to the ground and elastic ones.
include(ARGS[1])
using LinearAlgebra
using Printf

const T = 32
const MASS = 0.55

function orbit_potential(path)
    rows = split.(readlines(path)[2:end], ",")
    L = round(Int, cbrt(length(rows)))
    orbit = Dict{NTuple{3,Int},Float64}()
    for row in rows
        orbit[Tuple(sort(parse.(Int, row[1:3])))] = parse(Float64, row[5])
    end
    d(i) = min(i - 1, L - i + 1)
    return [orbit[Tuple(sort([d(x), d(y), d(z)]))] for x in 1:L, y in 1:L, z in 1:L]
end

V = orbit_potential(ARGS[2])
L = size(V, 1)
N = L^3
H = zeros(N, N)
for j in 1:N
    e = zeros(L, L, L); e[j] = 1
    H[:, j] = vec(-laplacian_periodic(e, (1.0, 1.0, 1.0)) ./ MASS .+ V .* e)
end
spectrum = eigen(Symmetric(H))
W = 2MASS .* (sqrt.(1 .+ spectrum.values ./ MASS) .- 1)
ψ = spectrum.vectors
forward(τ, states) = reshape(ψ[:, states] * (ψ[1, states] .* exp.(-(2MASS .+ W[states]) .* τ)), L, L, L)

cases = [
    ("ground", τ -> forward(τ, 1:1), τ -> exp(-MASS * τ)),
    ("elastic", τ -> forward(τ, 1:N), τ -> exp(-MASS * τ)),
    ("thermal ground", τ -> forward(τ, 1:1) .+ forward(T - τ, 1:1) .+ 2exp(-MASS * T) / N * ψ[1, 1] * sum(ψ[:, 1]),
        τ -> exp(-MASS * τ) + exp(-MASS * (T - τ))),
    ("thermal", τ -> forward(τ, 1:N) .+ forward(T - τ, 1:N) .+ 2exp(-MASS * T) / N,
        τ -> exp(-MASS * τ) + exp(-MASS * (T - τ))),
]

@printf("L=%d T=%d m=%.3f E_0=%.6e W_0=%.6e W_1-W_0=%.4f V(0)=%.4f\n", L, T, MASS, spectrum.values[1], W[1], W[2] - W[1], V[1, 1, 1])
near = [sqrt(min(x - 1, L - x + 1)^2 + min(y - 1, L - y + 1)^2 + min(z - 1, L - z + 1)^2) <= 4 for x in 1:L, y in 1:L, z in 1:L]
for (name, four, pion) in cases, thermal in (false, true)
    sample(t) = (pion=[pion(τ) for τ in 0:T-1], pipi=[sum(four(τ)) for τ in 0:T-1],
        stencil=[four(t + δ) for δ in -1:2])
    base = sample(10)
    println("== ", name, thermal ? " (wrap-around removed)" : "")
    println("   t  m_eff(t)     dE_eff(t)")
    for t in 2:14
        r = estimate([base], L; mass_window=t:t, shift_window=t:t, potential_time=10, thermal)
        @printf("  %2d  %.6f  %+.6f\n", t, r[1], r[2])
    end
    println("  window  m            dE_direct    dE_V(t=10)   (W_0 = $(round(W[1], sigdigits=6)))")
    for w in [4:7, 5:8, 6:9, 7:10, 8:11, 8:14]
        r = estimate([base], L; mass_window=w, shift_window=w, potential_time=10, thermal)
        @printf("  %-6s  %.6f  %+.6f  %+.6f\n", "$(w.start):$(w.stop)", r[1], r[2], r[4])
    end
    println("  t   dE_V (window 7:10)  V(0)      V(1,0,0)  max|V-V_in| r<=4  all r")
    for t in [6, 8, 10, 12, 14]
        r = estimate([sample(t)], L; mass_window=7:10, shift_window=7:10, potential_time=t, thermal)
        Vout = reshape(r[9:end], L, L, L)
        @printf("  %2d  %+.6f           %.5f   %.5f   %.2e          %.2e\n", t, r[4], r[6], r[7],
            maximum(abs.(Vout .- V)[near]), maximum(abs.(Vout .- V)))
    end
end
