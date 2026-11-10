#!/usr/bin/env julia

# julia bridge_clover_compare.jl CONFIG NC NX NT LOG_CSW=PATH...
# The Wilson-clover pion correlator of Bridge++ against HALQCD.jl and LatticeDiracOperators.jl
# v1 on the same gauge configuration in the Bridge++ text format. Each LOG_CSW=PATH is a
# Bridge++ log of Spectrum.Clover.Hadron2ptFunction with a local source at the origin,
# hopping parameter 0.12, boundary condition [1,1,1,-1], no gauge fixing, and
# clover_coefficient LOG_CSW. Its first "PS <-- PS" correlator is compared with
#   standard:  HALQCD.jl's operator, -κ c_SW Σ σF with F the traceless anti-Hermitian part;
#   trace:     the same with F = (Q - Q†)/8, its trace kept (as LatticeMatrices.jl);
# at csw = ±LOG_CSW, from Σ_{x,colors,spins} |S(x, t)|². The LatticeMatrices.jl path of
# LatticeDiracOperators.jl v1 (MPILattice fields) is checked through the residual
# |D_LM S - b|/|b| of the standard and trace propagators, and the legacy path
# (non-MPILattice fields) through juliaqcd_point_propagator.
using HALQCD
using Gaugefields
using JACC
using LatticeDiracOperators
using LatticeMatrices
using LinearAlgebra
using Printf

JACC.@init_backend

const KAPPA = 0.12
const BC = (1, 1, 1, -1)
const GAMMAS = Tuple(LatticeDiracOperators.Dirac_operators.mk_gamma(1)[1][:, :, μ] for μ in 1:4)

function bridge_ps(path, nt)
    lines = readlines(path)
    start = findfirst(l -> occursin("Flavor combination = 1, 1", l), lines)
    k = findnext(l -> occursin("PS <-- PS correlator", l), lines, start)
    return [parse(Float64, split(lines[k+t])[2]) for t in 1:nt]
end

function operator(NC, links, csw; trace=false)
    op = HALQCD._WilsonCloverOperator(SU{NC}(), links; kappa=KAPPA, csw, r=1, boundary_condition=BC, gammas=GAMMAS)
    trace || iszero(csw) || return op
    shape = size(links)[4:7]
    Lk(d, x) = HALQCD._link(links, d, x)
    sh(x, d, s) = HALQCD._shifted(x, d, s, shape)
    for index in CartesianIndices(shape), (plane, (μ, ν)) in enumerate(HALQCD._CLOVER_PLANES)
        x = Tuple(index)
        xpμ, xpν, xmμ, xmν = sh(x, μ, 1), sh(x, ν, 1), sh(x, μ, -1), sh(x, ν, -1)
        Q = Lk(μ, x) * Lk(ν, xpμ) * Lk(μ, xpν)' * Lk(ν, x)' +
            Lk(ν, x) * Lk(μ, sh(xmμ, ν, 1))' * Lk(ν, xmμ)' * Lk(μ, xmμ) +
            Lk(μ, xmμ)' * Lk(ν, sh(xmμ, ν, -1))' * Lk(μ, sh(xmμ, ν, -1)) * Lk(ν, xmν) +
            Lk(ν, xmν)' * Lk(μ, xmν) * Lk(ν, sh(xpμ, ν, -1)) * Lk(μ, x)'
        op.field_strength[:, :, plane, x...] = (Q - Q') / 8
    end
    return op
end

# Columns S[:, :, c, s, sites...] for a point source at the origin.
function propagator(NC, op, shape)
    S = zeros(ComplexF64, NC, 4, NC, 4, shape...)
    for s in 1:4, c in 1:NC
        b = zeros(ComplexF64, NC, 4, shape...)
        b[c, s, 1, 1, 1, 1] = 1
        x, _, residual = HALQCD._cgnr(op, b; tolerance=1e-13, maximum_iterations=20_000)
        residual < 1e-11 || error("no convergence, residual $residual")
        S[:, :, c, s, ntuple(_ -> Colon(), 4)...] = x
    end
    return S
end

pion(S) = [sum(abs2, selectdim(S, 8, t)) for t in axes(S, 8)]

core(field) = @view field.A[:, :, ntuple(d -> (field.nw+1):(field.nw+field.PN[d]), length(field.PN))...]

# max over columns of |D_LM S - b| / |b| for the LatticeMatrices.jl Wilson-clover operator.
function lm_residual(UM, NC, csw, S, shape)
    x = Initialize_pseudofermion_fields(UM[1], "Wilson")
    D = Dirac_operator(UM, x, Dict{String,Any}("Dirac_operator" => "WilsonClover", "κ" => KAPPA, "cSW" => csw,
        "verbose_level" => 0))
    out = similar(x)
    worst = 0.0
    for s in 1:4, c in 1:NC
        ψ = Array(S[:, :, c, s, ntuple(_ -> Colon(), 4)...])
        substitute!(x.f, LatticeMatrix(ψ, 4, (1, 1, 1, 1); nw=1, phases=x.f.phases, comm0=x.f.comm))
        set_wing_fermion!(x)
        mul!(out, D, x)
        b = zeros(ComplexF64, NC, 4, shape...)
        b[c, s, 1, 1, 1, 1] = 1
        worst = max(worst, norm(Array(core(out.f)) .- b) / norm(b))
    end
    return worst
end

function main()
    config, NC, NX, NT = ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3]), parse(Int, ARGS[4])
    shape = (NX, NX, NX, NT)
    U = Initialize_Gaugefields(NC, 0, shape...; condition="cold")
    load_BridgeText!(config, U, collect(shape), NC)
    links = HALQCD.juliaqcd_links(U)
    UM = Initialize_Gaugefields(NC, 1, shape...; condition="cold", isMPILattice=true, PEs=(1, 1, 1, 1), verbose_level=0)
    load_BridgeText!(config, UM, collect(shape), NC)
    unitarity = maximum(norm(links[:, :, μ, x...]' * links[:, :, μ, x...] - I) for μ in 1:4, x in Iterators.product(axes(links)[4:7]...))
    @printf("config=%s NC=%d lattice=%s max|U†U-1|=%.2e LatticeDiracOperators=%s LatticeMatrices=%s\n", config, NC,
        shape, unitarity, pkgversion(LatticeDiracOperators), pkgversion(LatticeMatrices))
    rel(a, b) = maximum(abs.(a .- b) ./ abs.(b))
    for arg in ARGS[5:end]
        csw_text, path = split(arg, "="; limit=2)
        csw = parse(Float64, csw_text)
        bridge = bridge_ps(path, NT)
        cases = iszero(csw) ? [("standard", csw, false)] :
                [("standard", csw, false), ("standard", -csw, false), ("trace", csw, true), ("trace", -csw, true)]
        @printf("== Bridge++ clover_coefficient=%+.2f (%s)\n", csw, path)
        @printf("   Bridge PS(t): %s\n", join((@sprintf("%.10e", v) for v in bridge), " "))
        for (name, c, trace) in cases
            S = propagator(NC, operator(NC, links, c; trace), shape)
            mine = pion(S)
            @printf("   %-8s csw=%+.2f  max_t |PS/PS_Bridge - 1| = %.3e   LatticeMatrices residual %.3e\n", name, c,
                rel(mine, bridge), lm_residual(UM, NC, c, S, shape))
        end
        try
            legacy = HALQCD.juliaqcd_point_propagator(U; κ=KAPPA, fermiontype=iszero(csw) ? "Wilson" : "WilsonClover",
                cSW=iszero(csw) ? nothing : csw, eps_CG=1e-24, BoundaryCondition=collect(BC), source_position=(1, 1, 1, 1)).data
            @printf("   legacy LatticeDiracOperators cSW=%+.2f  max_t |PS/PS_Bridge - 1| = %.3e\n", csw,
                rel(pion(legacy), bridge))
        catch err
            @printf("   legacy LatticeDiracOperators cSW=%+.2f failed: %s\n", csw, first(sprint(showerror, err), 160))
        end
    end
end

main()
