#!/usr/bin/env julia

# julia fit_stage6_potential.jl SAMPLES_FILE [GAUSSIANS] [RMAX]
# Fit V(r) of every jackknife sample (written by analyze_stage6_pipi.jl with
# HALQCD_STAGE6_POTENTIAL_SAMPLES) with Σ_i c_i exp(-(r/b_i)²) on the orbit sites
# with r ≤ RMAX (default 2 Gaussians, RMAX = L/2), weighted by the jackknife error,
# as in the HAL QCD I=2 ππ studies (Kawai et al. arXiv:1711.01883 Eq. (25),
# Akahoshi et al. arXiv:1904.09549 Eq. (33)). Then solve the zero-energy radial
# equation u'' = m V u (reduced mass m/2) in infinite volume and read the
# scattering length from u ∝ r - a at large r (a > 0 for repulsion, the
# convention of luscher_scattering_length; the particle-physics a₀ is -a).
#
#   HALQCD_STAGE6_FIT_SHRINKAGE  λ in [0, 1]: minimize the correlated χ² with the
#                                covariance (1-λ)C + λ diag(C) of V at the fitted sites,
#                                C from the jackknife samples, kept the same in every
#                                sample's fit (default 1, the uncorrelated fit; λ = 0
#                                needs more samples than sites)
#   HALQCD_STAGE6_FIT_CONSTANT   "1" adds a constant to the model; the scattering length
#                                uses the Gaussians only (a check on an r-independent
#                                remainder of V)
using HALQCD
using LinearAlgebra
using Printf

function read_samples(path)
    lines = filter(l -> !startswith(l, "#"), readlines(path))
    sites = [Tuple(parse.(Int, split(s, ","))) for s in split(lines[1])[2:end]]
    rows = [parse.(Float64, split(l)) for l in lines[2:end]]
    return sites, [r[2] for r in rows], [r[3:end] for r in rows]
end

# An odd number of parameters means a constant term p[end] after the Gaussians.
gaussians_only(p) = p[1:2*(length(p)÷2)]
model(p, r) = sum(p[2i-1] * exp(-r^2 / (p[2i]^2 + 1e-12)) for i in 1:length(p)÷2) + (isodd(length(p)) ? p[end] : 0.0)

# Levenberg–Marquardt with a numerical Jacobian; χ² = |W(model - y)|² with W = L⁻¹ for the
# covariance LLᵀ (diagonal W = 1/σ for the uncorrelated fit).
function fit(r, y, W, p0; iterations=500)
    p, λ = copy(p0), 1e-3
    residual(p) = W * (model.(Ref(p), r) .- y)
    χ2(p) = sum(abs2, residual(p))
    for _ in 1:iterations
        res = residual(p)
        J = hcat([(residual(p .+ h .* (1:length(p) .== k)) .- res) ./ h
                  for (k, h) in enumerate(1e-7 .* max.(abs.(p), 1e-3))]...)
        A = J' * J
        all(isfinite, J) || error("non-finite Jacobian at p = $p")
        # A floor on the damping keeps the step defined when a parameter drops out
        # (a Gaussian whose amplitude goes to 0 leaves its width with a zero column of J).
        step = -(A + λ * Diagonal(max.(diag(A), 1e-12 * maximum(diag(A))))) \ (J' * res)
        if all(isfinite, step) && χ2(p .+ step) < χ2(p)
            p .+= step; λ /= 3
            norm(step) < 1e-12 * (1 + norm(p)) && break
        else
            λ *= 5
            λ > 1e12 && break   # no step lowers χ²: at the minimum to rounding
        end
    end
    return p, χ2(p)
end

# Zero-energy S wave with reduced mass m/2, so u'' = m V u (Numerov, HALQCD.s_wave_scattering_length).
function scattering_length(p, m; rend=40.0, h=1e-2)
    radii = collect(0:h:rend)
    q = gaussians_only(p)
    return s_wave_scattering_length(radii, model.(Ref(q), radii); reduced_mass=m / 2, tail_start=rend - 10).value
end

# Jackknife error of x[2:end] (x[1] is the central value).
function jackerr(x)
    n = length(x) - 1
    return sqrt((n - 1) / n * sum(abs2, x[2:end] .- sum(x[2:end]) / n))
end

# Fit every sample of a samples file; returns the per-sample results and the fit setup.
function fit_samples(path; gaussians=2, rmax=nothing,
    shrinkage=parse(Float64, get(ENV, "HALQCD_STAGE6_FIT_SHRINKAGE", "1")),
    constant=get(ENV, "HALQCD_STAGE6_FIT_CONSTANT", "0") == "1")
    sites, masses, potentials = read_samples(path)
    L = 2 * maximum(maximum.(sites))
    rmax = something(rmax, L / 2)
    r = [norm(s) for s in sites]
    keep = r .<= rmax
    jack = [v[keep] for v in potentials[2:end]]
    n = length(jack)
    mean = sum(jack) ./ n
    C = (n - 1) / n .* sum(v -> (v .- mean) * (v .- mean)', jack)
    σ = sqrt.(diag(C))
    0 <= shrinkage <= 1 || error("HALQCD_STAGE6_FIT_SHRINKAGE must be in [0, 1]")
    shrunk = Symmetric((1 - shrinkage) .* C .+ shrinkage .* Diagonal(diag(C)))
    W = shrinkage == 1 ? Diagonal(1 ./ σ) : inv(cholesky(shrunk).L)
    p0 = vec(hcat([[potentials[1][1] / gaussians, 3.0 * i / gaussians] for i in 1:gaussians]...))
    constant && push!(p0, 0.0)
    results = map(eachindex(potentials)) do k
        p, χ2 = fit(r[keep], potentials[k][keep], W, p0)
        (p=p, χ2=χ2, a=scattering_length(p, masses[k]), m=masses[k])
    end
    mineig = shrinkage == 1 ? 1.0 : eigmin(Symmetric(Diagonal(1 ./ σ) * shrunk * Diagonal(1 ./ σ)))
    return (results=results, points=count(keep), rmax=rmax, gaussians=gaussians, constant=constant,
        samples=n, shrinkage=shrinkage, mineig=mineig)
end

function fit_main()
    gaussians = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
    rmax = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : nothing
    f = fit_samples(ARGS[1]; gaussians, rmax)
    results, n, shrinkage, constant = f.results, f.samples, f.shrinkage, f.constant
    central = results[1]
    @printf("points=%d rmax=%.2f gaussians=%d constant=%d samples=%d shrinkage=%.3f chi2/dof=%.3f\n", f.points, f.rmax,
        gaussians, constant, n, shrinkage, central.χ2 / (f.points - length(central.p)))
    shrinkage == 1 || @printf("smallest eigenvalue of the shrunk correlation matrix %.3e\n", f.mineig)
    for i in 1:gaussians
        @printf("c%d %.6e +- %.2e  b%d %.6e +- %.2e\n", i, central.p[2i-1], jackerr([x.p[2i-1] for x in results]),
            i, abs(central.p[2i]), jackerr([abs(x.p[2i]) for x in results]))
    end
    constant && @printf("constant %.6e +- %.2e\n", central.p[end], jackerr([x.p[end] for x in results]))
    @printf("a_infinite_volume %.6e +- %.2e\n", central.a, jackerr([x.a for x in results]))
    @printf("m_pi*a            %.6e +- %.2e\n", central.m * central.a, jackerr([x.m * x.a for x in results]))
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && fit_main()
