"""
    gauge_group(U)

Gauge group of a JuliaQCD gauge field `U` (a vector of the four link fields).
The methods are defined by the JuliaQCD package extension, which loads when
`Gaugefields` and `LatticeDiracOperators` are loaded. An `SU(N)` field with
`N` colors gives `SU{N}()`. A package that adds a group with its own field
type defines a more specific method.
"""
function gauge_group end

"""
    juliaqcd_links(U)

Links of a JuliaQCD gauge field `U` as a `ComplexF64` array with axes
`(row, column, μ, x, y, z, t)`, the layout taken by [`stout_smear`](@ref),
[`average_plaquette`](@ref) and [`wilson_clover_point_propagator`](@ref).
Defined by the JuliaQCD package extension.
"""
function juliaqcd_links end

"""
    juliaqcd_point_propagator(U; κ, kwargs...)

Point-source quark propagator computed with the JuliaQCD packages. The Dirac
operator is built from the same parameter dictionary as the Wilson branch of
`QCDMeasurements.Pion_correlator_measurement`, so the same keyword values give
the same operator. Defined by the JuliaQCD package extension.

`fermiontype` is `"Wilson"` (default) or `"WilsonClover"`; the clover operator
needs an explicit `cSW`. LatticeDiracOperators.jl 0.6.3 adds
`+κ c_SW Σ_{μ<ν} σ_μν F_μν` with `σ_μν = [γ_μ, γ_ν]/2`, the opposite of the
standard sign of Bridge++ and [`wilson_clover_point_propagator`](@ref); its
PR #80 with the proposed sign fix adds `-κ c_SW Σ σ_μν F_μν`, the standard one.
Both remove the trace of `F_μν`, which Bridge++ keeps, so for `SU(N ≥ 3)` neither
equals [`wilson_clover_point_propagator`](@ref). In the 1.x series this path
(non-MPILattice fields) stops in Gaugefields 1.1.11. The result therefore depends
on the installed version.
QCDMeasurements.jl 0.2.13 has no clover pion measurement to compare with.

`BoundaryCondition[μ]` multiplies the hops across the boundary in direction
μ, so `0` is a Dirichlet wall.

`U` may be a legacy Gaugefields field or an MPILattice field of the JuliaQCD v1
path (`Gaugefields.gauge_configuration`, on one process). For an MPILattice field
the boundary phases are set on the fermion field, because its Dirac operator
ignores the parameter, and the clover operator is the LatticeMatrices.jl one with
the standard sign and F_μν = (Q − Q†)/8 like [`wilson_clover_point_propagator`](@ref).
LatticeMatrices.jl rejects the phase 0, so a temporal Dirichlet wall is made on a
copy of the links by zeroing the links U_4(x, t=NT) that cross the time boundary.
The hopping term is then the same wall; the clover term differs from a phase-0
wall on the two time slices next to it. `method_CG` picks the LatticeDiracOperators.jl
solver (`"bicg"` by default, or `"bicgstab"`, which is much faster here; the
even-odd `"preconditiond_bicgstab"` does not run in LatticeDiracOperators
0.6.3). `source_smearing=(alpha=, steps=)` replaces the point by
[`wuppertal_smear`](@ref) of it (JuliaQCD has no quark-source smearing); the
sink stays a point. `source_type=:wall` fills the time slice
`source_position[4]` with the same color and spin; it is meaningful only for
links fixed to Coulomb gauge, for example with `Gaugefields.gaugefixing!` and
`D_fix=3` (check with [`coulomb_gauge_residual`](@ref)). Each source is
normalized before the solve, so `eps_CG`, a threshold on the squared residual
norm, is relative. With `progress=io` one line per solve (source color and spin,
seconds, relative residual) is written to `io`; it does not enter the result.

Returns a named tuple `(data, metadata)`. `data` has axes
`(sink_color, sink_spin, source_color, source_spin, x, y, z, t)`, with the
color axes of length `NC` of the gauge field.
"""
function juliaqcd_point_propagator end

"""
    juliaqcd_pipi_i2_nbs(U; κ, sample_id, source_positions=[(1, 1, 1, 1)],
                         kwargs...)

`I=2` two-pseudoscalar NBS dataset for one JuliaQCD gauge configuration `U`.
For each source position it computes [`juliaqcd_point_propagator`](@ref) and
passes the propagators to [`pipi_i2_nbs_dataset`](@ref). The metadata records
the gauge group, the field type, the Dirac parameters, the solver stopping
convention, the largest external residual, and the package versions. The
remaining keyword arguments are those of `juliaqcd_point_propagator`.
Defined by the JuliaQCD package extension.
"""
function juliaqcd_pipi_i2_nbs end

"""
    juliaqcd_two_nucleon_nbs(U; κ, sample_id, source_state, sink_state=source_state,
                             source_positions=[(1, 1, 1, 1)], source_type=:point,
                             source_smearing=nothing, nonrelativistic=false,
                             color_invariant=nothing, kwargs...)

Proton-neutron NBS dataset ([`two_nucleon_nbs_dataset`](@ref)) of one JuliaQCD
gauge configuration. The propagators come from
[`juliaqcd_point_propagator`](@ref) with the same keywords, and the nucleon
operators use the γ matrices of LatticeDiracOperators.jl. For `SU(3)` the
color invariant defaults to `ε_abc`; pass `baryon_invariant(group)` for other
groups. Defined by the JuliaQCD package extension.
"""
function juliaqcd_two_nucleon_nbs end

"""
    HAL_NBS_measurement(U; channel="pipi_I2", sample_id, kwargs...)

HAL QCD four-point measurement in the form of the QCDMeasurements.jl
measurements. It returns a subtype of `QCDMeasurements.AbstractMeasurement`,
and `measure(m, U)` returns a `Measurement_output` whose value is an
[`NBSCorrelatorDataset`](@ref). The keyword names and defaults follow
`Pion_correlator_measurement`. Defined by the extension that loads with
`Gaugefields`, `LatticeDiracOperators` and `QCDMeasurements`.
"""
function HAL_NBS_measurement end

function _check_point_propagator(propagator::AbstractArray{<:Number,8})
    ncolor, nspin = size(propagator, 1), size(propagator, 2)
    size(propagator, 3) == ncolor && size(propagator, 4) == nspin ||
        throw(
            DimensionMismatch(
                "propagator axes must be (Nc,Nspin,Nc,Nspin,x,y,z,t)",
            ),
        )
    return ncolor, nspin
end

"""
    pion_correlator_point_source(propagator)

Zero-momentum pseudoscalar correlator from a point-source propagator,

```text
C(t) = Σ_x Σ_{a,α,b,β} |S[a,α,b,β](x,t)|²,
```

which uses `γ₅`-hermiticity. The color sum runs over the length of the color
axes, so the function serves every gauge group. This is the normalization of
`QCDMeasurements.Pion_correlator_measurement` for Wilson fermions.
"""
function pion_correlator_point_source(propagator::AbstractArray{<:Number,8})
    _check_point_propagator(propagator)
    nt = size(propagator, 8)
    correlator = zeros(real(float(eltype(propagator))), nt)
    for t in 1:nt
        correlator[t] = sum(abs2, selectdim(propagator, 8, t))
    end
    return correlator
end
