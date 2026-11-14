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
path (`Gaugefields.gauge_configuration`). For an MPILattice field
the boundary phases are set on the fermion field, because its Dirac operator
ignores the parameter, and the clover operator is the LatticeMatrices.jl one with
the standard sign and F_μν = (Q − Q†)/8 like [`wilson_clover_point_propagator`](@ref).
A temporal Dirichlet wall (`BoundaryCondition[4] = 0`) is the boundary phase 0 of
LatticeMatrices.jl, which needs LatticeMatrices.jl 1.2.10 and
LatticeDiracOperators.jl 1.1.5 or later: it cuts the hops across the wall and
leaves the links, so the clover term is built from the links as given, as in
[`wilson_clover_point_propagator`](@ref). `method_CG` picks the LatticeDiracOperators.jl
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

MPILattice fields may be split over MPI processes with any `process_grid` of
`Gaugefields.gauge_configuration`, including `:auto`, as in the MPI
documentation of Gaugefields.jl. Every process calls the function with the same
arguments, in the same order. The solves run on the field as it is split. The
contractions need whole spatial slices, so after each solve the solution is
regrouped: every time slice goes to one of the processes that hold a part of it,
and `data` holds the whole spatial slices at the global times
`metadata["local_times"]` (its last axis). For a split in time only these are
the time slices of the process. The residuals are those of the whole lattice.
A smeared source is built on the whole lattice on every process.
"""
function juliaqcd_point_propagator end

# Sum of `values` over the MPI processes of `comm`, in place. The MPI extension
# adds the method for `MPI.Comm`.
_sum_over_processes!(comm, values::AbstractArray) = throw(ArgumentError(
    "a field split over MPI processes needs MPI.jl loaded (using MPI)"))

# Whole spatial slices at `_assigned_times(coords, dims, size(block)[end-3:end])`
# from the local blocks `block` (axes (internal..., x, y, z, t)) of a field split
# over the process grid `dims` of a lattice of size `shape`. Collective over
# `comm`; the MPI extension adds the method for `MPI.Comm`.
_time_slices(comm, block, coords, dims, shape) = throw(ArgumentError(
    "a field split over MPI processes needs MPI.jl loaded (using MPI)"))

# Global times whose whole spatial slices the process at the Cartesian
# coordinates `coords` (zero-based) contracts. The processes that share a block
# of time slices take its slices in turn, so with a split in time only every
# process takes its own slices.
function _assigned_times(coords, dims, local_size)
    nspatial = dims[1] * dims[2] * dims[3]
    spatial_rank = coords[1] + dims[1] * (coords[2] + dims[2] * coords[3])
    first_time = coords[4] * local_size[4]
    return [first_time + j for j in 1:local_size[4] if (j - 1) % nspatial == spatial_rank]
end

"""
    juliaqcd_pipi_i2_nbs(U; κ, sample_id, source_positions=[(1, 1, 1, 1)],
                         kwargs...)

`I=2` two-pseudoscalar NBS dataset for one JuliaQCD gauge configuration `U`.
For each source position it computes [`juliaqcd_point_propagator`](@ref) and
contracts it as [`pipi_i2_nbs_dataset`](@ref) does; one propagator is kept at a
time. The metadata records the gauge group, the field type, the Dirac
parameters, the solver stopping convention, the largest external residual, and
the package versions. The remaining keyword arguments are those of
`juliaqcd_point_propagator`. On a field split over MPI processes, each process
contracts the time slices it receives, the correlators are summed over the
processes, and every process returns the same dataset of the whole lattice.
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
groups. A field split over MPI processes is handled as in
[`juliaqcd_pipi_i2_nbs`](@ref). Defined by the JuliaQCD package extension.
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
