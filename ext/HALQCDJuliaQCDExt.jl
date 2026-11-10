module HALQCDJuliaQCDExt

using HALQCD
using Gaugefields
using LatticeDiracOperators
using LinearAlgebra

HALQCD.gauge_group(U::AbstractVector{<:AbstractGaugefields{NC,4}}) where {NC} = SU{NC}()

function HALQCD.juliaqcd_links(U::AbstractVector{<:AbstractGaugefields{NC,4}}) where {NC}
    length(U) == 4 || throw(DimensionMismatch("expected the four link fields"))
    links = zeros(ComplexF64, NC, NC, 4, U[1].NX, U[1].NY, U[1].NZ, U[1].NT)
    for μ in 1:4, t in 1:U[1].NT, z in 1:U[1].NZ, y in 1:U[1].NY, x in 1:U[1].NX,
        b in 1:NC, a in 1:NC
        links[a, b, μ, x, y, z, t] = U[μ][a, b, x, y, z, t]
    end
    return links
end

function _package_version(mod::Module)
    version = pkgversion(mod)
    return isnothing(version) ? "unknown" : string(version)
end

# Gauge fields of the JuliaQCD v1 path (Gaugefields.gauge_configuration), stored
# as LatticeMatrices.jl MPILattice fields.
_mpilattice(U) = isdefined(Gaugefields, :Gaugefields_4D_MPILattice) &&
    U[1] isa Gaugefields.Gaugefields_4D_MPILattice

# Wilson fermion field for the solves. An MPILattice field carries its boundary
# phases itself, and its Dirac operator ignores the "boundarycondition" parameter,
# so the phases are set here. HALQCD.jl reads and writes the whole lattice, so
# only one process is supported.
function _wilson_field(U, boundary, mpi::Bool)
    mpi || return Initialize_pseudofermion_fields(U[1], "Wilson"; nowing=true)
    # Initialize_WilsonFermion asserts nowing == true for MPILattice links (its
    # message says "should be false").
    x = LatticeDiracOperators.Initialize_WilsonFermion(U[1]; nowing=true, boundarycondition=collect(boundary))
    Tuple(x.f.PN) == (U[1].NX, U[1].NY, U[1].NZ, U[1].NT) || throw(ArgumentError(
        "HALQCD.jl supports MPILattice fields on a single process (process_grid=(1, 1, 1, 1))"))
    return x
end

_core(x) = view(x.f.A, :, :, ntuple(d -> (x.f.nw+1):(x.f.nw+x.f.PN[d]), 4)...)

# LatticeMatrices.jl rejects the boundary phase 0 (it divides by the phase for
# backward wraps). For MPILattice links a temporal Dirichlet wall is therefore made
# on a copy of the links by zeroing U_4(x, t=NT), the links that cross the time
# boundary, and solving with phase 1. The hopping term is then exactly the phase-0
# wall; the clover term differs from a phase-0 implementation on the two time
# slices next to the wall.
function _dirichlet_time_links(U)
    W = copy_configuration(U)
    f = W[4].U
    nw = f.nw
    view(f.A, :, :, (nw+1):(nw+f.PN[1]), (nw+1):(nw+f.PN[2]), (nw+1):(nw+f.PN[3]), nw + f.PN[4]) .= 0
    Gaugefields.LatticeMatricesCompat.mark_lattice_dirty!(f)
    Gaugefields.set_wing_U!(W[4])
    return W
end

# Field contents as an array with axes (color, spin, x, y, z, t).
function _field_array(x, mpi::Bool, NC, (nx, ny, nz, nt))
    mpi && return Array(_core(x))
    return [x[c, ix, iy, iz, it, s] for c in 1:NC, s in 1:4, ix in 1:nx, iy in 1:ny, iz in 1:nz, it in 1:nt]
end

function _set_field!(x, ψ::AbstractArray{<:Number,6}, mpi::Bool)
    clear_fermion!(x)
    if mpi
        _core(x) .= ψ
        set_wing_fermion!(x)
    else
        for index in findall(!iszero, ψ)
            c, s, ix, iy, iz, it = Tuple(index)
            setindex_global!(x, ψ[index], c, ix, iy, iz, it, s)
        end
    end
    return x
end

function HALQCD.juliaqcd_point_propagator(
    U::AbstractVector{<:AbstractGaugefields{NC,4}};
    κ,
    r=1,
    fermiontype="Wilson",
    eps_CG=1e-14,
    MaxCGstep=5000,
    BoundaryCondition=[1, 1, 1, -1],
    method_CG=nothing,
    cSW=nothing,
    source_position::NTuple{4,Int}=(1, 1, 1, 1),
    source_smearing::Union{Nothing,NamedTuple}=nothing,
    source_type::Symbol=:point,
    verbose_level=0,
    progress::Union{Nothing,IO}=nothing,
) where {NC}
    source_type in (:point, :wall) ||
        throw(ArgumentError("source_type must be :point or :wall"))
    source_type == :wall && !isnothing(source_smearing) &&
        throw(ArgumentError("a wall source is not smeared"))
    fermiontype in ("Wilson", "WilsonClover") ||
        throw(ArgumentError("only Wilson and WilsonClover fermions are supported so far"))
    fermiontype == "WilsonClover" && isnothing(cSW) &&
        throw(ArgumentError("WilsonClover needs an explicit cSW"))
    fermiontype == "Wilson" && !isnothing(cSW) &&
        throw(ArgumentError("cSW is only used with fermiontype=\"WilsonClover\""))
    nx, ny, nz, nt = U[1].NX, U[1].NY, U[1].NZ, U[1].NT
    all(1 .<= source_position .<= (nx, ny, nz, nt)) ||
        throw(BoundsError((nx, ny, nz, nt), source_position))

    # In LatticeDiracOperators 0.6.3 the even-odd solver fails for every Wilson
    # field: clear_fermion!(::WilsonFermion_4D_nowing, iseven) does not parse as
    # intended (`iy = i3 -` followed by the loop) and the fields with wings have no
    # shift_fermion with a boundary condition.
    method_CG == "preconditiond_bicgstab" && throw(ArgumentError(
        "method_CG=\"preconditiond_bicgstab\" does not run in LatticeDiracOperators 0.6.3"))

    # Same keys and values as the Wilson branch of
    # QCDMeasurements.Pion_correlator_measurement. LatticeDiracOperators has
    # no faster version of the clover operator.
    parameters = Dict{String,Any}(
        "Dirac_operator" => fermiontype,
        "κ" => κ,
        "r" => r,
        "faster version" => fermiontype == "Wilson",
        "eps_CG" => eps_CG,
        "verbose_level" => verbose_level,
        "MaxCGstep" => MaxCGstep,
        "boundarycondition" => BoundaryCondition,
    )
    isnothing(method_CG) || (parameters["method_CG"] = method_CG)
    isnothing(cSW) || (parameters["cSW"] = cSW)

    mpi = _mpilattice(U)
    operator_links, field_boundary = U, BoundaryCondition
    if mpi && any(iszero, BoundaryCondition)
        all(!iszero, BoundaryCondition[1:3]) || throw(ArgumentError(
            "for MPILattice links only the time direction can have a Dirichlet (0) boundary"))
        operator_links = _dirichlet_time_links(U)
        field_boundary = [BoundaryCondition[1:3]; 1]
        # The operator checks that this parameter equals the phases of the field.
        parameters["boundarycondition"] = field_boundary
        parameters["dirichlet_implementation"] = "zeroed U_4(x, t=NT) links, phase 1"
    end
    source = _wilson_field(operator_links, field_boundary, mpi)
    solution = similar(source)
    applied = similar(source)
    D = Dirac_operator(operator_links, source, parameters)
    nspin = 4
    dims = (nx, ny, nz, nt)
    data = zeros(ComplexF64, NC, nspin, NC, nspin, nx, ny, nz, nt)
    residuals = zeros(Float64, NC, nspin)
    solve_seconds = zeros(Float64, NC, nspin)
    # JuliaQCD has no quark-source smearing, so a smeared source is built with
    # HALQCD.wuppertal_smear and loaded into the LatticeDiracOperators field.
    # LatticeDiracOperators stops on the absolute squared residual, so the source
    # is normalized and the solution scaled back: eps_CG is then a relative
    # threshold for point and smeared sources alike.
    # A wall source fills the time slice of source_position[4] with the same
    # color and spin; it is gauge covariant only in a fixed gauge (Coulomb
    # gauge for the HAL QCD wall source), which the caller must impose on U.
    links = isnothing(source_smearing) ? nothing : HALQCD.juliaqcd_links(U)

    for source_color in 1:NC, source_spin in 1:nspin
        b = zeros(ComplexF64, NC, nspin, nx, ny, nz, nt)
        if source_type == :wall
            b[source_color, source_spin, :, :, :, source_position[4]] .= 1
        elseif isnothing(source_smearing)
            b[source_color, source_spin, source_position...] = 1
        else
            b[source_color, source_spin, source_position...] = 1
            b = HALQCD.wuppertal_smear(links, b; source_smearing...)
        end
        scale = norm(b)
        b ./= scale
        _set_field!(source, b, mpi)
        clear_fermion!(solution)
        solve_seconds[source_color, source_spin] =
            @elapsed solve_DinvX!(solution, D, source)
        mul!(applied, D, solution)
        x = _field_array(solution, mpi, NC, dims)
        residuals[source_color, source_spin] = norm(_field_array(applied, mpi, NC, dims) .- b) / norm(b)
        if !isnothing(progress)
            println(progress, "solve source_color=", source_color, " source_spin=", source_spin,
                " seconds=", round(solve_seconds[source_color, source_spin]; digits=1),
                " residual=", residuals[source_color, source_spin])
            flush(progress)
        end
        for sink_spin in 1:nspin, sink_color in 1:NC
            data[sink_color, sink_spin, source_color, source_spin, :, :, :, :] .=
                scale .* view(x, sink_color, sink_spin, :, :, :, :)
        end
    end

    metadata = Dict{String,Any}(
        "gauge_group" => string(HALQCD.gauge_group(U)),
        "gauge_field_type" => string(nameof(typeof(U[1]))),
        "ncolor" => NC,
        "lattice_shape" => (nx, ny, nz, nt),
        "source_position" => source_position,
        "source_smearing" => source_smearing,
        "source_type" => source_type,
        "dirac_parameters" => parameters,
        # LatticeDiracOperators' bicg compares the squared residual norm with
        # eps_CG, so eps_CG is a squared-residual threshold.
        "eps_CG_convention" => "squared residual norm for the normalized source (LatticeDiracOperators bicg)",
        "external_residuals" => residuals,
        "maximum_external_residual" => maximum(residuals),
        "solve_seconds" => solve_seconds,
        "Gaugefields_version" => _package_version(Gaugefields),
        "LatticeDiracOperators_version" => _package_version(LatticeDiracOperators),
    )
    return (data=data, metadata=metadata)
end

function HALQCD.juliaqcd_pipi_i2_nbs(
    U::AbstractVector{<:AbstractGaugefields{NC,4}};
    κ,
    sample_id::AbstractString,
    source_positions::AbstractVector{<:NTuple{4,Int}}=[(1, 1, 1, 1)],
    BoundaryCondition=[1, 1, 1, -1],
    source_smearing::Union{Nothing,NamedTuple}=nothing,
    kwargs...,
) where {NC}
    results = [
        HALQCD.juliaqcd_point_propagator(
            U; κ, BoundaryCondition, source_position=position, source_smearing, kwargs...,
        )
        for position in source_positions
    ]
    parameters = first(results).metadata["dirac_parameters"]
    time_boundary = BoundaryCondition[4]
    temporal_boundary_condition =
        time_boundary == -1 ? "antiperiodic-quark" :
        time_boundary == 1 ? "periodic-quark" :
        time_boundary == 0 ? "dirichlet-quark" :
        throw(ArgumentError("temporal boundary condition must be 1, -1 or 0"))
    # NBS metadata must be TOML scalars or one-dimensional sequences.
    metadata = (
        gauge_group=string(HALQCD.gauge_group(U)),
        gauge_field_type=string(nameof(typeof(U[1]))),
        fermion_type=parameters["Dirac_operator"],
        kappa=Float64(parameters["κ"]),
        wilson_r=Float64(parameters["r"]),
        clover_coefficient=Float64(get(parameters, "cSW", 0.0)),
        boundary_condition=Tuple(Int.(BoundaryCondition)),
        source_smearing=isnothing(source_smearing) ? "none" : "wuppertal",
        source_smearing_alpha=isnothing(source_smearing) ? 0.0 : Float64(source_smearing.alpha),
        source_smearing_steps=isnothing(source_smearing) ? 0 : Int(source_smearing.steps),
        solver_method=string(get(parameters, "method_CG", "LatticeDiracOperators default")),
        solver_squared_residual_threshold=Float64(parameters["eps_CG"]),
        solver_max_steps=Int(parameters["MaxCGstep"]),
        eps_CG_convention=first(results).metadata["eps_CG_convention"],
        maximum_external_residual=maximum(
            result.metadata["maximum_external_residual"] for result in results
        ),
        Gaugefields_version=_package_version(Gaugefields),
        LatticeDiracOperators_version=_package_version(LatticeDiracOperators),
        julia_version=string(VERSION),
    )
    return pipi_i2_nbs_dataset(
        [result.data for result in results],
        source_positions;
        sample_id,
        temporal_boundary_condition,
        metadata,
    )
end

# γ matrices of LatticeDiracOperators.jl, the basis of its propagators.
_ldo_gammas() = Tuple(LatticeDiracOperators.Dirac_operators.mk_gamma(1)[1][:, :, μ] for μ in 1:4)

function HALQCD.juliaqcd_two_nucleon_nbs(
    U::AbstractVector{<:AbstractGaugefields{NC,4}};
    κ,
    sample_id::AbstractString,
    source_state::AbstractMatrix,
    sink_state::AbstractMatrix=source_state,
    source_positions::AbstractVector{<:NTuple{4,Int}}=[(1, 1, 1, 1)],
    BoundaryCondition=[1, 1, 1, -1],
    source_smearing::Union{Nothing,NamedTuple}=nothing,
    source_type::Symbol=:point,
    nonrelativistic::Bool=false,
    color_invariant=nothing,
    channel::AbstractString="NN",
    kwargs...,
) where {NC}
    results = [
        HALQCD.juliaqcd_point_propagator(
            U; κ, BoundaryCondition, source_position=position, source_smearing,
            source_type, kwargs...,
        )
        for position in source_positions
    ]
    parameters = first(results).metadata["dirac_parameters"]
    time_boundary = BoundaryCondition[4]
    temporal_boundary_condition =
        time_boundary == -1 ? "antiperiodic-quark" :
        time_boundary == 1 ? "periodic-quark" :
        time_boundary == 0 ? "dirichlet-quark" :
        throw(ArgumentError("temporal boundary condition must be 1, -1 or 0"))
    source_kind = source_type == :wall ? "wall" :
        isnothing(source_smearing) ? "point" : "wuppertal"
    metadata = (
        gauge_group=string(HALQCD.gauge_group(U)),
        gauge_field_type=string(nameof(typeof(U[1]))),
        fermion_type=parameters["Dirac_operator"],
        kappa=Float64(parameters["κ"]),
        wilson_r=Float64(parameters["r"]),
        clover_coefficient=Float64(get(parameters, "cSW", 0.0)),
        boundary_condition=Tuple(Int.(BoundaryCondition)),
        source_smearing=isnothing(source_smearing) ? "none" : "wuppertal",
        source_smearing_alpha=isnothing(source_smearing) ? 0.0 : Float64(source_smearing.alpha),
        source_smearing_steps=isnothing(source_smearing) ? 0 : Int(source_smearing.steps),
        solver_method=string(get(parameters, "method_CG", "LatticeDiracOperators default")),
        solver_squared_residual_threshold=Float64(parameters["eps_CG"]),
        solver_max_steps=Int(parameters["MaxCGstep"]),
        eps_CG_convention=first(results).metadata["eps_CG_convention"],
        maximum_external_residual=maximum(
            result.metadata["maximum_external_residual"] for result in results
        ),
        gamma_basis="LatticeDiracOperators.Dirac_operators.mk_gamma(1)",
        Gaugefields_version=_package_version(Gaugefields),
        LatticeDiracOperators_version=_package_version(LatticeDiracOperators),
        julia_version=string(VERSION),
    )
    return two_nucleon_nbs_dataset(
        [result.data for result in results],
        source_positions;
        gammas=_ldo_gammas(),
        source_state,
        sink_state,
        sample_id,
        temporal_boundary_condition,
        color_invariant,
        nonrelativistic,
        channel,
        source_kind,
        metadata,
    )
end

end
