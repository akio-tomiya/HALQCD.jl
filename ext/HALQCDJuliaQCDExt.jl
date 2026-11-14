module HALQCDJuliaQCDExt

using HALQCD
using Gaugefields
using LatticeDiracOperators
using LinearAlgebra

HALQCD.gauge_group(U::AbstractVector{<:AbstractGaugefields{NC,4}}) where {NC} = SU{NC}()

function HALQCD.juliaqcd_links(U::AbstractVector{<:AbstractGaugefields{NC,4}}) where {NC}
    length(U) == 4 || throw(DimensionMismatch("expected the four link fields"))
    links = zeros(ComplexF64, NC, NC, 4, U[1].NX, U[1].NY, U[1].NZ, U[1].NT)
    if _mpilattice(U) && _layout(U, true).nprocs > 1
        # Collective: every process receives the whole lattice
        # (LatticeMatrices.gather_and_bcast_matrix).
        for μ in 1:4
            links[:, :, μ, :, :, :, :] .= _latticematrices().gather_and_bcast_matrix(U[μ].U)
        end
        return links
    end
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

_latticematrices() = Base.loaded_modules[_LATTICEMATRICES]

# How the links `U` are split over MPI processes, in the terms of
# LatticeMatrices.jl: the process grid (get_PEs), the Cartesian coordinates of
# this process, the local block size, the first global site of the block
# (global_site_coordinates), and the communicator of Gaugefields.jl. Any process
# grid that Gaugefields.gauge_configuration accepts, including :auto, is used as
# given for the solves. The contractions need whole spatial slices, so the
# solutions are regrouped into time slices (`times`) after each solve.
function _layout(U, mpi::Bool)
    shape = (U[1].NX, U[1].NY, U[1].NZ, U[1].NT)
    serial = (nprocs=1, comm=nothing, root=true, dims=(1, 1, 1, 1), coords=(0, 0, 0, 0),
        local_size=shape, start=(1, 1, 1, 1), shape=shape, times=collect(1:shape[4]))
    mpi || return serial
    lattice = U[1].U
    LM = _latticematrices()
    dims = Tuple(LM.get_PEs(lattice))
    prod(dims) == 1 && return serial
    local_size = Tuple(lattice.PN)
    coords = Tuple(lattice.coords)
    comm = isdefined(Gaugefields, :gauge_communicator) ?
        something(Gaugefields.gauge_communicator(U), lattice.comm) : lattice.comm
    return (nprocs=prod(dims), comm=comm, root=lattice.myrank == 0, dims=dims,
        coords=coords, local_size=local_size,
        start=Tuple(LM.global_site_coordinates(lattice, (1, 1, 1, 1))), shape=shape,
        times=HALQCD._assigned_times(coords, dims, local_size))
end

# Global ranges of the local block.
_block_ranges(layout) = ntuple(d -> layout.start[d] .+ (0:(layout.local_size[d] - 1)), 4)

function _sum_over_processes!(layout, values::AbstractArray)
    layout.nprocs == 1 && return values
    return HALQCD._sum_over_processes!(layout.comm, values)
end

# Whole spatial slices at layout.times from the local blocks of all processes.
function _time_slices(layout, block::AbstractArray)
    layout.nprocs == 1 && return block
    return HALQCD._time_slices(layout.comm, block, layout.coords, layout.dims, layout.shape)
end

_sum_over_processes(layout, value::Real) = _sum_over_processes!(layout, [Float64(value)])[1]

# Wilson fermion field for the solves. An MPILattice field carries its boundary
# phases itself, and its Dirac operator ignores the "boundarycondition" parameter,
# so the phases are set here.
function _wilson_field(U, boundary, mpi::Bool)
    mpi || return Initialize_pseudofermion_fields(U[1], "Wilson"; nowing=true)
    # Initialize_WilsonFermion asserts nowing == true for MPILattice links (its
    # message says "should be false").
    return LatticeDiracOperators.Initialize_WilsonFermion(U[1]; nowing=true, boundarycondition=collect(boundary))
end

_core(x) = view(x.f.A, :, :, ntuple(d -> (x.f.nw+1):(x.f.nw+x.f.PN[d]), 4)...)

# The boundary phase 0 on MPILattice fields (a Dirichlet wall that cuts the hops
# and leaves the links, so the clover term is built from the links as given) needs
# LatticeMatrices.jl 1.2.10 and LatticeDiracOperators.jl 1.1.5. Older versions
# reject the phase 0 or divide by it.
const _LATTICEMATRICES = Base.PkgId(Base.UUID("dd6a91e4-736f-4540-ac85-13822ca7b545"), "LatticeMatrices")

function _check_zero_phase_support()
    ldo = pkgversion(LatticeDiracOperators)
    lm_module = get(Base.loaded_modules, _LATTICEMATRICES, nothing)
    lm = isnothing(lm_module) ? nothing : pkgversion(lm_module)
    (isnothing(ldo) || ldo >= v"1.1.5") && (isnothing(lm) || lm >= v"1.2.10") ||
        throw(ArgumentError("a Dirichlet (0) boundary on MPILattice fields needs " *
            "LatticeDiracOperators 1.1.5 and LatticeMatrices 1.2.10 or later " *
            "(found $(something(ldo, "unknown")) and $(something(lm, "unknown")))"))
    return nothing
end

# Field contents as an array with axes (color, spin, x, y, z, t). For an
# MPILattice field these are the time slices of this process.
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
    mpi = _mpilattice(U)
    layout = _layout(U, mpi)
    times = layout.times
    ranges = _block_ranges(layout)

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

    if mpi && any(iszero, BoundaryCondition)
        all(!iszero, BoundaryCondition[1:3]) || throw(ArgumentError(
            "for MPILattice links only the time direction can have a Dirichlet (0) boundary"))
        _check_zero_phase_support()
        parameters["dirichlet_implementation"] = "boundary phase 0 (hops cut, clover from the unchanged links)"
    end
    source = _wilson_field(U, BoundaryCondition, mpi)
    solution = similar(source)
    applied = similar(source)
    D = Dirac_operator(U, source, parameters)
    nspin = 4
    dims = (nx, ny, nz, nt)
    local_nt = length(times)
    data = zeros(ComplexF64, NC, nspin, NC, nspin, nx, ny, nz, local_nt)
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
    # On a split field the smeared source is built on the whole lattice on every
    # process (juliaqcd_links gathers the links) and each process keeps its block.
    links = isnothing(source_smearing) ? nothing : HALQCD.juliaqcd_links(U)
    local_source = map(-, source_position, layout.start) .+ 1
    owns_source_time = source_position[4] in ranges[4]
    owns_source_site = all(source_position[d] in ranges[d] for d in 1:4)

    for source_color in 1:NC, source_spin in 1:nspin
        b = zeros(ComplexF64, NC, nspin, layout.local_size...)
        if source_type == :wall
            owns_source_time && (b[source_color, source_spin, :, :, :, local_source[4]] .= 1)
        elseif isnothing(source_smearing)
            owns_source_site && (b[source_color, source_spin, local_source...] = 1)
        else
            whole = zeros(ComplexF64, NC, nspin, nx, ny, nz, nt)
            whole[source_color, source_spin, source_position...] = 1
            b .= view(HALQCD.wuppertal_smear(links, whole; source_smearing...), :, :, ranges...)
        end
        scale = sqrt(_sum_over_processes(layout, sum(abs2, b)))
        b ./= scale
        _set_field!(source, b, mpi)
        clear_fermion!(solution)
        solve_seconds[source_color, source_spin] =
            @elapsed solve_DinvX!(solution, D, source)
        mul!(applied, D, solution)
        x = _time_slices(layout, _field_array(solution, mpi, NC, dims))
        # b has norm 1 over the whole lattice
        residuals[source_color, source_spin] =
            sqrt(_sum_over_processes(layout, sum(abs2, _field_array(applied, mpi, NC, dims) .- b)))
        if !isnothing(progress) && layout.root
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
        "local_times" => times,
        "process_grid" => layout.dims,
        "mpi_processes" => layout.nprocs,
        "Gaugefields_version" => _package_version(Gaugefields),
        "LatticeDiracOperators_version" => _package_version(LatticeDiracOperators),
    )
    return (data=data, metadata=metadata)
end

# Contract each source as soon as its propagator is solved, so one propagator
# is held at a time. `contract(data)` returns the four-point function and the
# one-hadron correlator on the local time slices; they are placed in arrays of
# the whole lattice and summed over the processes.
function _contract_sources(contract, U, source_positions; propagator_kwargs...)
    nx, ny, nz, nt = U[1].NX, U[1].NY, U[1].NZ, U[1].NT
    layout = _layout(U, _mpilattice(U))
    four_points = []
    one_hadrons = []
    first_metadata = nothing
    maximum_residual = 0.0
    for position in source_positions
        result = HALQCD.juliaqcd_point_propagator(U; source_position=position, propagator_kwargs...)
        local_four_point, local_one_hadron = contract(result.data)
        isnothing(first_metadata) && (first_metadata = result.metadata)
        maximum_residual = max(maximum_residual, result.metadata["maximum_external_residual"])
        result = nothing
        four_point = zeros(eltype(local_four_point), nx, ny, nz, nt)
        one_hadron = zeros(eltype(local_one_hadron), nt)
        four_point[:, :, :, layout.times] .= local_four_point
        one_hadron[layout.times] .= local_one_hadron
        push!(four_points, _sum_over_processes!(layout, four_point))
        push!(one_hadrons, _sum_over_processes!(layout, one_hadron))
    end
    return (four_points=[four_points...], one_hadrons=[one_hadrons...],
        metadata=first_metadata, maximum_residual=maximum_residual, nprocs=layout.nprocs)
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
    contracted = _contract_sources(U, source_positions; κ, BoundaryCondition, source_smearing, kwargs...) do data
        HALQCD.pipi_i2_four_point(data), HALQCD.pion_correlator_point_source(data)
    end
    parameters = contracted.metadata["dirac_parameters"]
    temporal_boundary_condition = _temporal_boundary_name(BoundaryCondition)
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
        eps_CG_convention=contracted.metadata["eps_CG_convention"],
        maximum_external_residual=contracted.maximum_residual,
        mpi_processes=contracted.nprocs,
        Gaugefields_version=_package_version(Gaugefields),
        LatticeDiracOperators_version=_package_version(LatticeDiracOperators),
        julia_version=string(VERSION),
    )
    return HALQCD._pipi_i2_nbs_dataset(
        contracted.four_points, contracted.one_hadrons, source_positions,
        (U[1].NX, U[1].NY, U[1].NZ, U[1].NT), NC, 4;
        sample_id,
        temporal_boundary_condition,
        metadata,
    )
end

function _temporal_boundary_name(BoundaryCondition)
    time_boundary = BoundaryCondition[4]
    return time_boundary == -1 ? "antiperiodic-quark" :
        time_boundary == 1 ? "periodic-quark" :
        time_boundary == 0 ? "dirichlet-quark" :
        throw(ArgumentError("temporal boundary condition must be 1, -1 or 0"))
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
    gammas = _ldo_gammas()
    contracted = _contract_sources(
        U, source_positions; κ, BoundaryCondition, source_smearing, source_type, kwargs...,
    ) do data
        HALQCD._two_nucleon_absolute(
            data; gammas, source_state, sink_state, color_invariant, nonrelativistic,
        )
    end
    parameters = contracted.metadata["dirac_parameters"]
    temporal_boundary_condition = _temporal_boundary_name(BoundaryCondition)
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
        eps_CG_convention=contracted.metadata["eps_CG_convention"],
        maximum_external_residual=contracted.maximum_residual,
        mpi_processes=contracted.nprocs,
        gamma_basis="LatticeDiracOperators.Dirac_operators.mk_gamma(1)",
        Gaugefields_version=_package_version(Gaugefields),
        LatticeDiracOperators_version=_package_version(LatticeDiracOperators),
        julia_version=string(VERSION),
    )
    return HALQCD._two_nucleon_nbs_dataset(
        contracted.four_points, contracted.one_hadrons, source_positions,
        (U[1].NX, U[1].NY, U[1].NZ, U[1].NT), NC, 4;
        source_state,
        sink_state,
        sample_id,
        temporal_boundary_condition,
        nonrelativistic,
        channel,
        source_kind,
        metadata,
    )
end

end
