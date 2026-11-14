using LinearAlgebra: I, norm, nullspace, svd

# Nucleon interpolating operators and two-nucleon spin states in the γ basis
# of the propagators. Everything is built from the four Euclidean γ matrices
# that the caller passes, so the same functions serve the chiral basis of
# HALQCD.jl and the basis of LatticeDiracOperators.jl.

function _check_gammas(gammas)
    length(gammas) == 4 && all(gamma -> size(gamma) == (4, 4), gammas) ||
        throw(DimensionMismatch("gammas must be four 4×4 matrices"))
    for μ in 1:4, ν in 1:4
        anticommutator = gammas[μ] * gammas[ν] + gammas[ν] * gammas[μ]
        norm(anticommutator - 2 * (μ == ν) * I) <= 1e-12 ||
            throw(ArgumentError("gammas must satisfy {γ_μ, γ_ν} = 2δ_μν"))
    end
    all(norm(gamma - gamma') <= 1e-12 for gamma in gammas) ||
        throw(ArgumentError("gammas must be Hermitian"))
    return gammas
end

"""
    charge_conjugation_matrix(gammas)

Charge-conjugation matrix `C` with `C γ_μ C⁻¹ = -γ_μᵀ` for the four Euclidean
γ matrices `gammas`. It is the one-dimensional solution of the linear
conditions, normalized to be unitary. The overall phase is free; it cancels
between a sink operator and its adjoint at the source. In the chiral basis of
HALQCD.jl and in the basis of LatticeDiracOperators.jl it is proportional to
`γ₄γ₂`, the choice of the HAL QCD papers.
"""
function charge_conjugation_matrix(gammas)
    _check_gammas(gammas)
    # Linear map C ↦ (C γ_μ + γ_μᵀ C) over the 16 entries of C.
    rows = Matrix{ComplexF64}[]
    for gamma in gammas
        push!(rows, kron(transpose(gamma), Matrix{ComplexF64}(I, 4, 4)) +
                    kron(Matrix{ComplexF64}(I, 4, 4), transpose(gamma)))
    end
    solutions = nullspace(reduce(vcat, rows))
    size(solutions, 2) == 1 ||
        throw(ArgumentError("the charge-conjugation condition has no unique solution"))
    # A unitary 4×4 matrix has Frobenius norm 2.
    C = reshape(solutions[:, 1], 4, 4) .* 2
    norm(C * C' - I) <= 1e-10 || throw(ArgumentError("charge-conjugation matrix is not unitary"))
    return C
end

"""
    nucleon_spin_basis(gammas)

Spin-up and spin-down Dirac spinors `(u↑, u↓)` of a positive-parity baryon at
rest: the common eigenvectors of `γ₄` (eigenvalue `+1`) and
`Σ₃ = -iγ₁γ₂` (eigenvalues `±1`). The phase of `u↑` makes its largest entry
real and positive, and `u↓ = Σ₋ u↑` with `Σ₋ = (Σ₁ - iΣ₂)/2`, so the Pauli
matrices in the basis `(u↑, u↓)` are the standard ones. With the Wilson
operator `1 - κ Σ_μ [(r - γ_μ) U_μ δ_{x+μ̂,y} + …]` of HALQCD.jl and
LatticeDiracOperators.jl, `(1 + γ₄)/2` projects on forward propagation.
"""
function nucleon_spin_basis(gammas)
    _check_gammas(gammas)
    γ1, γ2, γ3, γ4 = gammas
    Σ1 = -im * γ2 * γ3
    Σ2 = -im * γ3 * γ1
    Σ3 = -im * γ1 * γ2
    identity = Matrix{ComplexF64}(I, 4, 4)
    up = nullspace(vcat(γ4 - identity, Σ3 - identity))
    size(up, 2) == 1 || throw(ArgumentError("spin-up spinor is not unique"))
    u_up = up[:, 1]
    largest = u_up[findmax(abs, u_up)[2]]
    u_up .*= abs(largest) / largest
    u_down = (Σ1 - im * Σ2) / 2 * u_up
    norm(u_down) > 0.5 || throw(ArgumentError("spin lowering failed"))
    u_down ./= norm(u_down)
    return (u_up, u_down)
end

"""
    nucleon_spin_tensors(gammas; spin, nonrelativistic=false)

Sink and source spin tensors of the nucleon operator

```text
N_s = ε_abc (q₁ᵀ_a C γ₅ q₂_b) (u_s† q₃_c),
```

the proton `p = ε (uᵀCγ₅d) u` for flavors `(u, d, u)` and the neutron
`n = ε (uᵀCγ₅d) d` for `(u, d, d)`. `spin` is `1` (up) or `2` (down) in the
basis of [`nucleon_spin_basis`](@ref), which also fixes the positive-parity
projection of the third quark. With `nonrelativistic=true` the diquark is
`Cγ₅(1 + γ₄)/2`, the non-relativistic operator of Doi and Endres (2013) and
Iritani et al. (2016), which has fewer Wick terms.

The source tensor is that of the Euclidean adjoint `N̄_s = N_s† γ₄` written
in the order `q̄₃ q̄₂ q̄₁` of [`local_color_singlet_correlator`](@ref): the
diquark `γ₄ (Cγ₅)* γ₄ᵀ` and the spinor `γ₄ u_s = u_s`. Returns
`(sink, source)`.
"""
function nucleon_spin_tensors(gammas; spin::Integer, nonrelativistic::Bool=false)
    spin in (1, 2) || throw(ArgumentError("spin must be 1 (up) or 2 (down)"))
    γ1, γ2, γ3, γ4 = _check_gammas(gammas)
    γ5 = γ1 * γ2 * γ3 * γ4
    C = charge_conjugation_matrix(gammas)
    diquark = C * γ5
    if nonrelativistic
        diquark = diquark * (I + γ4) / 2
    end
    spinor = nucleon_spin_basis(gammas)[spin]
    sink = zeros(ComplexF64, 4, 4, 4)
    source = zeros(ComplexF64, 4, 4, 4)
    source_diquark = γ4 * conj(diquark) * transpose(γ4)
    source_spinor = γ4 * spinor
    for γ in 1:4, β in 1:4, α in 1:4
        sink[α, β, γ] = diquark[α, β] * conj(spinor[γ])
        source[α, β, γ] = source_diquark[α, β] * source_spinor[γ]
    end
    return (sink, source)
end

"""
    two_nucleon_spin_state(S, Sz)

Coefficients `χ[s₁, s₂]` of the two-nucleon spin state `|S, Sz⟩` in the basis
`(↑, ↓)` of [`nucleon_spin_basis`](@ref): the singlet `(↑↓ - ↓↑)/√2` for
`S = 0` and the triplet `↑↑`, `(↑↓ + ↓↑)/√2`, `↓↓` for `S = 1`.
"""
function two_nucleon_spin_state(S::Integer, Sz::Integer)
    χ = zeros(ComplexF64, 2, 2)
    if S == 0 && Sz == 0
        χ[1, 2], χ[2, 1] = 1 / sqrt(2), -1 / sqrt(2)
    elseif S == 1 && Sz == 1
        χ[1, 1] = 1
    elseif S == 1 && Sz == 0
        χ[1, 2], χ[2, 1] = 1 / sqrt(2), 1 / sqrt(2)
    elseif S == 1 && Sz == -1
        χ[2, 2] = 1
    else
        throw(ArgumentError("(S, Sz) must be (0, 0) or (1, -1:1)"))
    end
    return χ
end

const PROTON_FLAVORS = (:u, :d, :u)
const NEUTRON_FLAVORS = (:u, :d, :d)

"""
    two_nucleon_four_point(propagators; gammas, source_state, sink_state,
                           color_invariant=nothing, nonrelativistic=false)

Spatially resolved proton-neutron four-point function

```text
F(r, t) = Σ_x ⟨ [p n]_sink(x + r, x; t) [p n]_source† ⟩,
[p n]_χ(x + r, x) = Σ_{s₁ s₂} χ[s₁, s₂] p_{s₁}(x + r) n_{s₂}(x),
```

from [`two_baryon_four_point`](@ref). `propagators` maps `:u` and `:d` to
propagators (the same array in the isospin limit). `source_state` and
`sink_state` are 2×2 coefficient matrices such as
[`two_nucleon_spin_state`](@ref); a sink state with one nonzero entry gives a
single spin component `(s₁, s₂)`, which the tensor-force analysis needs. The
source is the adjoint `n̄ p̄` with conjugated coefficients.

In the isospin limit a `p(x + r) n(x)` sink contains both isospins; the part
even under `r → -r` is `I = 1` for the spin singlet and `I = 0` for the
triplet, and the `A₁⁺` projection keeps only that part.
"""
function two_nucleon_four_point(
    propagators::AbstractDict;
    gammas,
    source_state::AbstractMatrix,
    sink_state::AbstractMatrix=source_state,
    color_invariant::Union{Nothing,ColorInvariant}=nothing,
    nonrelativistic::Bool=false,
)
    size(source_state) == size(sink_state) == (2, 2) ||
        throw(DimensionMismatch("spin states must be 2×2 coefficient matrices"))
    tensors = [nucleon_spin_tensors(gammas; spin, nonrelativistic) for spin in 1:2]
    sink_terms = [
        (sink_state[s1, s2], tensors[s1][1], tensors[s2][1])
        for s1 in 1:2, s2 in 1:2 if !iszero(sink_state[s1, s2])
    ]
    source_terms = [
        (conj(source_state[s1, s2]), tensors[s1][2], tensors[s2][2])
        for s1 in 1:2, s2 in 1:2 if !iszero(source_state[s1, s2])
    ]
    flavors = (PROTON_FLAVORS, NEUTRON_FLAVORS)
    return two_baryon_four_point(
        propagators, flavors, flavors, vec(sink_terms), vec(source_terms);
        color_invariant,
    )
end

"""
    nucleon_two_point(propagators; gammas, spin=1, color_invariant=nothing,
                      nonrelativistic=false)

Zero-momentum proton correlator `Σ_x ⟨p_s(x, t) p̄_s(t₀)⟩` from
[`baryon_two_point`](@ref), with the time axis the absolute sink time.
"""
function nucleon_two_point(
    propagators::AbstractDict;
    gammas,
    spin::Integer=1,
    color_invariant::Union{Nothing,ColorInvariant}=nothing,
    nonrelativistic::Bool=false,
)
    sink, source = nucleon_spin_tensors(gammas; spin, nonrelativistic)
    return baryon_two_point(propagators, PROTON_FLAVORS, sink, source; color_invariant)
end

"""
    two_nucleon_nbs_dataset(propagators, source_positions; gammas, source_state,
                            sink_state=source_state, sample_id,
                            temporal_boundary_condition="antiperiodic-quark",
                            color_invariant=nothing, nonrelativistic=false,
                            channel="NN", lattice_spacing=(1.0, 1.0, 1.0, 1.0),
                            code_commit=<HALQCD.jl commit>, metadata=(;))

[`NBSCorrelatorDataset`](@ref) of the proton-neutron four-point function of
one gauge configuration. `propagators[k]` is the propagator from source `k`
(a point, smeared point, or wall at `source_positions[k]`), used for both `u`
and `d` (isospin limit). The four-point function is
[`two_nucleon_four_point`](@ref) and both one-hadron correlators are the
spin-averaged [`nucleon_two_point`](@ref). The values are complex: a single
sink spin component carries the angular dependence of the `D` wave.

The time axis is measured from the source time. With antiperiodic quarks a
three-quark correlator changes sign when the sink time wraps past the end of
the lattice; that sign is removed from the one-nucleon correlators, and the
six-quark function has none.
"""
function two_nucleon_nbs_dataset(
    propagators::AbstractVector{<:AbstractArray{<:Number,8}},
    source_positions::AbstractVector{<:NTuple{4,Integer}};
    gammas,
    source_state::AbstractMatrix,
    sink_state::AbstractMatrix=source_state,
    sample_id::AbstractString,
    temporal_boundary_condition::AbstractString="antiperiodic-quark",
    color_invariant::Union{Nothing,ColorInvariant}=nothing,
    nonrelativistic::Bool=false,
    channel::AbstractString="NN",
    source_kind::AbstractString="point",
    lattice_spacing=(1.0, 1.0, 1.0, 1.0),
    code_commit::AbstractString=_halqcd_code_commit(),
    metadata::NamedTuple=(;),
)
    isempty(propagators) && throw(ArgumentError("at least one propagator is required"))
    length(propagators) == length(source_positions) ||
        throw(DimensionMismatch("one source position is required per propagator"))
    shape = size(first(propagators))
    all(size(p) == shape for p in propagators) ||
        throw(DimensionMismatch("all propagators must have the same shape"))
    ncolor, nspin = _check_point_propagator(first(propagators))
    nx, ny, nz, nt = shape[5:8]
    absolute = [
        _two_nucleon_absolute(
            propagator; gammas, source_state, sink_state, color_invariant, nonrelativistic,
        )
        for propagator in propagators
    ]
    return _two_nucleon_nbs_dataset(
        first.(absolute), last.(absolute), source_positions, (nx, ny, nz, nt), ncolor, nspin;
        source_state, sink_state, sample_id, temporal_boundary_condition, nonrelativistic,
        channel, source_kind, lattice_spacing, code_commit, metadata,
    )
end

# Four-point function and spin-averaged one-nucleon correlator of one source,
# in absolute sink time, on the time slices of `propagator`.
function _two_nucleon_absolute(
    propagator::AbstractArray{<:Number,8};
    gammas, source_state, sink_state, color_invariant, nonrelativistic,
)
    quarks = Dict(:u => propagator, :d => propagator)
    four_point = two_nucleon_four_point(
        quarks; gammas, source_state, sink_state, color_invariant, nonrelativistic,
    )
    one_hadron = sum(
        nucleon_two_point(quarks; gammas, spin, color_invariant, nonrelativistic)
        for spin in 1:2
    ) ./ 2
    return four_point, one_hadron
end

# Dataset from the correlators of `_two_nucleon_absolute` over the whole
# lattice. The JuliaQCD extension calls it with correlators contracted on the
# time slices of each MPI process and summed over the processes.
function _two_nucleon_nbs_dataset(
    absolute_four_points::AbstractVector{<:AbstractArray{<:Number,4}},
    absolute_one_hadrons::AbstractVector{<:AbstractVector{<:Number}},
    source_positions::AbstractVector{<:NTuple{4,Integer}},
    (nx, ny, nz, nt)::NTuple{4,Int},
    ncolor::Integer,
    nspin::Integer;
    source_state::AbstractMatrix,
    sink_state::AbstractMatrix=source_state,
    sample_id::AbstractString,
    temporal_boundary_condition::AbstractString="antiperiodic-quark",
    nonrelativistic::Bool=false,
    channel::AbstractString="NN",
    source_kind::AbstractString="point",
    lattice_spacing=(1.0, 1.0, 1.0, 1.0),
    code_commit::AbstractString=_halqcd_code_commit(),
    metadata::NamedTuple=(;),
)
    antiperiodic = temporal_boundary_condition == "antiperiodic-quark"
    nsource = length(source_positions)
    length(absolute_four_points) == length(absolute_one_hadrons) == nsource ||
        throw(DimensionMismatch("one four-point and one one-hadron correlator are required per source"))
    four_point = zeros(ComplexF64, nx, ny, nz, nt, nsource, 1)
    one_hadron = zeros(ComplexF64, nt, nsource, 1)
    for (k, position) in enumerate(source_positions)
        absolute_four_point = absolute_four_points[k]
        absolute_one_hadron = absolute_one_hadrons[k]
        size(absolute_four_point) == (nx, ny, nz, nt) && length(absolute_one_hadron) == nt ||
            throw(DimensionMismatch("correlators must cover the whole lattice"))
        source_time = position[4]
        for τ in 1:nt
            t = mod(source_time + τ - 2, nt) + 1
            wrap_sign = antiperiodic && t < source_time ? -1 : 1
            four_point[:, :, :, τ, k, 1] .= view(absolute_four_point, :, :, :, t)
            one_hadron[τ, k, 1] = wrap_sign * absolute_one_hadron[t]
        end
    end
    source_ids = Tuple("$(source_kind)-$(join(position, '-'))" for position in source_positions)
    format_state(χ) = join(("$(s1)$(s2):$(χ[s1, s2])" for s1 in 1:2, s2 in 1:2 if !iszero(χ[s1, s2])), ",")
    base_metadata = (
        channel=String(channel),
        spin="source " * format_state(source_state) * "; sink " * format_state(sink_state),
        source_operator="n̄ p̄ at one $(source_kind) source, nucleon N = ε(qᵀCγ₅q)q",
        sink_operator="local p(x + r) n(x) with center-of-mass sum",
        temporal_boundary_condition=temporal_boundary_condition,
        lattice_shape=(nx, ny, nz, nt),
        lattice_spacing=Tuple(Float64.(lattice_spacing)),
        magnetic_flux=0,
        sample_ids=(String(sample_id),),
        source_ids=source_ids,
        code_commit=String(code_commit),
        ncolor=ncolor,
        nspin=nspin,
        nonrelativistic_operator=nonrelativistic,
        source_positions=Tuple(join(position, ",") for position in source_positions),
        time_origin="source time; index τ stores t₀ + τ - 1 modulo NT",
        relative_coordinate_convention="array index n stores displacement n-1 modulo spatial extent",
        spin_basis="(u↑, u↓) of nucleon_spin_basis; index 1 is up",
        four_point_contraction="block algorithm with all Wick permutations",
        one_hadron_normalization="Σ_x ⟨p p̄⟩ averaged over the two spins, wrap sign removed",
    )
    return NBSCorrelatorDataset(
        four_point,
        one_hadron,
        copy(one_hadron);
        metadata=merge(base_metadata, metadata),
    )
end
