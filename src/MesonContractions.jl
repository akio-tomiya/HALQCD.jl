using LinearAlgebra: I, tr

const PIPI_I2_CHANNEL = "pipi_I2"

# Chiral-basis γ₅. The I=2 ππ correlator does not depend on this choice,
# because γ₅ enters both the operators and the γ₅-hermiticity relation.
const _CHIRAL_GAMMA5 = ComplexF64[
    1 0 0 0
    0 1 0 0
    0 0 -1 0
    0 0 0 -1
]

function _site_matrix(propagator, ncolor, nspin, x, y, z, t)
    n = ncolor * nspin
    return reshape(view(propagator, :, :, :, :, x, y, z, t), n, n)
end

"""
    pipi_i2_four_point(propagator)

Zero-total-momentum `I=2` two-pseudoscalar four-point function
(`π⁺π⁺` in `SU(3)`) from one point-source propagator,

```text
F(r, t) = Σ_x [ tr A(x + r, t) tr A(x, t) - tr(A(x + r, t) A(x, t)) ],
A(x, t) = S(x, t)† S(x, t),
```

where `S(x, t)` is the `(Nc Nspin) × (Nc Nspin)` matrix
`propagator[:, :, :, :, x, y, z, t]` with sink rows and source columns. Both
source pions sit at the source point with operator `q̄ γ₅ δ q`. The `u` and
`d` quarks are degenerate, and the backward propagator is obtained from
`γ₅`-hermiticity, so the formula holds for Wilson and clover fermions. The
first term is the direct contraction and the second the quark exchange. The
overall Wick multiplicity 2 is dropped, so that the direct term is the
product of two one-pion blocks whose spatial sum is
[`pion_correlator_point_source`](@ref).

The color sum is the trace over the color axes of `propagator`, so the
function serves every gauge group whose meson operator uses `δ_ab`.

The sum over `x` is evaluated in momentum space with separable discrete
Fourier transforms. The array index `n` of the result stores the
displacement `n - 1` modulo the spatial extent. Time is the absolute sink
time of the propagator. [`pipi_i2_four_point_explicit`](@ref) evaluates the
same quantity by direct Wick sums and serves as the reference.
"""
function pipi_i2_four_point(propagator::AbstractArray{<:Number,8})
    ncolor, nspin = _check_point_propagator(propagator)
    n = ncolor * nspin
    nx, ny, nz, nt = size(propagator)[5:8]
    volume = nx * ny * nz
    forward = (_dft_matrix(nx), _dft_matrix(ny), _dft_matrix(nz))
    backward = conj.(forward)
    four_point = zeros(Float64, nx, ny, nz, nt)
    blocks = Array{ComplexF64}(undef, n * n, nx, ny, nz)
    traces = Array{ComplexF64}(undef, 1, nx, ny, nz)
    for t in 1:nt
        for z in 1:nz, y in 1:ny, x in 1:nx
            S = _site_matrix(propagator, ncolor, nspin, x, y, z, t)
            A = S' * S
            blocks[:, x, y, z] .= vec(A)
            traces[1, x, y, z] = tr(A)
        end
        block_momenta = _spatial_dft(blocks, forward)
        trace_momenta = _spatial_dft(traces, forward)
        # Σ_x f(x + r) g(x) has Fourier transform f̂(p) ĝ(-p). For the direct
        # term f = g = tr A is real; for the exchange term g = A_ji = conj(A_ij).
        spectrum = dropdims(abs2.(trace_momenta); dims=4) .-
            dropdims(sum(abs2, block_momenta; dims=4); dims=4)
        correlation = _spatial_dft(reshape(spectrum, 1, nx, ny, nz), backward)
        four_point[:, :, :, t] .= real.(dropdims(correlation; dims=4)) ./ volume
    end
    return four_point
end

function _dft_matrix(n::Integer)
    return ComplexF64[cispi(-2 * (j - 1) * (k - 1) / n) for j in 1:n, k in 1:n]
end

# Transform the three spatial axes of an array with axes (batch, x, y, z).
# Each pass contracts the last axis and then moves it to the front, so after
# the three passes the axes are (x, y, z, batch).
function _spatial_dft(values::AbstractArray{<:Number,4}, matrices)
    result = values
    for matrix in reverse(matrices)
        shape = size(result)
        transformed = reshape(reshape(result, :, shape[end]) * matrix, shape)
        result = permutedims(transformed, (4, 1, 2, 3))
    end
    return result
end

function _meson_operator_entries(color_invariant, gamma, ncolor)
    nspin = size(gamma, 1)
    entries = Tuple{Int,Int,ComplexF64}[]
    for ((a, b), color_value) in color_invariant.entries,
            β in 1:nspin, α in 1:nspin
        iszero(gamma[α, β]) && continue
        push!(
            entries,
            (a + (α - 1) * ncolor, b + (β - 1) * ncolor, color_value * gamma[α, β]),
        )
    end
    return entries
end

"""
    pipi_i2_four_point_explicit(propagator;
                                color_invariant=kronecker_invariant(Nc),
                                source_color_invariant=color_invariant,
                                gamma5=chiral γ₅)

Reference evaluation of [`pipi_i2_four_point`](@ref) by direct Wick sums.
The sink operator is `π(y) = I[a,b] γ₅[α,β] d̄[a,α](y) u[b,β](y)` and the
source operator `I'[a,b] γ₅[α,β] ū[a,α](0) d[b,β](0)`, with
`I = color_invariant` and `I' = source_color_invariant`. For a complex
invariant the caller passes the conjugate as `source_color_invariant`. For
each pair of sink points the sum over all operator components of the
product of the `2 × 2` Wick determinants of the `u` and `d` lines is taken,
with the backward `d` line `γ₅ S(y, 0)† γ₅`, and the two sink points are
summed over `x` directly. The result is divided by the Wick multiplicity 2.

The cost grows as `V² (Nspin Nc)⁴` per time slice, so use only small
lattices. For `I = δ_ab` the result does not depend on the representation of
`gamma5`, as long as it is Hermitian and squares to one.
"""
function pipi_i2_four_point_explicit(
    propagator::AbstractArray{<:Number,8};
    color_invariant::Union{Nothing,ColorInvariant{2}}=nothing,
    source_color_invariant::Union{Nothing,ColorInvariant{2}}=color_invariant,
    gamma5::AbstractMatrix{<:Number}=_CHIRAL_GAMMA5,
)
    ncolor, nspin = _check_point_propagator(propagator)
    sink_invariant =
        isnothing(color_invariant) ? kronecker_invariant(ncolor) : color_invariant
    source_invariant =
        isnothing(source_color_invariant) ? sink_invariant : source_color_invariant
    for invariant in (sink_invariant, source_invariant)
        invariant.ncolor == ncolor ||
            throw(DimensionMismatch("color invariant has $(invariant.ncolor) colors; propagator has $ncolor"))
    end
    size(gamma5) == (nspin, nspin) ||
        throw(DimensionMismatch("gamma5 must be $nspin × $nspin"))

    nx, ny, nz, nt = size(propagator)[5:8]
    sink_entries = _meson_operator_entries(sink_invariant, gamma5, ncolor)
    source_entries = _meson_operator_entries(source_invariant, gamma5, ncolor)
    spin_color_gamma5 = kron(Matrix{ComplexF64}(gamma5), Matrix{ComplexF64}(I, ncolor, ncolor))
    sites = vec(collect(Iterators.product(1:nx, 1:ny, 1:nz)))
    four_point = zeros(ComplexF64, nx, ny, nz, nt)
    for t in 1:nt
        forward = Dict(
            site => Matrix{ComplexF64}(_site_matrix(propagator, ncolor, nspin, site..., t))
            for site in sites
        )
        # backward[site][source index, sink index] = <d(0) d̄(site)>.
        backward = Dict(
            site => spin_color_gamma5 * forward[site]' * spin_color_gamma5
            for site in sites
        )
        for site_1 in sites, site_2 in sites
            S1, S2 = forward[site_1], forward[site_2]
            B1, B2 = backward[site_1], backward[site_2]
            value = zero(ComplexF64)
            for (d̄1, u1, w1) in sink_entries, (d̄2, u2, w2) in sink_entries,
                    (ūA, dA, wA) in source_entries, (ūB, dB, wB) in source_entries
                up = S1[u1, ūA] * S2[u2, ūB] - S1[u1, ūB] * S2[u2, ūA]
                down = B1[dA, d̄1] * B2[dB, d̄2] - B2[dA, d̄2] * B1[dB, d̄1]
                value += w1 * w2 * wA * wB * up * down
            end
            displacement = mod.(site_1 .- site_2, (nx, ny, nz)) .+ 1
            four_point[displacement..., t] += value / 2
        end
    end
    return four_point
end

function _halqcd_code_commit()
    root = pkgdir(@__MODULE__)
    isnothing(root) && return "unknown"
    try
        commit = readchomp(`git -C $root rev-parse HEAD`)
        dirty = !isempty(readchomp(`git -C $root status --porcelain --untracked-files=no`))
        return dirty ? commit * "-dirty" : commit
    catch
        return "unknown"
    end
end

"""
    pipi_i2_nbs_dataset(propagators, source_positions;
                        sample_id,
                        temporal_boundary_condition="antiperiodic-quark",
                        lattice_spacing=(1.0, 1.0, 1.0, 1.0),
                        code_commit=<HALQCD.jl commit>,
                        metadata=(;))

Build an [`NBSCorrelatorDataset`](@ref) for the `I=2` two-pseudoscalar
channel from point-source propagators of one gauge configuration.
`propagators[k]` is the propagator from `source_positions[k]`, with axes
`(sink_color, sink_spin, source_color, source_spin, x, y, z, t)`.

The time axis of the dataset is measured from the source time: index `τ`
stores the sink time `t₀ + τ - 1` modulo the temporal extent. The four-point
function is [`pipi_i2_four_point`](@ref) and both one-hadron correlators are
[`pion_correlator_point_source`](@ref). Both are built from `|S|²`-type
products, so the sign that antiperiodic boundary conditions give to
propagators wrapping around the time direction cancels. Entries of
`metadata` are added to the dataset metadata.
"""
function pipi_i2_nbs_dataset(
    propagators::AbstractVector{<:AbstractArray{<:Number,8}},
    source_positions::AbstractVector{<:NTuple{4,Integer}};
    sample_id::AbstractString,
    temporal_boundary_condition::AbstractString="antiperiodic-quark",
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
    for position in source_positions
        all(1 .<= position .<= (nx, ny, nz, nt)) ||
            throw(BoundsError((nx, ny, nz, nt), position))
    end

    nsource = length(propagators)
    four_point = zeros(Float64, nx, ny, nz, nt, nsource, 1)
    one_hadron = zeros(Float64, nt, nsource, 1)
    for (k, (propagator, position)) in enumerate(zip(propagators, source_positions))
        absolute_four_point = pipi_i2_four_point(propagator)
        absolute_one_hadron = pion_correlator_point_source(propagator)
        source_time = position[4]
        for τ in 1:nt
            t = mod(source_time + τ - 2, nt) + 1
            four_point[:, :, :, τ, k, 1] .= view(absolute_four_point, :, :, :, t)
            one_hadron[τ, k, 1] = absolute_one_hadron[t]
        end
    end

    source_ids = Tuple(
        "point-$(join(position, '-'))" for position in source_positions
    )
    base_metadata = (
        channel=PIPI_I2_CHANNEL,
        spin="0",
        source_operator="two pseudoscalars q̄γ₅δq at one point source",
        sink_operator="local pseudoscalar pair with center-of-mass sum",
        temporal_boundary_condition=temporal_boundary_condition,
        lattice_shape=(nx, ny, nz, nt),
        lattice_spacing=Tuple(Float64.(lattice_spacing)),
        magnetic_flux=0,
        sample_ids=(String(sample_id),),
        source_ids=source_ids,
        code_commit=String(code_commit),
        ncolor=ncolor,
        nspin=nspin,
        source_positions=Tuple(join(position, ",") for position in source_positions),
        time_origin="source time; index τ stores t₀ + τ - 1 modulo NT",
        relative_coordinate_convention="array index n stores displacement n-1 modulo spatial extent",
        four_point_contraction="direct minus exchange, Wick multiplicity 2 dropped",
        one_hadron_normalization="Σ_x |S|², same as QCDMeasurements Pion_correlator_measurement",
    )
    return NBSCorrelatorDataset(
        four_point,
        one_hadron,
        copy(one_hadron);
        metadata=merge(base_metadata, metadata),
    )
end
