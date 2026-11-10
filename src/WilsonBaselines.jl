using LinearAlgebra: I, inv

"""
    free_wilson_quark_pole_mass(kappa)

Return the zero-spatial-momentum pole mass, in lattice units, of a free
isotropic Wilson quark with Wilson parameter `r=1`:

`m_pole = log(1 + m0)`, where `m0 = 1/(2*kappa) - 4`.

The accepted range `0 < kappa < 1/8` corresponds to a positive bare mass.
This is a free-field baseline, not an interacting hadron-mass estimator.
"""
function free_wilson_quark_pole_mass(kappa::Real)
    isfinite(kappa) && 0 < kappa < 1 / 8 ||
        throw(
            ArgumentError(
                "the free Wilson pole-mass baseline requires 0 < kappa < 1/8",
            ),
        )
    bare_mass = inv(2 * kappa) - 4
    return log1p(bare_mass)
end

const _PAULI_MATRICES = (
    ComplexF64[0 1; 1 0],
    ComplexF64[0 -im; im 0],
    ComplexF64[1 0; 0 -1],
)

# Hermitian Euclidean γ matrices in the chiral basis.
const _EUCLIDEAN_GAMMAS = (
    [zeros(ComplexF64, 2, 2) -im*_PAULI_MATRICES[1]; im*_PAULI_MATRICES[1] zeros(ComplexF64, 2, 2)],
    [zeros(ComplexF64, 2, 2) -im*_PAULI_MATRICES[2]; im*_PAULI_MATRICES[2] zeros(ComplexF64, 2, 2)],
    [zeros(ComplexF64, 2, 2) -im*_PAULI_MATRICES[3]; im*_PAULI_MATRICES[3] zeros(ComplexF64, 2, 2)],
    ComplexF64[0 0 1 0; 0 0 0 1; 1 0 0 0; 0 1 0 0],
)

"""
    free_wilson_point_propagator(lattice_shape; kappa, r=1, ncolor=1,
                                 boundary_condition=(1, 1, 1, -1),
                                 source_position=(1, 1, 1, 1),
                                 gammas=<chiral basis>)

Free Wilson-quark propagator from a point source, computed by inverting the
Dirac operator momentum by momentum. The operator has the hopping
normalization `D = 1 - κ Σ_μ [(r - γ_μ) δ_{x+μ̂,y} + (r + γ_μ) δ_{x-μ̂,y}]`,

```text
D(p) = 1 - 2κr Σ_μ cos p_μ + 2iκ Σ_μ γ_μ sin p_μ,
```

with `p_μ = (2πn + θ_μ)/L_μ` and `θ_μ = 0` or `π` for the periodic (`1`) or
antiperiodic (`-1`) entries of `boundary_condition`. The color part is the
identity. The array has axes
`(sink_color, sink_spin, source_color, source_spin, x, y, z, t)`.

The default γ matrices are Hermitian and in the chiral basis. Pass the four
matrices of another code as `gammas` to compare propagators entry by entry;
for LatticeDiracOperators.jl they are
`LatticeDiracOperators.Dirac_operators.mk_gamma(1)[1][:, :, μ]`. Quantities
built from `S†S`, such as [`pion_correlator_point_source`](@ref) and
[`pipi_i2_four_point`](@ref), do not depend on the basis and serve as
free-field checks of solver output in any basis. The cost grows as the square of the
lattice volume, so use small lattices.
"""
function free_wilson_point_propagator(
    lattice_shape::NTuple{4,Integer};
    kappa::Real,
    r::Real=1,
    ncolor::Integer=1,
    boundary_condition::NTuple{4,Integer}=(1, 1, 1, -1),
    source_position::NTuple{4,Integer}=(1, 1, 1, 1),
    gammas=_EUCLIDEAN_GAMMAS,
)
    length(gammas) == 4 && all(gamma -> size(gamma) == (4, 4), gammas) ||
        throw(DimensionMismatch("gammas must be four 4×4 matrices"))
    all(>(0), lattice_shape) ||
        throw(ArgumentError("lattice extents must be positive"))
    all(in((1, -1)), boundary_condition) ||
        throw(ArgumentError("boundary_condition entries must be 1 or -1"))
    all(1 .<= source_position .<= lattice_shape) ||
        throw(BoundsError(lattice_shape, source_position))
    ncolor >= 1 || throw(ArgumentError("ncolor must be positive"))

    momenta = Tuple(
        [(2π * k + (boundary_condition[μ] == -1 ? π : 0.0)) / lattice_shape[μ]
         for k in 0:lattice_shape[μ]-1]
        for μ in 1:4
    )
    identity4 = Matrix{ComplexF64}(I, 4, 4)
    inverses = Dict{NTuple{4,Int},Matrix{ComplexF64}}()
    for index in Iterators.product((1:L for L in lattice_shape)...)
        p = ntuple(μ -> momenta[μ][index[μ]], 4)
        operator = (1 - 2 * kappa * r * sum(cos, p)) * identity4
        for μ in 1:4
            operator += 2im * kappa * sin(p[μ]) * gammas[μ]
        end
        inverses[index] = inv(operator)
    end

    volume = prod(lattice_shape)
    spin_propagator = zeros(ComplexF64, 4, 4, lattice_shape...)
    for site in Iterators.product((1:L for L in lattice_shape)...)
        displacement = site .- Tuple(source_position)
        value = zeros(ComplexF64, 4, 4)
        for (index, inverse) in inverses
            phase = cis(sum(momenta[μ][index[μ]] * displacement[μ] for μ in 1:4))
            value .+= phase .* inverse
        end
        spin_propagator[:, :, site...] .= value ./ volume
    end

    propagator = zeros(ComplexF64, ncolor, 4, ncolor, 4, lattice_shape...)
    for color in 1:ncolor
        propagator[color, :, color, :, ntuple(_ -> Colon(), 4)...] .= spin_propagator
    end
    return propagator
end
