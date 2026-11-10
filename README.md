# HALQCD.jl

HALQCD.jl is a Julia package for the time-dependent HAL QCD method. It builds
Nambu–Bethe–Salpeter (NBS) correlation functions from quark propagators,
extracts the leading-order potential from them, and compares the result with
Lüscher's finite-volume method. The gauge group is a type parameter (`SU{N}`),
so the contractions and the smearing work for any number of colors.

The analysis layer depends only on the Julia standard library. Propagators are
computed with the JuliaQCD packages (Gaugefields.jl and
LatticeDiracOperators.jl), which HALQCD.jl loads as optional package
extensions.

The package is at an early stage and its interface may still change.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/akio-tomiya/HALQCD.jl")
```

HALQCD.jl itself supports Julia 1.10 and later. The JuliaQCD v1 packages need
Julia 1.11 or later.

## What it does

Analysis of correlation functions:

- the leading-order potential of the time-dependent HAL QCD method for two
  hadrons of equal mass (`lo_potential_equal_mass`), with first and second
  Euclidean time derivatives and the periodic Laplacian;
- R-correlators normalized by single-hadron correlators, with masks for small
  or non-finite values;
- the cubic A1+ projection and shell averages;
- delete-one jackknife, applied per sample through the whole potential
  extraction (`lo_potential_from_jackknife_equal_mass`);
- the Lüscher zeta function Z₀₀(1; q²), p cot δ and the scattering length from
  an energy shift (`luscher_zeta`, `luscher_pcotdelta`,
  `luscher_scattering_length`), and the S-wave scattering length of a radial
  potential (`s_wave_scattering_length`);
- an NBS dataset type that keeps sources, times and samples apart, and an
  on-disk format with version, axes, metadata and SHA-256 checks
  (`NBSCorrelatorDataset`, `write_nbs_dataset`, `read_nbs_dataset`).

Contractions and gauge groups:

- gauge groups as types (`SU{N}`) with the dimension of the representation and
  the invariant tensors defined by multiple dispatch; another group is added by
  defining the methods of `AbstractGaugeGroup` in another package;
- sparse color-invariant tensors (`ColorInvariant`: δ_ab, ε_{a1…aN}, or user
  supplied);
- local color-singlet and two-baryon contractions including the Wick exchange
  terms of identical flavors;
- the I=2 four-point function of two pseudoscalars (π⁺π⁺ for SU(3)), with a
  fast path that sums the center of mass in momentum space and a reference path
  that sums the Wick determinants directly;
- proton–neutron NBS correlators and the central and tensor potentials.

Lattice tools, for any SU(N):

- projection onto the Lie algebra, stout smearing, Wuppertal smearing of quark
  sources, and the plaquette;
- a reference Wilson-clover point propagator
  (`wilson_clover_point_propagator`). Its clover term uses the sign and the
  traceful projection F_μν = (Q_μν − Q_μν†)/8 of Bridge++ and
  LatticeMatrices.jl; pion correlators agree with Bridge++ 2.1.3 to 10⁻¹² on
  SU(2), SU(3) and SU(4) configurations (`scripts/bridge_clover_compare.jl`);
- free Wilson propagators in momentum space for tests.

JuliaQCD extensions:

- with Gaugefields.jl and LatticeDiracOperators.jl loaded: `gauge_group`
  (the group of a gauge field), `juliaqcd_point_propagator` (point or wall
  source, Wilson or Wilson-clover with `fermiontype="WilsonClover"` and `cSW`),
  `juliaqcd_pipi_i2_nbs` and `juliaqcd_two_nucleon_nbs` (NBS datasets of one
  configuration), and `juliaqcd_links` (copy the links to a plain array);
- with QCDMeasurements.jl also loaded: `HAL_NBS_measurement(U; ...)` returns an
  `AbstractMeasurement`, and `measure(m, U)` returns an NBS dataset.

The extensions have been run with SU(2), SU(3) and SU(4) gauge fields of
Gaugefields 0.7 (`scripts/juliaqcd_stage4_groups.jl`) and with SU(3) fields of
the v1 packages, which use LatticeMatrices.jl (`test/juliaqcd_v1`).

Reading public data:

- a reader for the binary R-correlators of
  [`tkmiriya/hal_pot_code`](https://github.com/tkmiriya/hal_pot_code), and a
  reader that rebuilds the double-singlet scalar directly from the compressed
  NBS files of its public ΞΞ sample. `scripts/xixi_raw_julia_regression.jl`
  reproduces the published potential table from the compressed files
  (`docs/reference_data.md`).

## Quick start

```julia
using HALQCD

rminus = ones(ComplexF64, 8, 8, 8)
rzero  = ones(ComplexF64, 8, 8, 8)
rplus  = ones(ComplexF64, 8, 8, 8)

result = lo_potential_equal_mass(
    rminus,
    rzero,
    rplus;
    mass=1.5,
    spacing=(0.1, 0.1, 0.1),
    at=0.1,
)

@assert all(result.valid)
@assert maximum(abs, result.potential) < 1e-13
```

All inputs are given in one system of units. The core functions do not mix
lattice and physical units. More examples are in `examples/`.

## Comparison with Kurth et al.

We recomputed the I=2 ππ scattering of T. Kurth, N. Ishii, T. Doi, S. Aoki and
T. Hatsuda, JHEP 12 (2013) 015 (arXiv:1305.4462), with the action and
parameters of the paper. The configurations are quenched SU(3) at β=5.8726 on a
16³×128 lattice. The quarks are tree-level clover Wilson quarks (c_SW=1) on
2HEX-smeared links, with a Coulomb-gauge wall source and a Dirichlet wall in
the time direction half the lattice away from the source. We used κ=0.1220, 99
configurations and four sources per configuration. Gauge generation, gauge
fixing, smearing and the solver are those of the JuliaQCD v1 packages; the
driver is `scripts/kurth_pipi_production.jl` and the analysis is
`scripts/analyze_stage6_pipi.jl`. Errors are from the jackknife.

| quantity | HALQCD.jl | paper |
|---|---|---|
| m_π | 950(2) MeV | about 940 MeV |
| ΔE | 17.3(1.2) MeV | about 15 MeV (read from Fig. 6) |
| scattering length | 0.161(9) fm (Lüscher, one volume) | −0.1615(20) fm (four volumes) |
| δ | about −6.0° | about −5.6° (read from Fig. 6) |

Our scattering length is positive for repulsion; the paper writes the same
quantity with a negative sign. Physical units use a=0.115 fm.

The energy shift obtained from the leading-order HAL QCD potential agrees with
the energy shift of the correlation function within 1σ at every time from
t=10 to 18. On the diagonal of the cube the potential is 3 to 8 MeV at 0.8 fm
and −1 to 2 MeV at 1.0 fm, in agreement with Fig. 3 of the paper.

The comparison covers a single volume, 16³. The paper obtains the scattering
length from four volumes with the large-volume expansion; we use one volume,
Lüscher's formula and the leading order of the effective range expansion. The
close agreement of the central values of the scattering length is therefore
partly accidental. We have not yet determined the scattering length with the
method of the paper.

## Conventions

- scalar field: `(x, y, z)`
- R-correlator bins: `(x, y, z, bin)`
- color links: `(color_out, color_in, mu, x, y, z, t)`; the color axes have
  the dimension of the representation (N for SU(N))
- `mu=1,2,3,4` is `x,y,z,t`

The leading-order potential for equal mass `m` is

```text
V_LO = R⁻¹ [(1/(4m)) ∂ₜ² - ∂ₜ + ∇²/m] R
```

With the reduced mass `m_red=m/2` of `tkmiriya/hal_pot_code`, the spatial term
is `∇²/(2m_red)` and the second time derivative is `∂ₜ²/(8m_red)`.

## Tests

```julia
using Pkg
Pkg.test("HALQCD")
```

The tests of the JuliaQCD v1 extension have their own environment, because
they need Julia 1.11 and the JuliaQCD packages:

```sh
julia --project=test/juliaqcd_v1 -e 'using Pkg; Pkg.instantiate(); include("test/juliaqcd_v1/runtests.jl")'
```

## License

MIT License.
