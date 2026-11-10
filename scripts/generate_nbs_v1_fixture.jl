using HALQCD

length(ARGS) == 1 || error("usage: julia generate_nbs_v1_fixture.jl OUTPUT")

four_point = reshape(
    ComplexF32.(1:8) .+ ComplexF32(0, 0.25),
    2,
    1,
    1,
    2,
    1,
    2,
)
one_hadron_1 = reshape(Float32[2, 4, 8, 16], 2, 1, 2)
one_hadron_2 = reshape(Float64[3, 5, 7, 11], 2, 1, 2)
metadata = (
    channel="XiXi",
    spin="singlet",
    source_operator="local-Xi",
    sink_operator="local-Xi",
    temporal_boundary_condition="antiperiodic-quark",
    lattice_shape=(2, 1, 1, 2),
    lattice_spacing=(0.1, 0.1, 0.1, 0.2),
    magnetic_flux=0,
    sample_ids=("reference-configuration-1", "reference-configuration-2"),
    source_ids=("reference-source-1",),
    code_commit="format-v1-reference",
    generator="scripts/generate_nbs_v1_fixture.jl",
)
dataset = NBSCorrelatorDataset(
    four_point,
    one_hadron_1,
    one_hadron_2;
    metadata=metadata,
)

destination = write_nbs_dataset(ARGS[1], dataset)
loaded = read_nbs_dataset(destination)
loaded.four_point == dataset.four_point ||
    error("four-point round trip failed")
loaded.one_hadron_1 == dataset.one_hadron_1 ||
    error("first one-hadron round trip failed")
loaded.one_hadron_2 == dataset.one_hadron_2 ||
    error("second one-hadron round trip failed")
r_correlator(loaded) == r_correlator(dataset) ||
    error("normalized R round trip failed")

println("fixture_path=", destination)
println(
    "manifest_sha256=",
    strip(read(joinpath(destination, "manifest.sha256"), String)),
)
