using TOML

@testset "Committed NBS v1 reader fixture" begin
    fixture_path = joinpath(@__DIR__, "fixtures", "nbs_v1_reference")
    @test strip(read(joinpath(fixture_path, "manifest.sha256"), String)) ==
          "8070fe8488e8fe715354580c8f3604a41a6c3c938d8c7ef5f6713fbfe68a37ab"

    dataset = read_nbs_dataset(fixture_path)
    expected_four_point = reshape(
        ComplexF32.(1:8) .+ ComplexF32(0, 0.25),
        2,
        1,
        1,
        2,
        1,
        2,
    )
    @test dataset.four_point == expected_four_point
    @test dataset.one_hadron_1 ==
          reshape(Float32[2, 4, 8, 16], 2, 1, 2)
    @test dataset.one_hadron_2 ==
          reshape(Float64[3, 5, 7, 11], 2, 1, 2)
    @test dataset.metadata.lattice_shape == (2, 1, 1, 2)
    @test dataset.metadata.sample_ids ==
          ("reference-configuration-1", "reference-configuration-2")
    @test dataset.metadata.generator ==
          "scripts/generate_nbs_v1_fixture.jl"
    normalized = r_correlator(dataset)
    @test normalized[1, 1, 1, 1, 1, 1] ≈
          ComplexF64(1 / 6, 1 / 24)
end

@testset "Versioned NBS disk round trip" begin
    four_point = reshape(
        ComplexF32.(1:96) .+ ComplexF32(0, 0.5),
        2,
        2,
        2,
        3,
        2,
        2,
    )
    one_hadron_1 = reshape(Float32.(1:12), 3, 2, 2)
    one_hadron_2 = reshape(Float64.(13:24), 3, 2, 2)
    metadata = merge(
        _nbs_fixture_metadata(),
        (
            solver_residual=1e-11,
            package_versions=["Gaugefields=0.7.2", "LDO=0.6.3"],
        ),
    )
    dataset = NBSCorrelatorDataset(
        four_point,
        one_hadron_1,
        one_hadron_2;
        metadata=metadata,
    )

    mktempdir() do parent
        path = joinpath(parent, "fixture")
        @test write_nbs_dataset(path, dataset) == abspath(path)
        @test isfile(joinpath(path, "manifest.toml"))
        @test isfile(joinpath(path, "manifest.sha256"))
        loaded = read_nbs_dataset(path)
        @test loaded.four_point == dataset.four_point
        @test loaded.one_hadron_1 == dataset.one_hadron_1
        @test loaded.one_hadron_2 == dataset.one_hadron_2
        @test loaded.sample_kind == dataset.sample_kind
        @test loaded.schema_version == dataset.schema_version
        @test loaded.metadata.channel == dataset.metadata.channel
        @test loaded.metadata.sample_ids == dataset.metadata.sample_ids
        @test loaded.metadata.package_versions ==
              dataset.metadata.package_versions
        @test r_correlator(loaded) == r_correlator(dataset)
        @test_throws ArgumentError write_nbs_dataset(path, dataset)
    end

    jackknife = delete_one_jackknife_dataset(dataset)
    mktempdir() do parent
        path = joinpath(parent, "jackknife")
        write_nbs_dataset(path, jackknife)
        loaded = read_nbs_dataset(path)
        @test loaded.sample_kind == :derived_resample
        @test loaded.metadata.parent_configuration_ids ==
              jackknife.metadata.parent_configuration_ids
        @test loaded.four_point == jackknife.four_point
    end
end

@testset "NBS disk corruption and schema controls" begin
    four_point = reshape(ComplexF64.(1:16), 2, 2, 1, 1, 2, 2)
    one_hadron_1 = ones(Float64, 1, 2, 2)
    one_hadron_2 = fill(2.0, 1, 2, 2)
    metadata = _nbs_fixture_metadata(
        lattice_shape=(2, 2, 1, 1),
    )
    dataset = NBSCorrelatorDataset(
        four_point,
        one_hadron_1,
        one_hadron_2;
        metadata=metadata,
    )

    mktempdir() do parent
        path = joinpath(parent, "tampered-payload")
        write_nbs_dataset(path, dataset)
        payload = joinpath(path, "four_point.bin")
        open(payload, "r+") do io
            first_byte = read(io, UInt8)
            seekstart(io)
            write(io, xor(first_byte, 0x01))
        end
        @test_throws ArgumentError read_nbs_dataset(path)
    end

    mktempdir() do parent
        path = joinpath(parent, "tampered-metadata")
        write_nbs_dataset(path, dataset)
        manifest_path = joinpath(path, "manifest.toml")
        manifest = TOML.parsefile(manifest_path)
        manifest["metadata"]["channel"] = "not-the-original-channel"
        open(manifest_path, "w") do io
            TOML.print(io, manifest; sorted=true)
        end
        @test_throws ArgumentError read_nbs_dataset(path)
    end

    mktempdir() do parent
        path = joinpath(parent, "tampered-axis")
        write_nbs_dataset(path, dataset)
        manifest_path = joinpath(path, "manifest.toml")
        manifest = TOML.parsefile(manifest_path)
        manifest["axis_order"][1:2] = manifest["axis_order"][2:-1:1]
        open(manifest_path, "w") do io
            TOML.print(io, manifest; sorted=true)
        end
        open(joinpath(path, "manifest.sha256"), "w") do io
            println(io, HALQCD._file_sha256(manifest_path))
        end
        @test_throws ArgumentError read_nbs_dataset(path)
    end

    unsupported = NBSCorrelatorDataset(
        round.(Int, real.(four_point)),
        one_hadron_1,
        one_hadron_2;
        metadata=metadata,
    )
    mktempdir() do parent
        @test_throws ArgumentError write_nbs_dataset(
            joinpath(parent, "unsupported-type"),
            unsupported,
        )
    end

    unsupported_metadata = NBSCorrelatorDataset(
        four_point,
        one_hadron_1,
        one_hadron_2;
        metadata=merge(metadata, (nested=Dict("not" => "portable"),)),
    )
    mktempdir() do parent
        @test_throws ArgumentError write_nbs_dataset(
            joinpath(parent, "unsupported-metadata"),
            unsupported_metadata,
        )
    end
end
