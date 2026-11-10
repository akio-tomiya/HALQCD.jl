@testset "hal_pot_code R-correlator importer" begin
    lattice_size = 3
    bins = 2
    expected = Array{ComplexF64}(undef, lattice_size, lattice_size, lattice_size, bins)
    for bin in 1:bins, x in 1:lattice_size, y in 1:lattice_size, z in 1:lattice_size
        value = 1000 * bin + 100 * x + 10 * y + z
        expected[x, y, z, bin] = complex(value, -value / 10)
    end

    c_order = ComplexF64[]
    for bin in 1:bins, x in 1:lattice_size, y in 1:lattice_size, z in 1:lattice_size
        push!(c_order, expected[x, y, z, bin])
    end

    mktemp() do path, io
        write(io, reinterpret(Float64, c_order))
        close(io)
        loaded = load_hal_pot_rcorr(
            path;
            lattice_size=lattice_size,
            bins=bins,
        )
        @test size(loaded) == size(expected)
        @test loaded == expected
        @test_throws DimensionMismatch load_hal_pot_rcorr(
            path;
            lattice_size=lattice_size,
            bins=bins + 1,
        )
    end
end

@testset "hal_pot_code potential table importer" begin
    mktemp() do path, io
        write(io, """
        # M_red = 0.3325
        # r H0R/R dR/R d2R/R total
        0.0 1.0 0.1 -0.2 0.02 0.03 0.003 0.83 0.12
        1.0 0.5 0.05 -0.1 0.01 0.02 0.002 0.42 0.06
        """)
        close(io)
        table = load_hal_pot_table(path)
        @test table.radius == [0.0, 1.0]
        @test table.spatial.mean == [1.0, 0.5]
        @test table.first_time.stderr == [0.02, 0.01]
        @test table.total.mean == [0.83, 0.42]
    end

    mktemp() do path, io
        write(io, "0 1 2\n")
        close(io)
        @test_throws ArgumentError load_hal_pot_table(path)
    end
end

@testset "PH1 compressed A1+ singlet importer" begin
    function write_record(io, values)
        record_bytes = reinterpret(UInt8, values)
        byte_count = Int32(length(record_bytes))
        write(io, htol(byte_count))
        write(io, record_bytes)
        write(io, htol(byte_count))
    end

    lattice_size = 3
    representatives = Int32[0, 9, 10, 13]
    expected_values = ComplexF64[1 + 2im, 3 - 4im, -2 + 0.5im, 7 - im]
    field_payload = zeros(Float64, 32 * length(representatives))
    for (representative, value) in enumerate(expected_values)
        component = 0 + 2 * (1 + 2 * (0 + 2 * 1))
        offset = 32 * (representative - 1) + 2 * component
        field_payload[offset + 1] = 2 * real(value)
        field_payload[offset + 2] = 2 * imag(value)
    end

    mktemp() do path, io
        write_record(
            io,
            htol.(Int32[lattice_size, lattice_size, lattice_size, length(representatives)]),
        )
        write_record(io, htol.(representatives))
        write_record(
            io,
            htol.(Int32[0, 0, 0, length(representatives), 256]),
        )
        little_endian_bits = htol.(reinterpret(UInt64, field_payload))
        write_record(io, collect(reinterpret(Float64, little_endian_bits)))
        close(io)

        field = load_ph1_compressed_a1plus_singlet(path)
        @test size(field) == (lattice_size, lattice_size, lattice_size)
        @test field[1, 1, 1] == expected_values[1]
        @test field[1, 1, 2] == expected_values[2]
        @test field[2, 1, 2] == expected_values[3]
        @test field[2, 2, 2] == expected_values[4]
        @test a1plus_project(field) ≈ field
    end
end
