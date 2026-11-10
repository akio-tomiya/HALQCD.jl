"""
    load_hal_pot_rcorr(path; lattice_size, bins)

Load a binary `Rcorr_*.dat` file written by
`tkmiriya/hal_pot_code/notes/hal_pot_single_ch.py`.

The reference Python code stores native-endian `ComplexF64` values in NumPy C
order with logical shape `(bin, x, y, z)`. This importer returns a Julia array
with shape `(x, y, z, bin)`, matching HALQCD.jl's spatial-axis convention.

This function imports already constructed R-correlators. Import of compressed
NBS wavefunctions remains a separate milestone because it depends on the
reference decompressor and spin/parity projection conventions.
"""
function load_hal_pot_rcorr(
    path::AbstractString;
    lattice_size::Integer,
    bins::Integer,
)
    lattice_size > 0 || throw(ArgumentError("lattice_size must be positive"))
    bins > 0 || throw(ArgumentError("bins must be positive"))

    spatial_volume = Base.checked_mul(
        Base.checked_mul(lattice_size, lattice_size),
        lattice_size,
    )
    ncomplex = Base.checked_mul(spatial_volume, bins)
    expected_bytes = Base.checked_mul(2 * sizeof(Float64), ncomplex)
    actual_bytes = filesize(path)
    actual_bytes == expected_bytes ||
        throw(DimensionMismatch(
            "expected $expected_bytes bytes for lattice_size=$lattice_size and bins=$bins, " *
            "found $actual_bytes",
        ))

    raw = open(path, "r") do io
        buffer = Vector{Float64}(undef, 2 * ncomplex)
        read!(io, buffer)
    end
    values = reinterpret(ComplexF64, raw)

    c_order = reshape(
        values,
        lattice_size,
        lattice_size,
        lattice_size,
        bins,
    )
    return permutedims(c_order, (3, 2, 1, 4))
end

"""
    load_hal_pot_table(path)

Load the nine-column text potential table written by
`tkmiriya/hal_pot_code`. The returned fields are `radius`, and the mean and
jackknife standard error for `spatial`, `first_time`, `second_time`, and
`total`.

Rows are cubic-orbit representatives, not statistically independent samples
and not spherical-shell averages.
"""
function load_hal_pot_table(path::AbstractString)
    columns = [Float64[] for _ in 1:9]
    open(path, "r") do io
        for (line_number, line) in enumerate(eachline(io))
            stripped = strip(line)
            (isempty(stripped) || startswith(stripped, '#')) && continue
            values = split(stripped)
            length(values) == 9 ||
                throw(ArgumentError(
                    "expected 9 columns at line $line_number, found $(length(values))",
                ))
            for column in 1:9
                value = tryparse(Float64, values[column])
                value === nothing &&
                    throw(ArgumentError(
                        "invalid floating-point value at line $line_number, column $column",
                    ))
                push!(columns[column], value)
            end
        end
    end
    isempty(columns[1]) && throw(ArgumentError("potential table contains no data rows"))

    return (
        radius=columns[1],
        spatial=(mean=columns[2], stderr=columns[3]),
        first_time=(mean=columns[4], stderr=columns[5]),
        second_time=(mean=columns[6], stderr=columns[7]),
        total=(mean=columns[8], stderr=columns[9]),
    )
end

function _read_fortran_record(io)
    eof(io) && throw(EOFError())
    byte_count = Int(ltoh(read(io, Int32)))
    byte_count >= 0 || throw(ArgumentError("negative Fortran record length"))
    payload = read(io, byte_count)
    length(payload) == byte_count || throw(EOFError())
    trailer = Int(ltoh(read(io, Int32)))
    trailer == byte_count ||
        throw(ArgumentError(
            "Fortran record marker mismatch: $byte_count != $trailer",
        ))
    return payload
end

function _little_endian_int32(payload)
    length(payload) % sizeof(Int32) == 0 ||
        throw(ArgumentError("Int32 record has a non-integral byte count"))
    return [ltoh(value) for value in reinterpret(Int32, payload)]
end

function _little_endian_float64(payload)
    length(payload) % sizeof(Float64) == 0 ||
        throw(ArgumentError("Float64 record has a non-integral byte count"))
    return [
        reinterpret(Float64, ltoh(bits))
        for bits in reinterpret(UInt64, payload)
    ]
end

@inline function _ph1_complex_component(values, representative, a, b, c, d)
    component = d + 2 * (c + 2 * (b + 2 * a))
    offset = 32 * (representative - 1) + 2 * component
    return complex(values[offset + 1], values[offset + 2])
end

"""
    load_ph1_compressed_a1plus_singlet(path)

Reconstruct the even-parity, cubic `A1+`, double spin-singlet scalar field
directly from the four-record compressed NBS format used by the public
`tkmiriya/hal_pot_code` XiXi sample.

The compressed payload is representative-major and stores 256 bytes (32
`Float64` values) for every cubic-orbit representative. This importer performs
the sink and source singlet contractions before expanding signed spatial-axis
permutations. It is therefore not a general replacement for `PH1.compress48`
and cannot reconstruct triplet or tensor components.

The returned array has shape `(x, y, z)` and element type `ComplexF64`.
"""
function load_ph1_compressed_a1plus_singlet(path::AbstractString)
    records = open(path, "r") do io
        result = [_read_fortran_record(io) for _ in 1:4]
        eof(io) || throw(ArgumentError("unexpected bytes after fourth record"))
        result
    end

    lattice_header = _little_endian_int32(records[1])
    length(lattice_header) == 4 ||
        throw(ArgumentError("compressed lattice header must contain four Int32 values"))
    nx, ny, nz, nrepresentatives = Int.(lattice_header)
    nx == ny == nz ||
        throw(DimensionMismatch("compressed NBS lattice must be cubic"))
    nx > 0 || throw(ArgumentError("lattice size must be positive"))
    nrepresentatives > 0 ||
        throw(ArgumentError("representative count must be positive"))

    representatives = _little_endian_int32(records[2])
    length(representatives) == nrepresentatives ||
        throw(ArgumentError("representative count does not match lattice header"))

    compression_header = Int.(_little_endian_int32(records[3]))
    compression_header == [0, 0, 0, nrepresentatives, 256] ||
        throw(ArgumentError("unsupported PH1 compression header $compression_header"))

    values = _little_endian_float64(records[4])
    length(values) == 32 * nrepresentatives ||
        throw(ArgumentError(
            "expected $(32 * nrepresentatives) payload values, found $(length(values))",
        ))

    singlet = Vector{ComplexF64}(undef, nrepresentatives)
    for representative in 1:nrepresentatives
        singlet[representative] = 0.5 * (
            _ph1_complex_component(values, representative, 1, 0, 1, 0) -
            _ph1_complex_component(values, representative, 1, 0, 0, 1) -
            _ph1_complex_component(values, representative, 0, 1, 1, 0) +
            _ph1_complex_component(values, representative, 0, 1, 0, 1)
        )
    end

    field = Array{ComplexF64}(undef, nx, nx, nx)
    assigned = falses(nx, nx, nx)
    for (linear_index, value) in zip(representatives, singlet)
        0 <= linear_index < nx^3 ||
            throw(ArgumentError("representative index $linear_index is out of bounds"))
        coordinates = (
            mod(linear_index, nx),
            mod(linear_index ÷ nx, nx),
            linear_index ÷ nx^2,
        )
        for permutation in _AXIS_PERMUTATIONS
            c1 = coordinates[permutation[1]]
            c2 = coordinates[permutation[2]]
            c3 = coordinates[permutation[3]]
            for s3 in (-1, 1), s2 in (-1, 1), s1 in (-1, 1)
                index = (
                    mod(s1 * c1, nx) + 1,
                    mod(s2 * c2, nx) + 1,
                    mod(s3 * c3, nx) + 1,
                )
                if assigned[index...]
                    isapprox(field[index...], value; rtol=2e-13, atol=0) ||
                        throw(ArgumentError(
                            "inconsistent values for cubic orbit at $index",
                        ))
                else
                    field[index...] = value
                    assigned[index...] = true
                end
            end
        end
    end
    all(assigned) ||
        throw(ArgumentError("compressed representatives do not cover the lattice"))
    return field
end
