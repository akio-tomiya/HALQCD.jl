using SHA
using TOML

const NBS_DISK_FORMAT_NAME = "HALQCD.NBSCorrelatorDataset"
const NBS_DISK_FORMAT_VERSION = v"1.0.0"
const _NBS_DISK_MANIFEST = "manifest.toml"
const _NBS_DISK_MANIFEST_HASH = "manifest.sha256"
const _NBS_DISK_ARRAY_NAMES = (
    :four_point,
    :one_hadron_1,
    :one_hadron_2,
)

_nbs_disk_type_name(::Type{Float32}) = "Float32"
_nbs_disk_type_name(::Type{Float64}) = "Float64"
_nbs_disk_type_name(::Type{ComplexF32}) = "ComplexF32"
_nbs_disk_type_name(::Type{ComplexF64}) = "ComplexF64"
function _nbs_disk_type_name(type)
    throw(ArgumentError(
        "unsupported NBS disk element type $type; " *
        "expected Float32, Float64, ComplexF32, or ComplexF64",
    ))
end

function _nbs_disk_type(name::AbstractString)
    name == "Float32" && return Float32
    name == "Float64" && return Float64
    name == "ComplexF32" && return ComplexF32
    name == "ComplexF64" && return ComplexF64
    throw(ArgumentError("unsupported NBS disk element type name $name"))
end

_nbs_disk_element_bytes(::Type{Float32}) = 4
_nbs_disk_element_bytes(::Type{Float64}) = 8
_nbs_disk_element_bytes(::Type{ComplexF32}) = 8
_nbs_disk_element_bytes(::Type{ComplexF64}) = 16

function _write_nbs_value(io::IO, value::Float32)
    write(io, htol(reinterpret(UInt32, value)))
    return nothing
end

function _write_nbs_value(io::IO, value::Float64)
    write(io, htol(reinterpret(UInt64, value)))
    return nothing
end

function _write_nbs_value(io::IO, value::ComplexF32)
    _write_nbs_value(io, real(value))
    _write_nbs_value(io, imag(value))
    return nothing
end

function _write_nbs_value(io::IO, value::ComplexF64)
    _write_nbs_value(io, real(value))
    _write_nbs_value(io, imag(value))
    return nothing
end

function _read_nbs_value(io::IO, ::Type{Float32})
    return reinterpret(Float32, ltoh(read(io, UInt32)))
end

function _read_nbs_value(io::IO, ::Type{Float64})
    return reinterpret(Float64, ltoh(read(io, UInt64)))
end

function _read_nbs_value(io::IO, ::Type{ComplexF32})
    return complex(
        _read_nbs_value(io, Float32),
        _read_nbs_value(io, Float32),
    )
end

function _read_nbs_value(io::IO, ::Type{ComplexF64})
    return complex(
        _read_nbs_value(io, Float64),
        _read_nbs_value(io, Float64),
    )
end

function _file_sha256(path::AbstractString)
    return open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

function _toml_metadata_value(value, location::AbstractString)
    if value isa AbstractString || value isa Bool || value isa Integer
        return value
    elseif value isa AbstractFloat
        isfinite(value) ||
            throw(ArgumentError("$location must not contain NaN or infinity"))
        return value
    elseif value isa Tuple || value isa AbstractVector
        return [
            _toml_metadata_value(item, "$location[$index]")
            for (index, item) in enumerate(value)
        ]
    end
    throw(ArgumentError(
        "$location has unsupported type $(typeof(value)); " *
        "use TOML scalars or one-dimensional sequences",
    ))
end

function _toml_metadata(metadata::NamedTuple)
    result = Dict{String,Any}()
    for key in keys(metadata)
        result[string(key)] = _toml_metadata_value(
            getproperty(metadata, key),
            "metadata.$key",
        )
    end
    return result
end

function _metadata_namedtuple(metadata::AbstractDict)
    names = sort!(collect(keys(metadata)))
    all(name -> name isa AbstractString, names) ||
        throw(ArgumentError("metadata keys must be strings"))
    symbols = Tuple(Symbol.(names))
    values = Tuple(metadata[name] for name in names)
    return NamedTuple{symbols}(values)
end

function _write_nbs_array(
    directory::AbstractString,
    name::Symbol,
    values::AbstractArray,
)
    type_name = _nbs_disk_type_name(eltype(values))
    filename = string(name, ".bin")
    path = joinpath(directory, filename)
    open(path, "w") do io
        for value in values
            _write_nbs_value(io, value)
        end
    end
    return Dict{String,Any}(
        "file" => filename,
        "eltype" => type_name,
        "shape" => collect(size(values)),
        "endianness" => "little",
        "storage_order" => "julia-column-major",
        "sha256" => _file_sha256(path),
    )
end

"""
    write_nbs_dataset(path, dataset)

Write a versioned NBS dataset directory containing a TOML manifest and three
little-endian binary payloads. The destination must not already exist.
Payload SHA-256 digests, element types, shapes, and axis order are recorded in
the manifest.
"""
function write_nbs_dataset(
    path::AbstractString,
    dataset::NBSCorrelatorDataset,
)
    destination = abspath(path)
    ispath(destination) &&
        throw(ArgumentError("refusing to overwrite existing path $destination"))
    parent = dirname(destination)
    mkpath(parent)
    temporary = mktempdir(parent; prefix=".halqcd-nbs-")
    try
        arrays = Dict{String,Any}()
        for name in _NBS_DISK_ARRAY_NAMES
            arrays[string(name)] = _write_nbs_array(
                temporary,
                name,
                getproperty(dataset, name),
            )
        end
        manifest = Dict{String,Any}(
            "format" => NBS_DISK_FORMAT_NAME,
            "format_version" => string(NBS_DISK_FORMAT_VERSION),
            "dataset_schema_version" => string(dataset.schema_version),
            "axis_order" => collect(string.(NBS_DATASET_AXIS_ORDER)),
            "one_hadron_axis_order" => ["time", "source", "sample"],
            "sample_kind" => string(dataset.sample_kind),
            "metadata" => _toml_metadata(dataset.metadata),
            "arrays" => arrays,
        )
        open(joinpath(temporary, "manifest.toml"), "w") do io
            TOML.print(io, manifest; sorted=true)
        end
        open(joinpath(temporary, _NBS_DISK_MANIFEST_HASH), "w") do io
            println(io, _file_sha256(joinpath(temporary, _NBS_DISK_MANIFEST)))
        end
        mv(temporary, destination)
    catch
        ispath(temporary) && rm(temporary; recursive=true, force=true)
        rethrow()
    end
    return destination
end

function _checked_payload_bytes(shape, type)
    count = 1
    for dimension in shape
        dimension isa Integer && dimension > 0 ||
            throw(ArgumentError(
                "array shape must contain positive integers",
            ))
        count = Base.Checked.checked_mul(count, Int(dimension))
    end
    return Base.Checked.checked_mul(
        count,
        _nbs_disk_element_bytes(type),
    )
end

function _read_nbs_array(
    directory::AbstractString,
    arrays::AbstractDict,
    name::Symbol,
)
    key = string(name)
    haskey(arrays, key) ||
        throw(ArgumentError("manifest is missing arrays.$key"))
    descriptor = arrays[key]
    descriptor isa AbstractDict ||
        throw(ArgumentError("manifest arrays.$key must be a table"))

    expected_file = string(name, ".bin")
    get(descriptor, "file", nothing) == expected_file ||
        throw(ArgumentError("manifest arrays.$key has an invalid filename"))
    get(descriptor, "endianness", nothing) == "little" ||
        throw(ArgumentError("manifest arrays.$key must be little-endian"))
    get(descriptor, "storage_order", nothing) == "julia-column-major" ||
        throw(ArgumentError(
            "manifest arrays.$key has an unsupported storage order",
        ))

    type_name = get(descriptor, "eltype", nothing)
    type_name isa AbstractString ||
        throw(ArgumentError("manifest arrays.$key is missing eltype"))
    type = _nbs_disk_type(type_name)

    shape = get(descriptor, "shape", nothing)
    shape isa AbstractVector ||
        throw(ArgumentError("manifest arrays.$key is missing shape"))
    expected_bytes = _checked_payload_bytes(shape, type)

    payload_path = joinpath(directory, expected_file)
    isfile(payload_path) ||
        throw(ArgumentError("missing NBS payload $expected_file"))
    filesize(payload_path) == expected_bytes ||
        throw(ArgumentError(
            "payload size mismatch for $expected_file",
        ))

    expected_hash = get(descriptor, "sha256", nothing)
    expected_hash isa AbstractString && length(expected_hash) == 64 ||
        throw(ArgumentError("manifest arrays.$key has an invalid SHA-256"))
    actual_hash = _file_sha256(payload_path)
    lowercase(expected_hash) == actual_hash ||
        throw(ArgumentError("SHA-256 mismatch for $expected_file"))

    values = Array{type}(undef, Tuple(Int.(shape)))
    open(payload_path, "r") do io
        for index in eachindex(values)
            values[index] = _read_nbs_value(io, type)
        end
        eof(io) ||
            throw(ArgumentError("trailing bytes in NBS payload $expected_file"))
    end
    return values
end

function _validate_nbs_manifest_hash(directory::AbstractString)
    manifest_path = joinpath(directory, _NBS_DISK_MANIFEST)
    hash_path = joinpath(directory, _NBS_DISK_MANIFEST_HASH)
    isfile(hash_path) ||
        throw(ArgumentError("NBS dataset is missing $_NBS_DISK_MANIFEST_HASH"))
    expected_hash = strip(read(hash_path, String))
    valid_hex = length(expected_hash) == 64 && all(expected_hash) do character
        lowercase_character = lowercase(character)
        isdigit(character) ||
            ('a' <= lowercase_character <= 'f')
    end
    valid_hex ||
        throw(ArgumentError("invalid NBS manifest SHA-256"))
    actual_hash = _file_sha256(manifest_path)
    lowercase(expected_hash) == actual_hash ||
        throw(ArgumentError("SHA-256 mismatch for $_NBS_DISK_MANIFEST"))
    return nothing
end

"""
    read_nbs_dataset(path)

Read and validate a versioned NBS dataset directory. Manifest identity, axis
order, payload sizes, SHA-256 digests, and the in-memory dataset invariants are
checked before the dataset is returned.
"""
function read_nbs_dataset(path::AbstractString)
    directory = abspath(path)
    isdir(directory) ||
        throw(ArgumentError("NBS dataset directory does not exist: $directory"))
    manifest_path = joinpath(directory, _NBS_DISK_MANIFEST)
    isfile(manifest_path) ||
        throw(ArgumentError("NBS dataset is missing $_NBS_DISK_MANIFEST"))
    _validate_nbs_manifest_hash(directory)
    manifest = TOML.parsefile(manifest_path)

    get(manifest, "format", nothing) == NBS_DISK_FORMAT_NAME ||
        throw(ArgumentError("unsupported NBS disk format"))
    get(manifest, "format_version", nothing) ==
        string(NBS_DISK_FORMAT_VERSION) ||
        throw(ArgumentError("unsupported NBS disk format version"))
    get(manifest, "dataset_schema_version", nothing) ==
        string(NBS_DATASET_SCHEMA_VERSION) ||
        throw(ArgumentError("unsupported NBS dataset schema version"))
    get(manifest, "axis_order", nothing) ==
        collect(string.(NBS_DATASET_AXIS_ORDER)) ||
        throw(ArgumentError("NBS four-point axis order mismatch"))
    get(manifest, "one_hadron_axis_order", nothing) ==
        ["time", "source", "sample"] ||
        throw(ArgumentError("NBS one-hadron axis order mismatch"))

    sample_kind_value = get(manifest, "sample_kind", nothing)
    sample_kind_value isa AbstractString ||
        throw(ArgumentError("manifest is missing sample_kind"))
    sample_kind = Symbol(sample_kind_value)

    metadata_value = get(manifest, "metadata", nothing)
    metadata_value isa AbstractDict ||
        throw(ArgumentError("manifest is missing metadata"))
    metadata = _metadata_namedtuple(metadata_value)

    arrays = get(manifest, "arrays", nothing)
    arrays isa AbstractDict ||
        throw(ArgumentError("manifest is missing arrays"))
    return NBSCorrelatorDataset(
        _read_nbs_array(directory, arrays, :four_point),
        _read_nbs_array(directory, arrays, :one_hadron_1),
        _read_nbs_array(directory, arrays, :one_hadron_2);
        metadata=metadata,
        sample_kind=sample_kind,
        schema_version=NBS_DATASET_SCHEMA_VERSION,
    )
end
