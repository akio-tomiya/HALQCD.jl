module HALQCDMPIExt

# MPI communication of the JuliaQCD extension, kept apart as LatticeMatrices.jl
# keeps its own in LatticeMatricesMPIExt: HALQCD.jl itself never loads MPI.jl.

using HALQCD
using MPI

# Correlators contracted on the time slices of each process are summed here.
function HALQCD._sum_over_processes!(comm::MPI.Comm, values::AbstractArray)
    MPI.Allreduce!(values, MPI.SUM, comm)
    return values
end

# The processes with the same time coordinate hold the spatial blocks of the
# same time slices. For each local slice, the process that contracts it
# (HALQCD._assigned_times) gathers the blocks of the others and places them by
# their Cartesian coordinates. The blocks have equal sizes (LatticeMatrices.jl
# requires the lattice to be divisible by the process grid).
function HALQCD._time_slices(comm::MPI.Comm, block::AbstractArray, coords, dims, shape)
    local_size = size(block)[(end - 3):end]
    internal = size(block)[1:(end - 4)]
    nspatial = dims[1] * dims[2] * dims[3]
    spatial_rank = coords[1] + dims[1] * (coords[2] + dims[2] * coords[3])
    times = HALQCD._assigned_times(coords, dims, local_size)
    slices = zeros(eltype(block), internal..., shape[1], shape[2], shape[3], length(times))
    nspatial == 1 && return copyto!(slices, block)

    group = MPI.Comm_split(comm, coords[4], spatial_rank)
    MPI.Comm_rank(group) == spatial_rank ||
        error("unexpected rank order in the spatial subcommunicator")
    count = prod(internal) * prod(local_size[1:3])
    colons = ntuple(_ -> Colon(), length(internal))
    received = 0
    for j in 1:local_size[4]
        root = (j - 1) % nspatial
        send = Array(selectdim(block, ndims(block), j))
        if spatial_rank == root
            buffer = Vector{eltype(block)}(undef, count * nspatial)
            MPI.Gather!(send, MPI.UBuffer(buffer, count), group; root)
            received += 1
            for r in 0:(nspatial - 1)
                cx = r % dims[1]
                cy = (r ÷ dims[1]) % dims[2]
                cz = r ÷ (dims[1] * dims[2])
                piece = reshape(view(buffer, (r * count + 1):((r + 1) * count)),
                    internal..., local_size[1:3]...)
                slices[colons..., cx * local_size[1] .+ (1:local_size[1]),
                    cy * local_size[2] .+ (1:local_size[2]),
                    cz * local_size[3] .+ (1:local_size[3]), received] .= piece
            end
        else
            MPI.Gather!(send, nothing, group; root)
        end
    end
    MPI.free(group)
    return slices
end

end
