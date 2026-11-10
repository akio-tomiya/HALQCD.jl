using HALQCD

dims = (8, 8, 4, 4)
nflux = 1
links = uniform_u1_links(dims, nflux; charge_ratio=2)
target = cis(flux_qBa2(nflux, dims[1], dims[2]; charge_ratio=2))

for t in 1:dims[4], z in 1:dims[3], y in 1:dims[2], x in 1:dims[1]
    @assert isapprox(
        u1_plaquette_xy(links, x, y, z, t),
        target;
        atol=1e-12,
        rtol=1e-12,
    )
end
