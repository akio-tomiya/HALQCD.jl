@testset "uniform U(1) magnetic flux" begin
    dims = (7, 5, 3, 2)
    nflux = 2
    charge_ratio = -1
    qBa2 = flux_qBa2(
        nflux,
        dims[1],
        dims[2];
        charge_ratio=charge_ratio,
    )
    links = uniform_u1_links(
        dims,
        nflux;
        charge_ratio=charge_ratio,
    )
    target = cis(qBa2)

    for t in 1:dims[4], z in 1:dims[3], y in 1:dims[2], x in 1:dims[1]
        @test isapprox(
            u1_plaquette_xy(links, x, y, z, t),
            target;
            atol=1e-12,
            rtol=1e-12,
        )
    end

    zero_links = uniform_u1_links(dims, 0)
    @test all(zero_links .== 1)

    links32 = uniform_u1_links(dims, 1; float_type=Float32)
    @test eltype(links32) == ComplexF32

    @test isapprox(flux_eBa2(1, 8, 8), 6 * pi / 64; atol=1e-15)

    su3 = zeros(ComplexF64, 3, 3, 4, dims...)
    for t in 1:dims[4], z in 1:dims[3], y in 1:dims[2], x in 1:dims[1],
        mu in 1:4, c in 1:3
        su3[c, c, mu, x, y, z, t] = 1
    end
    su3_before = copy(su3)
    u3 = apply_u1_links(su3, links)
    @test su3 == su3_before
    @test u3[2, 2, 2, 3, 2, 1, 1] == links[2, 3, 2, 1, 1]

    inplace = copy(su3)
    @test apply_u1_links!(inplace, inplace, links) === inplace
    @test inplace == u3

    @test_throws ArgumentError uniform_u1_links((0, 2, 2, 2), 1)
    @test_throws DimensionMismatch apply_u1_links(
        su3,
        ones(ComplexF64, 4, 2, 2, 2, 2),
    )
end

@testset "gauge-covariant spatial Laplacian" begin
    dims = (6, 5, 4, 2)
    zero_links = uniform_u1_links(dims, 0)
    constant_field = ones(ComplexF64, dims[1:3])
    @test covariant_laplacian_periodic(constant_field, zero_links) ==
          zeros(ComplexF64, dims[1:3])

    momentum = (1, 2, 1)
    plane_wave = Array{ComplexF64}(undef, dims[1:3])
    for z in 1:dims[3], y in 1:dims[2], x in 1:dims[1]
        plane_wave[x, y, z] = cis(
            2pi * (
                momentum[1] * (x - 1) / dims[1] +
                momentum[2] * (y - 1) / dims[2] +
                momentum[3] * (z - 1) / dims[3]
            ),
        )
    end
    eigenvalue = sum(
        2cos(2pi * momentum[axis] / dims[axis]) - 2
        for axis in 1:3
    )
    plane_wave_residual = maximum(
        abs,
        covariant_laplacian_periodic(plane_wave, zero_links) .-
        eigenvalue .* plane_wave,
    )
    @test plane_wave_residual < 5e-14

    links = uniform_u1_links(dims, 2; charge_ratio=-1)
    field = copy(plane_wave)
    gauge = Array{ComplexF64}(undef, dims[1:3])
    for z in 1:dims[3], y in 1:dims[2], x in 1:dims[1]
        gauge[x, y, z] = cis(
            0.17 * (x - 1) +
            0.11 * (y - 1)^2 -
            0.07 * (z - 1),
        )
    end
    transformed_links = copy(links)
    for t in 1:dims[4], z in 1:dims[3], y in 1:dims[2], x in 1:dims[1]
        neighbors = (
            (mod1(x + 1, dims[1]), y, z),
            (x, mod1(y + 1, dims[2]), z),
            (x, y, mod1(z + 1, dims[3])),
        )
        for mu in 1:3
            transformed_links[mu, x, y, z, t] =
                gauge[x, y, z] *
                links[mu, x, y, z, t] *
                conj(gauge[neighbors[mu]...])
        end
    end
    original_result = covariant_laplacian_periodic(field, links)
    transformed_result = covariant_laplacian_periodic(
        gauge .* field,
        transformed_links,
    )
    @test transformed_result ≈ gauge .* original_result atol=3e-14

    aliased = copy(field)
    covariant_laplacian_periodic!(aliased, aliased, links)
    @test aliased == original_result
    @test_throws DimensionMismatch covariant_laplacian_periodic(
        field,
        ones(ComplexF64, 4, 2, 2, 2, 1),
    )
end
