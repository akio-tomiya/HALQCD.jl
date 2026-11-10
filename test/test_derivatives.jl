@testset "periodic derivatives" begin
    field = fill(3.0, 8, 7, 6)
    lap = laplacian_periodic(field, (1.0, 1.0, 1.0))
    @test maximum(abs, lap) == 0.0

    field32 = fill(3.0f0, 4, 4, 4)
    @test eltype(laplacian_periodic(field32, (1.0f0, 1.0f0, 1.0f0))) == Float32

    complex32 = fill(ComplexF32(2 + 3im), 4, 4, 4)
    @test eltype(laplacian_periodic(complex32, (1.0f0, 1.0f0, 1.0f0))) ==
          ComplexF32

    nx, ny, nz = 16, 5, 4
    k = 2
    wave = Array{ComplexF64}(undef, nx, ny, nz)
    for z in 1:nz, y in 1:ny, x in 1:nx
        wave[x, y, z] = cis(2 * pi * k * (x - 1) / nx)
    end
    lap_wave = laplacian_periodic(wave, (1.0, 1.0, 1.0))
    eigenvalue = 2 * cos(2 * pi * k / nx) - 2
    @test maximum(abs, lap_wave .- eigenvalue .* wave) < 1e-12

    out = similar(wave)
    @test laplacian_periodic!(out, wave, (1.0, 1.0, 1.0)) === out
    @test out == lap_wave
    @test_throws ArgumentError laplacian_periodic!(wave, wave, (1.0, 1.0, 1.0))
    @test_throws ArgumentError laplacian_periodic(wave, (1.0, 0.0, 1.0))

    rminus = fill(1.0, 2, 2, 2)
    rzero = fill(2.0, 2, 2, 2)
    rplus = fill(5.0, 2, 2, 2)
    dt, d2t = time_derivatives(rminus, rzero, rplus, 1.0)
    @test all(dt .== 2.0)
    @test all(d2t .== 2.0)
    @test_throws ArgumentError time_derivatives(rminus, rzero, rplus, 0.0)
    @test_throws DimensionMismatch time_derivatives(
        ones(2, 2),
        ones(2, 2, 2),
        ones(2, 2, 2),
        1.0,
    )
end
