@testset "zero-energy S-wave scattering length" begin
    radii = collect(0.0:0.01:8.0)
    free_result = s_wave_scattering_length(
        radii,
        zeros(length(radii));
        reduced_mass=0.5,
        tail_start=4.0,
    )
    @test free_result isa ScatteringLengthResult
    @test abs(free_result.value) < 1e-11
    @test free_result.tail_spread < 1e-11

    reduced_mass = 0.5
    depth = 0.5
    radius = 1.2
    fine_radii = collect(0.0:0.001:8.0)
    square_well = [
        r < radius ? -depth : 0.0
        for r in fine_radii
    ]
    well_result = s_wave_scattering_length(
        fine_radii,
        square_well;
        reduced_mass=reduced_mass,
        tail_start=3.0,
    )
    momentum = sqrt(2reduced_mass * depth)
    analytic = radius - tan(momentum * radius) / momentum
    @test well_result.value ≈ analytic atol=1.5e-3
    @test well_result.tail_spread < 1e-9

    square_barrier = [
        r < radius ? depth : 0.0
        for r in fine_radii
    ]
    barrier_result = s_wave_scattering_length(
        fine_radii,
        square_barrier;
        reduced_mass=reduced_mass,
        tail_start=3.0,
    )
    barrier_analytic = radius - tanh(momentum * radius) / momentum
    @test barrier_result.value ≈ barrier_analytic atol=1.5e-3
    @test barrier_result.tail_spread < 1e-9

    @test_throws DimensionMismatch s_wave_scattering_length(
        radii,
        zeros(length(radii) - 1);
        reduced_mass=0.5,
        tail_start=4.0,
    )
    @test_throws ArgumentError s_wave_scattering_length(
        [0.0, 0.1, 0.21, 0.3, 0.4],
        zeros(5);
        reduced_mass=0.5,
        tail_start=0.2,
    )
end
