@testset "equal-mass LO potential" begin
    rminus = ones(ComplexF64, 8, 8, 8)
    rzero = ones(ComplexF64, 8, 8, 8)
    rplus = ones(ComplexF64, 8, 8, 8)

    result = lo_potential_equal_mass(
        rminus,
        rzero,
        rplus;
        mass=1.5,
        spacing=(0.1, 0.1, 0.1),
        at=0.1,
    )
    @test result isa PotentialResult
    @test all(result.valid)
    @test maximum(abs, result.potential) < 1e-13
    @test result.numerator ==
          result.spatial .+ result.first_time .+ result.second_time

    scaled = lo_potential_equal_mass(
        7 .* rminus,
        7 .* rzero,
        7 .* rplus;
        mass=1.5,
        spacing=(0.1, 0.1, 0.1),
        at=0.1,
    )
    @test scaled.potential == result.potential
    @test scaled.valid == result.valid

    tiny = 1e-30 .* rzero
    tiny_result = lo_potential_equal_mass(
        tiny,
        tiny,
        tiny;
        mass=1.5,
        rtol=1e-3,
    )
    @test all(tiny_result.valid)

    zero_at_one_site = copy(rzero)
    zero_at_one_site[1, 1, 1] = 0
    masked = lo_potential_equal_mass(
        rminus,
        zero_at_one_site,
        rplus;
        mass=1.5,
        rtol=1e-12,
    )
    @test !masked.valid[1, 1, 1]
    @test masked.potential[1, 1, 1] == 0

    nx = 16
    k = 2
    energy = 0.3
    at = 0.2
    mass = 1.7
    wave = Array{ComplexF64}(undef, nx, 3, 2)
    for z in axes(wave, 3), y in axes(wave, 2), x in axes(wave, 1)
        wave[x, y, z] = cis(2 * pi * k * (x - 1) / nx)
    end
    manufactured = lo_potential_equal_mass(
        exp(energy * at) .* wave,
        wave,
        exp(-energy * at) .* wave;
        mass=mass,
        at=at,
    )
    lambda = 2 * cos(2 * pi * k / nx) - 2
    expected =
        lambda / mass +
        sinh(energy * at) / at +
        (cosh(energy * at) - 1) / (2 * mass * at^2)
    @test maximum(abs, manufactured.potential .- expected) < 1e-12

    @test_throws ArgumentError lo_potential_equal_mass(
        rminus,
        rzero,
        rplus;
        mass=0.0,
    )

    components = divided_potential_components(manufactured, wave)
    @test components.total == manufactured.potential
    @test maximum(
        abs,
        components.spatial .+
        components.first_time .+
        components.second_time .-
        components.total,
    ) < 1e-12
end

@testset "NBS dataset to equal-mass LO potential" begin
    nx, ny, nz, nt = 8, 3, 2, 3
    energy = 0.2
    at = 0.1
    mass = 1.4
    wave = Array{ComplexF64}(undef, nx, ny, nz)
    for z in 1:nz, y in 1:ny, x in 1:nx
        wave[x, y, z] = cis(2 * pi * (x - 1) / nx)
    end
    normalized = (
        exp(energy * at) .* wave,
        wave,
        exp(-energy * at) .* wave,
    )
    one_hadron_1 = reshape([2.0, 3.0, 4.0], nt, 1, 1)
    one_hadron_2 = reshape([5.0, 7.0, 11.0], nt, 1, 1)
    four_point = Array{ComplexF64}(undef, nx, ny, nz, nt, 1, 1)
    for time in 1:nt
        four_point[:, :, :, time, 1, 1] .=
            normalized[time] .* one_hadron_1[time] .* one_hadron_2[time]
    end
    dataset = NBSCorrelatorDataset(
        four_point,
        one_hadron_1,
        one_hadron_2;
        metadata=_nbs_fixture_metadata(
            lattice_shape=(nx, ny, nz, nt),
            lattice_spacing=(0.6, 0.7, 0.8, at),
            sample_ids=["cfg-1"],
            source_ids=["src-1"],
        ),
    )
    direct = lo_potential_equal_mass(
        normalized...;
        mass=mass,
        spacing=(0.6, 0.7, 0.8),
        at=at,
    )
    from_dataset = lo_potential_equal_mass(
        dataset,
        2;
        mass=mass,
    )
    @test from_dataset.potential ≈ direct.potential
    @test from_dataset.valid == direct.valid
    @test from_dataset.spatial ≈ direct.spatial
    @test from_dataset.first_time ≈ direct.first_time
    @test from_dataset.second_time ≈ direct.second_time

    unavailable_one_hadron = copy(one_hadron_1)
    unavailable_one_hadron[1] = 0
    unavailable = NBSCorrelatorDataset(
        four_point,
        unavailable_one_hadron,
        one_hadron_2;
        metadata=dataset.metadata,
    )
    @test_throws DomainError lo_potential_equal_mass(
        unavailable,
        2;
        mass=mass,
    )
    @test_throws ArgumentError lo_potential_equal_mass(
        dataset,
        1;
        mass=mass,
    )
    @test_throws BoundsError lo_potential_equal_mass(
        dataset,
        2;
        source=2,
        mass=mass,
    )
end

@testset "jackknife potential pipeline" begin
    n = 4
    samples = 3
    rzero = ones(Float64, n, n, n, samples)
    rminus = similar(rzero)
    rplus = similar(rzero)
    energies = (0.1, 0.2, 0.3)
    for sample in 1:samples
        rminus[:, :, :, sample] .= exp(energies[sample])
        rplus[:, :, :, sample] .= exp(-energies[sample])
    end

    result = lo_potential_from_jackknife_equal_mass(
        rminus,
        rzero,
        rplus;
        mass=1.0,
    )
    @test result isa JackknifePotentialResult
    @test length(result.estimates) == samples
    @test all(result.valid)
    direct = jackknife_summary([estimate.potential for estimate in result.estimates])
    @test result.components.total.mean == direct.mean
    @test result.components.total.stderr == direct.stderr

    @test_throws ArgumentError lo_potential_from_jackknife_equal_mass(
        rminus[:, :, :, 1:1],
        rzero[:, :, :, 1:1],
        rplus[:, :, :, 1:1];
        mass=1.0,
    )
end
