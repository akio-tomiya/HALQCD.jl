using Random

@testset "point-source pion correlator" begin
    rng = MersenneTwister(7)
    for ncolor in (2, 3, 7)
        propagator = randn(rng, ComplexF64, ncolor, 4, ncolor, 4, 2, 2, 2, 3)
        correlator = pion_correlator_point_source(propagator)
        @test length(correlator) == 3
        for t in 1:3
            expected = 0.0
            for x in 1:2, y in 1:2, z in 1:2,
                    a in 1:ncolor, α in 1:4, b in 1:ncolor, β in 1:4
                expected += abs2(propagator[a, α, b, β, x, y, z, t])
            end
            @test correlator[t] ≈ expected rtol = 1e-13
        end
    end
    @test_throws DimensionMismatch pion_correlator_point_source(
        zeros(ComplexF64, 3, 4, 2, 4, 2, 2, 2, 2),
    )
    # Without the JuliaQCD extension the field-dependent functions have no
    # methods.
    @test_throws MethodError gauge_group(nothing)
    @test_throws MethodError juliaqcd_point_propagator(nothing; κ=0.1)
end
