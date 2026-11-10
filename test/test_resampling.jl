@testset "jackknife" begin
    samples = [1.0, 2.0, 3.0, 4.0]
    estimates = jackknife_estimates(samples)
    @test estimates == [3.0, 8 / 3, 7 / 3, 2.0]
    summary = jackknife_summary(estimates)
    @test isapprox(summary.mean, 2.5)
    @test isapprox(summary.stderr, sqrt(5 / 12))

    arrays = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    array_estimates = jackknife_estimates(arrays, x -> x .^ 2)
    @test array_estimates[1] == [16.0, 25.0]
    array_summary = jackknife_summary(array_estimates)
    @test size(array_summary.mean) == (2,)
    @test all(array_summary.stderr .>= 0)

    @test_throws ArgumentError jackknife_estimates([1.0])
    @test_throws ArgumentError jackknife_summary([1.0])
end
