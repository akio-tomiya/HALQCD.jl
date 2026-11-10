@testset "R-correlator normalization" begin
    four_point = reshape(ComplexF64.(1:8), 2, 2, 2)
    result = r_correlator(four_point, 2.0, 0.5)
    @test result == four_point

    out = similar(four_point)
    @test r_correlator!(out, four_point, 2.0, 4.0) === out
    @test out == four_point ./ 8

    @test_throws DomainError r_correlator(four_point, 0.0, 1.0)
    @test_throws DomainError r_correlator(
        four_point,
        1e-14,
        1.0;
        atol=1e-12,
    )
    @test_throws DimensionMismatch r_correlator!(
        zeros(2, 2),
        four_point,
        1.0,
        1.0,
    )
end
