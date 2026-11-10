include(joinpath(@__DIR__, "..", "scripts", "analyze_spatial_nbs_potential.jl"))

@testset "spatial NBS potential analysis helpers" begin
    @test free_wilson_quark_pole_mass(0.1) ≈ log(2)
    @test free_wilson_quark_pole_mass(0.09f0) isa Float32
    @test_throws ArgumentError free_wilson_quark_pole_mass(0)
    @test_throws ArgumentError free_wilson_quark_pole_mass(1 / 8)

    values = reshape(ComplexF64.(1:8), 2, 2, 2)
    @test symmetric_relative_error(values, values) == 0
    @test symmetric_relative_error(
        values,
        permutedims(values, (1, 3, 2)),
    ) > 0
    @test_throws DimensionMismatch symmetric_relative_error(
        values,
        values[:, :, 1:1],
    )

    @test parse_selected_times("", [2, 3, 4]) == [2, 3, 4]
    @test parse_selected_times("4,2", [2, 3, 4]) == [4, 2]
    @test_throws ArgumentError parse_selected_times("2,2", [2, 3])
    @test_throws ArgumentError parse_selected_times("2,4", [2, 3])
end
