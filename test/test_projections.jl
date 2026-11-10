@testset "A1+ projection" begin
    n = 7
    constant_field = fill(3.25, n, n, n)
    @test a1plus_project(constant_field) == constant_field

    axis_delta = zeros(Float64, n, n, n)
    axis_delta[2, 1, 1] = 1
    projected = a1plus_project(axis_delta)
    axis_orbit = (
        (2, 1, 1),
        (n, 1, 1),
        (1, 2, 1),
        (1, n, 1),
        (1, 1, 2),
        (1, 1, n),
    )
    @test all(projected[index...] == 1 / 6 for index in axis_orbit)
    @test sum(projected) ≈ 1 atol=2e-15
    @test a1plus_project(projected) ≈ projected atol=2e-15

    aliased = copy(axis_delta)
    a1plus_project!(aliased, aliased)
    @test aliased == projected

    @test_throws DimensionMismatch a1plus_project(zeros(2, 3, 2))
end

@testset "spherical shell average" begin
    n = 5
    field = zeros(Float64, n, n, n)
    for z in 1:n, y in 1:n, x in 1:n
        cx = x - 1 <= n ÷ 2 ? x - 1 : x - 1 - n
        cy = y - 1 <= n ÷ 2 ? y - 1 : y - 1 - n
        cz = z - 1 <= n ÷ 2 ? z - 1 : z - 1 - n
        field[x, y, z] = cx^2 + cy^2 + cz^2
    end

    shells = shell_average(field; spacing=0.2)
    @test first(shells) == (r2=0, radius=0.0, mean=0.0, count=1)
    shell_one = only(filter(shell -> shell.r2 == 1, shells))
    @test shell_one.count == 6
    @test shell_one.radius == 0.2
    @test shell_one.mean == 1

    valid = trues(size(field))
    valid[2, 1, 1] = false
    masked = shell_average(field; valid=valid)
    @test only(filter(shell -> shell.r2 == 1, masked)).count == 5
    @test_throws DimensionMismatch shell_average(field; valid=trues(2, 2, 2))
end
