@testset "Lüscher zeta function" begin
    # √(4π) Z₀₀(1; q²) + 1/q² tends to -8.91363292 as q² → 0.
    q2 = 1e-6
    @test sqrt(4π) * luscher_zeta(q2) + 1 / q2 ≈ -8.91363292 atol = 1e-4
    @test_throws ArgumentError luscher_zeta(1.0)

    # Threshold expansion: p² = 4πa/L³ [1 - c₁ a/L + c₂ (a/L)²] for a in the
    # convention u(r) ∝ r - a at zero energy; the omitted terms are O((a/L)³).
    c1, c2 = -2.837297, 6.375183
    for (a, L) in ((0.3, 24.0), (-0.3, 24.0), (1.0, 40.0))
        p2 = 4π * a / L^3 * (1 - c1 * a / L + c2 * (a / L)^2)
        @test -1 / luscher_pcotdelta(p2, L) ≈ a rtol = 20 * abs(a / L)^3
    end

    # Energy shift to momentum. At small momentum the lattice relation gives
    # p² ≈ sinh(m) ΔE instead of m ΔE.
    m = 0.5
    p2 = luscher_momentum_squared(0.002, m)
    @test p2 ≈ (m + 0.001)^2 - m^2
    @test luscher_momentum_squared(0.002, m; dispersion=:lattice) ≈ sinh(m) * 0.002 rtol = 2e-3
    @test luscher_momentum_squared(-0.002, m; dispersion=:lattice) < 0
    @test luscher_scattering_length(0.002, m, 16.0) > 0
    @test_throws ArgumentError luscher_momentum_squared(0.1, m; dispersion=:other)
end
