using HALQCD

rminus = ones(ComplexF64, 8, 8, 8)
rzero = ones(ComplexF64, 8, 8, 8)
rplus = ones(ComplexF64, 8, 8, 8)

result = lo_potential_equal_mass(
    rminus,
    rzero,
    rplus;
    mass=1.0,
    spacing=(1.0, 1.0, 1.0),
    at=1.0,
)

@assert all(result.valid)
@assert maximum(abs, result.potential) == 0
