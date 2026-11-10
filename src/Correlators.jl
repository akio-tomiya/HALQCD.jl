"""
    r_correlator!(out, four_point, one_hadron_1, one_hadron_2; atol=0)

Construct a single-time-slice normalized correlator

`R(x,y,z,t) = F(x,y,z,t) / (C₁(t) C₂(t))`.

The one-hadron correlators must use the same time slice, spin projection, and
background-field setup as `four_point`.
"""
function r_correlator!(
    out::AbstractArray,
    four_point::AbstractArray,
    one_hadron_1::Number,
    one_hadron_2::Number;
    atol::Real=0,
)
    size(out) == size(four_point) ||
        throw(DimensionMismatch("out and four_point must have identical sizes"))
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))

    denominator = one_hadron_1 * one_hadron_2
    magnitude = abs(denominator)
    isfinite(magnitude) && magnitude > atol ||
        throw(DomainError(denominator, "one-hadron normalization is too small or non-finite"))

    out .= four_point ./ denominator
    return out
end

"""
    r_correlator(four_point, one_hadron_1, one_hadron_2; atol=0)

Allocating form of [`r_correlator!`](@ref).
"""
function r_correlator(
    four_point::AbstractArray{TF},
    one_hadron_1::T1,
    one_hadron_2::T2;
    atol::Real=0,
) where {TF,T1<:Number,T2<:Number}
    denominator = one_hadron_1 * one_hadron_2
    Tout = Base.promote_op(/, TF, typeof(denominator))
    out = similar(four_point, Tout)
    return r_correlator!(
        out,
        four_point,
        one_hadron_1,
        one_hadron_2;
        atol=atol,
    )
end
