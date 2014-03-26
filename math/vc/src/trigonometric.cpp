/*  This file is part of the Vc library. {{{

    Copyright (C) 2012 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

}}}*/

#include <Vc/Vc>
#if defined(VC_IMPL_SSE) || defined(VC_IMPL_AVX)
#include <Vc/common/macros.h>

namespace ROOT {
namespace Vc
{
namespace
{
    using Vc::Vector;
    using Vc::float_v;
    using Vc::double_v;
    using Vc::sfloat_v;

    template<typename T> static Vc_ALWAYS_INLINE Vector<T> cosSeries(const Vector<T> &x)
    {
        typedef Const<T> C;
        const Vector<T> x2 = x * x;
        return ((C::cosCoeff(2)  * x2 +
                 C::cosCoeff(1)) * x2 +
                 C::cosCoeff(0)) * (x2 * x2)
            - C::_1_2() * x2 + Vector<T>::One();
    }
    static Vc_ALWAYS_INLINE double_v cosSeries(const double_v &x)
    {
        typedef Const<double> C;
        const double_v x2 = x * x;
        return (((((C::cosCoeff(5)  * x2 +
                    C::cosCoeff(4)) * x2 +
                    C::cosCoeff(3)) * x2 +
                    C::cosCoeff(2)) * x2 +
                    C::cosCoeff(1)) * x2 +
                    C::cosCoeff(0)) * (x2 * x2)
            - C::_1_2() * x2 + double_v::One();
    }
    template<typename T> static Vc_ALWAYS_INLINE Vector<T> sinSeries(const Vector<T> &x)
    {
        typedef Const<T> C;
        const Vector<T> x2 = x * x;
        return ((C::sinCoeff(2)  * x2 +
                 C::sinCoeff(1)) * x2 +
                 C::sinCoeff(0)) * (x2 * x)
            + x;
    }
    static Vc_ALWAYS_INLINE double_v sinSeries(const double_v &x)
    {
        typedef Const<double> C;
        const double_v x2 = x * x;
        return (((((C::sinCoeff(5)  * x2 +
                    C::sinCoeff(4)) * x2 +
                    C::sinCoeff(3)) * x2 +
                    C::sinCoeff(2)) * x2 +
                    C::sinCoeff(1)) * x2 +
                    C::sinCoeff(0)) * (x2 * x)
            + x;
    }
    template<typename V> struct signed_integer { typedef   int_v type; };
    template<> struct signed_integer<sfloat_v> { typedef short_v type; };

    template<typename _T, typename IV> static Vc_ALWAYS_INLINE Vector<_T> foldInput(const Vector<_T> &_x, IV &quadrant)
    {
        typedef Vector<_T> V;
        typedef Const<_T> C;

        const V x = abs(_x);
#if defined(VC_IMPL_FMA4) || defined(VC_IMPL_FMA)
        quadrant = static_cast<IV>(x * C::_4_pi() + V::One()); // prefer the fma here
        quadrant &= ~IV::One();
#else
        quadrant = static_cast<IV>(x * C::_4_pi());
        quadrant += quadrant & IV::One();
#endif
        const V y = static_cast<V>(quadrant);
        quadrant &= 7;

        return ((x - y * C::_pi_4_hi()) - y * C::_pi_4_rem1()) - y * C::_pi_4_rem2();
    }
    static Vc_ALWAYS_INLINE double_v foldInput(const double_v &_x, int_v &quadrant)
    {
        typedef double_v V;
        typedef Const<double> C;

        const V x = abs(_x);
        V y = trunc(x / C::_pi_4()); // * C::_4_pi() would work, but is >twice as imprecise
        V z = y - trunc(y * C::_1_16()) * C::_16(); // y modulo 16
        quadrant = static_cast<int_v>(z);
        int_m mask = (quadrant & int_v::One()) != int_v::Zero();
        ++quadrant(mask);
        y(static_cast<double_m>(mask)) += V::One();
        quadrant &= 7;

        // since y is an integer we don't need to split y into low and high parts until the integer
        // requires more bits than there are zero bits at the end of _pi_4_hi (30 bits -> 1e9)
        return ((x - y * C::_pi_4_hi()) - y * C::_pi_4_rem1()) - y * C::_pi_4_rem2();
    }
} // anonymous namespace

/*
 * algorithm for sine and cosine:
 *
 * The result can be calculated with sine or cosine depending on the π/4 section the input is
 * in.
 * sine   ≈ x + x³
 * cosine ≈ 1 - x²
 *
 * sine:
 * Map -x to x and invert the output
 * Extend precision of x - n * π/4 by calculating
 * ((x - n * p1) - n * p2) - n * p3 (p1 + p2 + p3 = π/4)
 *
 * Calculate Taylor series with tuned coefficients.
 * Fix sign.
 */
template<> template<typename _T> Vector<_T> Trigonometric<Vc::Internal::TrigonometricImplementation>::sin(const Vector<_T> &_x)
{
    typedef Vector<_T> V;
    typedef typename V::Mask M;
    typedef typename signed_integer<V>::type IV;

    IV quadrant;
    const V z = foldInput(_x, quadrant);
    const M sign = (_x < V::Zero()) ^ static_cast<M>(quadrant > 3);
    quadrant(quadrant > 3) -= 4;

    V y = sinSeries(z);
    y(quadrant == IV::One() || quadrant == 2) = cosSeries(z);
    y(sign) = -y;
    return y;
}

template<> template<> double_v Trigonometric<Vc::Internal::TrigonometricImplementation>::sin(const double_v &_x)
{
    typedef double_v V;
    typedef V::Mask M;

    int_v quadrant;
    M sign = _x < V::Zero();
    const V x = foldInput(_x, quadrant);
    sign ^= static_cast<M>(quadrant > 3);
    quadrant(quadrant > 3) -= 4;

    V y = sinSeries(x);
    y(static_cast<M>(quadrant == int_v::One() || quadrant == 2)) = cosSeries(x);
    y(sign) = -y;
    return y;
}
template<> template<typename _T> Vector<_T> Trigonometric<Vc::Internal::TrigonometricImplementation>::cos(const Vector<_T> &_x) {
    typedef Vector<_T> V;
    typedef typename V::Mask M;
    typedef typename signed_integer<V>::type IV;

    IV quadrant;
    const V x = foldInput(_x, quadrant);
    M sign = quadrant > 3;
    quadrant(quadrant > 3) -= 4;
    sign ^= quadrant > IV::One();

    V y = cosSeries(x);
    y(quadrant == IV::One() || quadrant == 2) = sinSeries(x);
    y(sign) = -y;
    return y;
}
template<> template<> double_v Trigonometric<Vc::Internal::TrigonometricImplementation>::cos(const double_v &_x)
{
    typedef double_v V;
    typedef V::Mask M;

    int_v quadrant;
    const V x = foldInput(_x, quadrant);
    M sign = static_cast<M>(quadrant > 3);
    quadrant(quadrant > 3) -= 4;
    sign ^= static_cast<M>(quadrant > int_v::One());

    V y = cosSeries(x);
    y(static_cast<M>(quadrant == int_v::One() || quadrant == 2)) = sinSeries(x);
    y(sign) = -y;
    return y;
}
template<> template<typename _T> void Trigonometric<Vc::Internal::TrigonometricImplementation>::sincos(const Vector<_T> &_x, Vector<_T> *_sin, Vector<_T> *_cos) {
    typedef Vector<_T> V;
    typedef typename V::Mask M;
    typedef typename signed_integer<V>::type IV;

    IV quadrant;
    const V x = foldInput(_x, quadrant);
    M sign = static_cast<M>(quadrant > 3);
    quadrant(quadrant > 3) -= 4;

    const V cos_s = cosSeries(x);
    const V sin_s = sinSeries(x);

    V c = cos_s;
    c(static_cast<M>(quadrant == IV::One() || quadrant == 2)) = sin_s;
    c(sign ^ static_cast<M>(quadrant > IV::One())) = -c;
    *_cos = c;

    V s = sin_s;
    s(static_cast<M>(quadrant == IV::One() || quadrant == 2)) = cos_s;
    s(sign ^ static_cast<M>(_x < V::Zero())) = -s;
    *_sin = s;
}
template<> template<> void Trigonometric<Vc::Internal::TrigonometricImplementation>::sincos(const double_v &_x, double_v *_sin, double_v *_cos) {
    typedef double_v V;
    typedef V::Mask M;

    int_v quadrant;
    const V x = foldInput(_x, quadrant);
    M sign = static_cast<M>(quadrant > 3);
    quadrant(quadrant > 3) -= 4;

    const V cos_s = cosSeries(x);
    const V sin_s = sinSeries(x);

    V c = cos_s;
    c(static_cast<M>(quadrant == int_v::One() || quadrant == 2)) = sin_s;
    c(sign ^ static_cast<M>(quadrant > int_v::One())) = -c;
    *_cos = c;

    V s = sin_s;
    s(static_cast<M>(quadrant == int_v::One() || quadrant == 2)) = cos_s;
    s(sign ^ static_cast<M>(_x < V::Zero())) = -s;
    *_sin = s;
}
template<> template<typename _T> Vector<_T> Trigonometric<Vc::Internal::TrigonometricImplementation>::asin (const Vector<_T> &_x) {
    typedef Const<_T> C;
    typedef Vector<_T> V;
    typedef typename V::Mask M;

    const M &negative = _x < V::Zero();

    const V &a = abs(_x);
    const M outOfRange = a > V::One();
    const M &small = a < C::smallAsinInput();
    const M &gt_0_5 = a > C::_1_2();
    V x = a;
    V z = a * a;
    z(gt_0_5) = (V::One() - a) * C::_1_2();
    x(gt_0_5) = sqrt(z);
    z = ((((C::asinCoeff0(0)  * z
          + C::asinCoeff0(1)) * z
          + C::asinCoeff0(2)) * z
          + C::asinCoeff0(3)) * z
          + C::asinCoeff0(4)) * z * x
          + x;
    z(gt_0_5) = C::_pi_2() - (z + z);
    z(small) = a;
    z(negative) = -z;
    z.setQnan(outOfRange);

    return z;
}
template<> template<> double_v Trigonometric<Vc::Internal::TrigonometricImplementation>::asin (const double_v &_x) {
    typedef Const<double> C;
    typedef double_v V;
    typedef V::Mask M;

    const M negative = _x < V::Zero();

    const V a = abs(_x);
    const M outOfRange = a > V::One();
    const M small = a < C::smallAsinInput();
    const M large = a > C::largeAsinInput();

    V zz = V::One() - a;
    const V r = (((C::asinCoeff0(0) * zz + C::asinCoeff0(1)) * zz + C::asinCoeff0(2)) * zz +
            C::asinCoeff0(3)) * zz + C::asinCoeff0(4);
    const V s = (((zz + C::asinCoeff1(0)) * zz + C::asinCoeff1(1)) * zz +
            C::asinCoeff1(2)) * zz + C::asinCoeff1(3);
    V sqrtzz = sqrt(zz + zz);
    V z = C::_pi_4() - sqrtzz;
    z -= sqrtzz * (zz * r / s) - C::_pi_2_rem();
    z += C::_pi_4();

    V a2 = a * a;
    const V p = ((((C::asinCoeff2(0) * a2 + C::asinCoeff2(1)) * a2 + C::asinCoeff2(2)) * a2 +
                C::asinCoeff2(3)) * a2 + C::asinCoeff2(4)) * a2 + C::asinCoeff2(5);
    const V q = ((((a2 + C::asinCoeff3(0)) * a2 + C::asinCoeff3(1)) * a2 +
                C::asinCoeff3(2)) * a2 + C::asinCoeff3(3)) * a2 + C::asinCoeff3(4);
    z(!large) = a * (a2 * p / q) + a;

    z(negative) = -z;
    z(small) = _x;
    z.setQnan(outOfRange);

    return z;
}
template<> template<typename _T> Vector<_T> Trigonometric<Vc::Internal::TrigonometricImplementation>::atan (const Vector<_T> &_x) {
    typedef Const<_T> C;
    typedef Vector<_T> V;
    typedef typename V::Mask M;
    V x = abs(_x);
    const M &gt_tan_3pi_8 = x > C::atanThrsHi();
    const M &gt_tan_pi_8  = x > C::atanThrsLo() && !gt_tan_3pi_8;
    V y = V::Zero();
    y(gt_tan_3pi_8) = C::_pi_2();
    y(gt_tan_pi_8)  = C::_pi_4();
    x(gt_tan_3pi_8) = -V::One() / x;
    x(gt_tan_pi_8)  = (x - V::One()) / (x + V::One());
    const V &x2 = x * x;
    y += (((C::atanP(0)  * x2
          - C::atanP(1)) * x2
          + C::atanP(2)) * x2
          - C::atanP(3)) * x2 * x
          + x;
    y(_x < V::Zero()) = -y;
    y.setQnan(isnan(_x));
    return y;
}
template<> template<> double_v Trigonometric<Vc::Internal::TrigonometricImplementation>::atan (const double_v &_x) {
    typedef Const<double> C;
    typedef double_v V;
    typedef V::Mask M;

    M sign = _x < V::Zero();
    V x = abs(_x);
    M finite = isfinite(_x);
    V ret = C::_pi_2();
    V y = V::Zero();
    const M large = x > C::atanThrsHi();
    const M gt_06 = x > C::atanThrsLo();
    V tmp = (x - V::One()) / (x + V::One());
    tmp(large) = -V::One() / x;
    x(gt_06) = tmp;
    y(gt_06) = C::_pi_4();
    y(large) = C::_pi_2();
    V z = x * x;
    const V p = (((C::atanP(0) * z + C::atanP(1)) * z + C::atanP(2)) * z + C::atanP(3)) * z + C::atanP(4);
    const V q = ((((z + C::atanQ(0)) * z + C::atanQ(1)) * z + C::atanQ(2)) * z + C::atanQ(3)) * z + C::atanQ(4);
    z = z * p / q;
    z = x * z + x;
    V morebits = C::_pi_2_rem();
    morebits(!large) *= C::_1_2();
    z(gt_06) += morebits;
    ret(finite) = y + z;
    ret(sign) = -ret;
    ret.setQnan(isnan(_x));
    return ret;
}
template<> template<typename _T> Vector<_T> Trigonometric<Vc::Internal::TrigonometricImplementation>::atan2(const Vector<_T> &y, const Vector<_T> &x) {
    typedef Const<_T> C;
    typedef Vector<_T> V;
    typedef typename V::Mask M;

    const M xZero = x == V::Zero();
    const M yZero = y == V::Zero();
    const M xMinusZero = xZero && x.isNegative();
    const M yNeg = y < V::Zero();
    const M xInf = !isfinite(x);
    const M yInf = !isfinite(y);

    V a = C::_pi().copySign(y);
    a.setZero(x >= V::Zero());

    // setting x to any finite value will have atan(y/x) return sign(y/x)*pi/2, just in case x is inf
    V _x = x;
    _x(yInf) = V::One().copySign(x);

    a += atan(y / _x);

    // if x is +0 and y is +/-0 the result is +0
    a.setZero(xZero && yZero);

    // for x = -0 we add/subtract pi to get the correct result
    a(xMinusZero) += C::_pi().copySign(y);

    // atan2(-Y, +/-0) = -pi/2
    a(xZero && yNeg) = -C::_pi_2();

    // if both inputs are inf the output is +/- (3)pi/4
    a(xInf && yInf) += C::_pi_4().copySign(x ^ ~y);

    // correct the sign of y if the result is 0
    a(a == V::Zero()) = a.copySign(y);

    // any NaN input will lead to NaN output
    a.setQnan(isnan(y) || isnan(x));

    return a;
}
template<> template<> double_v Trigonometric<Vc::Internal::TrigonometricImplementation>::atan2 (const double_v &y, const double_v &x) {
    typedef Const<double> C;
    typedef double_v V;
    typedef V::Mask M;

    const M xZero = x == V::Zero();
    const M yZero = y == V::Zero();
    const M xMinusZero = xZero && x.isNegative();
    const M yNeg = y < V::Zero();
    const M xInf = !isfinite(x);
    const M yInf = !isfinite(y);

    V a = V(C::_pi()).copySign(y);
    a.setZero(x >= V::Zero());

    // setting x to any finite value will have atan(y/x) return sign(y/x)*pi/2, just in case x is inf
    V _x = x;
    _x(yInf) = V::One().copySign(x);

    a += atan(y / _x);

    // if x is +0 and y is +/-0 the result is +0
    a.setZero(xZero && yZero);

    // for x = -0 we add/subtract pi to get the correct result
    a(xMinusZero) += C::_pi().copySign(y);

    // atan2(-Y, +/-0) = -pi/2
    a(xZero && yNeg) = -C::_pi_2();

    // if both inputs are inf the output is +/- (3)pi/4
    a(xInf && yInf) += C::_pi_4().copySign(x ^ ~y);

    // correct the sign of y if the result is 0
    a(a == V::Zero()) = a.copySign(y);

    // any NaN input will lead to NaN output
    a.setQnan(isnan(y) || isnan(x));

    return a;
}
} // namespace Vc
} // namespace ROOT

#include <Vc/common/undomacros.h>

// instantiate the non-specialized template functions above
template Vc::float_v  Vc::Trigonometric<Vc::Internal::TrigonometricImplementation>::sin(const Vc::float_v  &);
template Vc::sfloat_v Vc::Trigonometric<Vc::Internal::TrigonometricImplementation>::sin(const Vc::sfloat_v &);

template Vc::float_v  Vc::Trigonometric<Vc::Internal::TrigonometricImplementation>::cos(const Vc::float_v  &);
template Vc::sfloat_v Vc::Trigonometric<Vc::Internal::TrigonometricImplementation>::cos(const Vc::sfloat_v &);

template void Vc::Trigonometric<Vc::Internal::TrigonometricImplementation>::sincos(const Vc::float_v  &, Vc::float_v  *, Vc::float_v  *);
template void Vc::Trigonometric<Vc::Internal::TrigonometricImplementation>::sincos(const Vc::sfloat_v &, Vc::sfloat_v *, Vc::sfloat_v *);

template Vc::float_v  Vc::Trigonometric<Vc::Internal::TrigonometricImplementation>::asin(const Vc::float_v  &);
template Vc::sfloat_v Vc::Trigonometric<Vc::Internal::TrigonometricImplementation>::asin(const Vc::sfloat_v &);

template Vc::float_v  Vc::Trigonometric<Vc::Internal::TrigonometricImplementation>::atan(const Vc::float_v  &);
template Vc::sfloat_v Vc::Trigonometric<Vc::Internal::TrigonometricImplementation>::atan(const Vc::sfloat_v &);

template Vc::float_v  Vc::Trigonometric<Vc::Internal::TrigonometricImplementation>::atan2(const Vc::float_v  &, const Vc::float_v  &);
template Vc::sfloat_v Vc::Trigonometric<Vc::Internal::TrigonometricImplementation>::atan2(const Vc::sfloat_v &, const Vc::sfloat_v &);
#endif
