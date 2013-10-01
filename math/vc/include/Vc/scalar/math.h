/*  This file is part of the Vc library.

    Copyright (C) 2009-2012 Matthias Kretz <kretz@kde.org>

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

*/

#ifndef VC_SCALAR_MATH_H
#define VC_SCALAR_MATH_H

#include "macros.h"

namespace ROOT {
namespace Vc
{
namespace Scalar
{

#define VC_MINMAX(V) \
static Vc_ALWAYS_INLINE V min(const V &x, const V &y) { return V(std::min(x.data(), y.data())); } \
static Vc_ALWAYS_INLINE V max(const V &x, const V &y) { return V(std::max(x.data(), y.data())); }
VC_ALL_VECTOR_TYPES(VC_MINMAX)
#undef VC_MINMAX

template<typename T> static Vc_ALWAYS_INLINE Vector<T> sqrt (const Vector<T> &x)
{
    return Vector<T>(std::sqrt(x.data()));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> rsqrt(const Vector<T> &x)
{
    const typename Vector<T>::EntryType one = 1; return Vector<T>(one / std::sqrt(x.data()));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> abs  (const Vector<T> &x)
{
    return Vector<T>(std::abs(x.data()));
}

template<> Vc_ALWAYS_INLINE int_v abs(const int_v &x) { return x < 0 ? -x : x; }
template<> Vc_ALWAYS_INLINE uint_v abs(const uint_v &x) { return x; }
template<> Vc_ALWAYS_INLINE short_v abs(const short_v &x) { return x < 0 ? -x : x; }
template<> Vc_ALWAYS_INLINE ushort_v abs(const ushort_v &x) { return x; }

template<typename T> static Vc_ALWAYS_INLINE void sincos(const Vector<T> &x, Vector<T> *sin, Vector<T> *cos)
{
#if (defined(VC_CLANG) && VC_HAS_BUILTIN(__builtin_sincosf)) || (!defined(VC_CLANG) && defined(__GNUC__) && !defined(_WIN32))
    __builtin_sincosf(x.data(), &sin->data(), &cos->data());
#elif defined(_GNU_SOURCE)
    sincosf(x.data(), &sin->data(), &cos->data());
#else
    sin->data() = std::sin(x.data());
    cos->data() = std::cos(x.data());
#endif
}

template<> Vc_ALWAYS_INLINE void sincos(const Vector<double> &x, Vector<double> *sin, Vector<double> *cos)
{
#if (defined(VC_CLANG) && VC_HAS_BUILTIN(__builtin_sincos)) || (!defined(VC_CLANG) && defined(__GNUC__) && !defined(_WIN32))
    __builtin_sincos(x.data(), &sin->data(), &cos->data());
#elif defined(_GNU_SOURCE)
    ::sincos(x.data(), &sin->data(), &cos->data());
#else
    sin->data() = std::sin(x.data());
    cos->data() = std::cos(x.data());
#endif
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> sin  (const Vector<T> &x)
{
    return Vector<T>(std::sin(x.data()));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> asin (const Vector<T> &x)
{
    return Vector<T>(std::asin(x.data()));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> cos  (const Vector<T> &x)
{
    return Vector<T>(std::cos(x.data()));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> log  (const Vector<T> &x)
{
    return Vector<T>(std::log(x.data()));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> log10(const Vector<T> &x)
{
    return Vector<T>(std::log10(x.data()));
}

#if (defined(_XOPEN_SOURCE) && _XOPEN_SOURCE >= 600) || defined(_ISOC99_SOURCE) || (defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L)
static Vc_ALWAYS_INLINE double_v log2(double_v::AsArg x) { return double_v(::log2 (x.data())); }
static Vc_ALWAYS_INLINE sfloat_v log2(sfloat_v::AsArg x) { return sfloat_v(::log2f(x.data())); }
static Vc_ALWAYS_INLINE  float_v log2( float_v::AsArg x) { return  float_v(::log2f(x.data())); }
#else
namespace {
template<typename T> static _VC_CONSTEXPR T c_ln2() { return Vc_buildFloat(1, 0x317218, -1); } // .693147182464599609375
template<> _VC_CONSTEXPR double c_ln2() { return Vc_buildDouble(1, 0x62E42FEFA39EFull, -1); } // .69314718055994528622676398299518041312694549560546875
}
#define VC_LOG2(V) \
static Vc_ALWAYS_INLINE V log2(const V &x) \
{ \
    return V(std::log(x.data()) / c_ln2<V::EntryType>()); \
}
VC_ALL_FLOAT_VECTOR_TYPES(VC_LOG2)
#undef VC_LOG2
#endif

template<typename T> static Vc_ALWAYS_INLINE Vector<T> exp (const Vector<T> &x)
{
    return Vector<T>(std::exp(x.data()));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> atan (const Vector<T> &x)
{
    return Vector<T>(std::atan( x.data() ));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> atan2(const Vector<T> &x, const Vector<T> &y)
{
    return Vector<T>(std::atan2( x.data(), y.data() ));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> trunc(const Vector<T> &x)
{
#if __cplusplus >= 201103 /*C++11*/
    return std::trunc(x.data());
#else
    return x.data() > 0 ? std::floor(x.data()) : std::ceil(x.data());
#endif
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> floor(const Vector<T> &x)
{
    return Vector<T>(std::floor(x.data()));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> ceil(const Vector<T> &x)
{
    return Vector<T>(std::ceil(x.data()));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> round(const Vector<T> &x)
{
    return x;
}

namespace
{
    template<typename T> bool _realIsEvenHalf(T x) {
        const T two = 2;
        const T half = 0.5;
        const T f = std::floor(x * half) * two;
        return (x - f) == half;
    }
} // namespace
template<> Vc_ALWAYS_INLINE Vector<float>  round(const Vector<float>  &x)
{
    return float_v(std::floor(x.data() + 0.5f) - (_realIsEvenHalf(x.data()) ? 1.f : 0.f));
}

template<> Vc_ALWAYS_INLINE Vector<sfloat> round(const Vector<sfloat> &x)
{
    return sfloat_v(std::floor(x.data() + 0.5f) - (_realIsEvenHalf(x.data()) ? 1.f : 0.f));
}

template<> Vc_ALWAYS_INLINE Vector<double> round(const Vector<double> &x)
{
    return double_v(std::floor(x.data() + 0.5 ) - (_realIsEvenHalf(x.data()) ? 1.  : 0. ));
}

template<typename T> static Vc_ALWAYS_INLINE Vector<T> reciprocal(const Vector<T> &x)
{
    const typename Vector<T>::EntryType one = 1; return Vector<T>(one / x.data());
}

#ifdef isfinite
#undef isfinite
#endif
#ifdef isnan
#undef isnan
#endif
template<typename T> static Vc_ALWAYS_INLINE typename Vector<T>::Mask isfinite(const Vector<T> &x)
{
    return typename Vector<T>::Mask(
#ifdef _MSC_VER
            !!_finite(x.data())
#elif defined(__INTEL_COMPILER)
            ::isfinite(x.data())
#else
            std::isfinite(x.data())
#endif
            );
}

template<typename T> static Vc_ALWAYS_INLINE typename Vector<T>::Mask isnan(const Vector<T> &x)
{
    return typename Vector<T>::Mask(
#ifdef _MSC_VER
            !!_isnan(x.data())
#elif defined(__INTEL_COMPILER)
            ::isnan(x.data())
#else
            std::isnan(x.data())
#endif
            );
}

Vc_ALWAYS_INLINE Vector<float> frexp(Vector<float> x, Vector<int> *e) {
    return float_v(::frexpf(x.data(), &e->data()));
}
Vc_ALWAYS_INLINE Vector<double> frexp(Vector<double> x, Vector<int> *e) {
    return double_v(::frexp(x.data(), &e->data()));
}
Vc_ALWAYS_INLINE sfloat_v frexp(sfloat_v x, short_v *e) {
    int ee;
    const float r = ::frexpf(x.data(), &ee);
    e->data() = ee;
    return sfloat_v(r);
}

Vc_ALWAYS_INLINE Vector<float> ldexp(Vector<float> x, Vector<int> e) {
    return float_v(::ldexpf(x.data(), e.data()));
}
Vc_ALWAYS_INLINE Vector<double> ldexp(Vector<double> x, Vector<int> e) {
    return double_v(::ldexp(x.data(), e.data()));
}
Vc_ALWAYS_INLINE sfloat_v ldexp(sfloat_v x, short_v e) {
    return sfloat_v(::ldexpf(x.data(), e.data()));
}

} // namespace Scalar
} // namespace Vc
} // namespace ROOT

#include "undomacros.h"

#endif // VC_SCALAR_MATH_H
