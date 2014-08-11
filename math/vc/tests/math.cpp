/*  This file is part of the Vc library. {{{

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

}}}*/
/*includes {{{*/
#include "unittest.h"
#include <iostream>
#include "vectormemoryhelper.h"
#include "const.h"
#include <cmath>
#include <algorithm>
#include <Vc/common/macros.h>
/*}}}*/
using namespace Vc;
/*fix isfinite and isnan{{{*/
#ifdef isfinite
#undef isfinite
#endif
#ifdef isnan
#undef isnan
#endif
/*}}}*/
template<typename T> struct SincosReference/*{{{*/
{
    T x, s, c;
};
template<typename T> struct Reference
{
    T x, ref;
};

template<typename T> struct Array
{
    size_t size;
    T *data;
    Array() : size(0), data(0) {}
};
template<typename T> struct StaticDeleter
{
    T *ptr;
    StaticDeleter(T *p) : ptr(p) {}
    ~StaticDeleter() { delete[] ptr; }
};

enum Function {
    Sincos, Atan, Asin, Acos, Log, Log2, Log10
};
template<typename T, Function F> static inline const char *filename();
template<> inline const char *filename<float , Sincos>() { return "reference-sincos-sp.dat"; }
template<> inline const char *filename<double, Sincos>() { return "reference-sincos-dp.dat"; }
template<> inline const char *filename<float , Atan  >() { return "reference-atan-sp.dat"; }
template<> inline const char *filename<double, Atan  >() { return "reference-atan-dp.dat"; }
template<> inline const char *filename<float , Asin  >() { return "reference-asin-sp.dat"; }
template<> inline const char *filename<double, Asin  >() { return "reference-asin-dp.dat"; }
// template<> inline const char *filename<float , Acos  >() { return "reference-acos-sp.dat"; }
// template<> inline const char *filename<double, Acos  >() { return "reference-acos-dp.dat"; }
template<> inline const char *filename<float , Log   >() { return "reference-ln-sp.dat"; }
template<> inline const char *filename<double, Log   >() { return "reference-ln-dp.dat"; }
template<> inline const char *filename<float , Log2  >() { return "reference-log2-sp.dat"; }
template<> inline const char *filename<double, Log2  >() { return "reference-log2-dp.dat"; }
template<> inline const char *filename<float , Log10 >() { return "reference-log10-sp.dat"; }
template<> inline const char *filename<double, Log10 >() { return "reference-log10-dp.dat"; }

template<typename T>
static Array<SincosReference<T> > sincosReference()
{
    static Array<SincosReference<T> > data;
    if (data.data == 0) {
        FILE *file = fopen(filename<T, Sincos>(), "rb");
        if (file) {
            fseek(file, 0, SEEK_END);
            const size_t size = ftell(file) / sizeof(SincosReference<T>);
            rewind(file);
            data.data = new SincosReference<T>[size];
            static StaticDeleter<SincosReference<T> > _cleanup(data.data);
            data.size = fread(data.data, sizeof(SincosReference<T>), size, file);
            fclose(file);
        } else {
            FAIL() << "the reference data " << filename<T, Sincos>() << " does not exist in the current working directory.";
        }
    }
    return data;
}

template<typename T, Function Fun>
static Array<Reference<T> > referenceData()
{
    static Array<Reference<T> > data;
    if (data.data == 0) {
        FILE *file = fopen(filename<T, Fun>(), "rb");
        if (file) {
            fseek(file, 0, SEEK_END);
            const size_t size = ftell(file) / sizeof(Reference<T>);
            rewind(file);
            data.data = new Reference<T>[size];
            static StaticDeleter<Reference<T> > _cleanup(data.data);
            data.size = fread(data.data, sizeof(Reference<T>), size, file);
            fclose(file);
        } else {
            FAIL() << "the reference data " << filename<T, Fun>() << " does not exist in the current working directory.";
        }
    }
    return data;
}/*}}}*/

template<typename T> struct Denormals { static T *data; };/*{{{*/
template<> float  *Denormals<float >::data = 0;
template<> double *Denormals<double>::data = 0;
enum {
    NDenormals = 64
};
/*}}}*/
template<typename V> V apply_v(VC_ALIGNED_PARAMETER(V) x, typename V::EntryType (func)(typename V::EntryType))/*{{{*/
{
    V r;
    for (size_t i = 0; i < V::Size; ++i) {
        r[i] = func(x[i]);
    }
    return r;
}
/*}}}*/
template<typename Vec> void testAbs()/*{{{*/
{
    for (int i = 0; i < 0x7fff; ++i) {
        Vec a(i);
        Vec b(-i);
        COMPARE(a, Vc::abs(a));
        COMPARE(a, Vc::abs(b));
    }
}
/*}}}*/
static inline float my_trunc(float x)/*{{{*/
{
#if __cplusplus >= 201103 /*C++11*/
    return std::trunc(x);
#elif defined(_ISOC99_SOURCE)
    return truncf(x);
#else
    return x > 0 ? std::floor(x) : std::ceil(x);
#endif
}

static inline double my_trunc(double x)
{
#if __cplusplus >= 201103 /*C++11*/
    return std::trunc(x);
#elif defined(_ISOC99_SOURCE)
    return trunc(x);
#else
    return x > 0 ? std::floor(x) : std::ceil(x);
#endif
}
/*}}}*/
template<typename V> void testTrunc()/*{{{*/
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    for (size_t i = 0; i < 100000 / V::Size; ++i) {
        V x = (V::Random() - T(0.5)) * T(100);
        V reference = apply_v(x, my_trunc);
        COMPARE(Vc::trunc(x), reference) << ", x = " << x << ", i = " << i;
    }
    V x = static_cast<V>(I::IndexesFromZero());
    V reference = apply_v(x, my_trunc);
    COMPARE(Vc::trunc(x), reference) << ", x = " << x;
}
/*}}}*/
template<typename V> void testFloor()/*{{{*/
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    for (size_t i = 0; i < 100000 / V::Size; ++i) {
        V x = (V::Random() - T(0.5)) * T(100);
        V reference = apply_v(x, std::floor);
        COMPARE(Vc::floor(x), reference) << ", x = " << x << ", i = " << i;
    }
    V x = static_cast<V>(I::IndexesFromZero());
    V reference = apply_v(x, std::floor);
    COMPARE(Vc::floor(x), reference) << ", x = " << x;
}
/*}}}*/
template<typename V> void testCeil()/*{{{*/
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    for (size_t i = 0; i < 100000 / V::Size; ++i) {
        V x = (V::Random() - T(0.5)) * T(100);
        V reference = apply_v(x, std::ceil);
        COMPARE(Vc::ceil(x), reference) << ", x = " << x << ", i = " << i;
    }
    V x = static_cast<V>(I::IndexesFromZero());
    V reference = apply_v(x, std::ceil);
    COMPARE(Vc::ceil(x), reference) << ", x = " << x;
}
/*}}}*/
template<typename V> void testExp()/*{{{*/
{
    setFuzzyness<float>(1);
    setFuzzyness<double>(2);
    typedef typename V::EntryType T;
    for (size_t i = 0; i < 100000 / V::Size; ++i) {
        V x = (V::Random() - T(0.5)) * T(20);
        V reference = apply_v(x, std::exp);
        FUZZY_COMPARE(Vc::exp(x), reference) << ", x = " << x << ", i = " << i;
    }
    COMPARE(Vc::exp(V::Zero()), V::One());
}
/*}}}*/
template<typename V> void testLog()/*{{{*/
{
    setFuzzyness<float>(1);
    typedef typename V::EntryType T;
    Array<Reference<T> > reference = referenceData<T, Log>();
    for (size_t i = 0; i + V::Size - 1 < reference.size; i += V::Size) {
        V x, ref;
        for (int j = 0; j < V::Size; ++j) {
            x[j] = reference.data[i + j].x;
            ref[j] = reference.data[i + j].ref;
        }
        FUZZY_COMPARE(Vc::log(x), ref) << " x = " << x << ", i = " << i;
    }

    COMPARE(Vc::log(V::Zero()), V(std::log(T(0))));
    for (int i = 0; i < NDenormals; i += V::Size) {
        V x(&Denormals<T>::data[i]);
        V ref = apply_v(x, std::log);
        FUZZY_COMPARE(Vc::log(x), ref) << ", x = " << x << ", i = " << i;
    }
}
/*}}}*/
#if (defined(_XOPEN_SOURCE) && _XOPEN_SOURCE >= 600) || defined(_ISOC99_SOURCE) || (defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L)
static inline float my_log2(float x) { return ::log2f(x); }
/* I need to make sure whether the log2 that I compare against is really precise to <0.5ulp. At
 * least I get different results when I use "double log2(double)", which is somewhat unexpected.
 * Well, conversion from double to float goes via truncation, so if the most significant truncated
 * mantissa bit is set the resulting float is incorrect by 1 ulp

static inline float my_log2(float x) { return ::log2(static_cast<double>(x)); }
static inline float my_log2(float x) {
    double tmp = ::log2(static_cast<double>(x));
    int e;
    frexp(tmp, &e); // frexp(0.5) -> e = 0
    return tmp + ldexp(tmp < 0 ? -0.5 : 0.5, e - 24);
}
 */
static inline double my_log2(double x) { return ::log2(x); }
#else
static inline float my_log2(float x) { return ::logf(x) / Vc::Math<float>::ln2(); }
static inline double my_log2(double x) { return ::log(x) / Vc::Math<double>::ln2(); }
#endif
/*}}}*/
template<typename V> void testLog2()/*{{{*/
{
#if defined(VC_LOG_ILP) || defined(VC_LOG_ILP2)
    setFuzzyness<float>(3);
#else
    setFuzzyness<float>(1);
#endif
#if (defined(VC_MSVC) || defined(__APPLE__)) && defined(VC_IMPL_Scalar)
    setFuzzyness<double>(2);
#else
    setFuzzyness<double>(1);
#endif
    typedef typename V::EntryType T;
    Array<Reference<T> > reference = referenceData<T, Log2>();
    for (size_t i = 0; i + V::Size - 1 < reference.size; i += V::Size) {
        V x, ref;
        for (int j = 0; j < V::Size; ++j) {
            x[j] = reference.data[i + j].x;
            ref[j] = reference.data[i + j].ref;
        }
        FUZZY_COMPARE(Vc::log2(x), ref) << " x = " << x << ", i = " << i;
    }

    COMPARE(Vc::log2(V::Zero()), V(my_log2(T(0))));
    for (int i = 0; i < NDenormals; i += V::Size) {
        V x(&Denormals<T>::data[i]);
        V ref = apply_v(x, my_log2);
        FUZZY_COMPARE(Vc::log2(x), ref) << ", x = " << x << ", i = " << i;
    }
}
/*}}}*/
template<typename V> void testLog10()/*{{{*/
{
    setFuzzyness<float>(2);
    setFuzzyness<double>(2);
    typedef typename V::EntryType T;
    Array<Reference<T> > reference = referenceData<T, Log10>();
    for (size_t i = 0; i + V::Size - 1 < reference.size; i += V::Size) {
        V x, ref;
        for (int j = 0; j < V::Size; ++j) {
            x[j] = reference.data[i + j].x;
            ref[j] = reference.data[i + j].ref;
        }
        FUZZY_COMPARE(Vc::log10(x), ref) << " x = " << x << ", i = " << i;
    }

    COMPARE(Vc::log10(V::Zero()), V(std::log10(T(0))));
    for (int i = 0; i < NDenormals; i += V::Size) {
        V x(&Denormals<T>::data[i]);
        V ref = apply_v(x, std::log10);
        FUZZY_COMPARE(Vc::log10(x), ref) << ", x = " << x << ", i = " << i;
    }
}
/*}}}*/
template<typename Vec> void testMax()/*{{{*/
{
    typedef typename Vec::EntryType T;
    VectorMemoryHelper<Vec> mem(3);
    T *data = mem;
    for (int i = 0; i < Vec::Size; ++i) {
        data[i] = i;
        data[i + Vec::Size] = Vec::Size + 1 - i;
        data[i + 2 * Vec::Size] = std::max(data[i], data[i + Vec::Size]);
    }
    Vec a(&data[0]);
    Vec b(&data[Vec::Size]);
    Vec c(&data[2 * Vec::Size]);

    COMPARE(Vc::max(a, b), c);
}
/*}}}*/
    /*{{{*/
#define FillHelperMemory(code) \
    typename V::Memory data; \
    typename V::Memory reference; \
    for (int ii = 0; ii < V::Size; ++ii) { \
        const T i = static_cast<T>(ii); \
        data[ii] = i; \
        reference[ii] = code; \
    } do {} while (false)
/*}}}*/
template<typename V> void testSqrt()/*{{{*/
{
    typedef typename V::EntryType T;
    FillHelperMemory(std::sqrt(i));
    V a(data);
    V b(reference);

    FUZZY_COMPARE(Vc::sqrt(a), b);
}
/*}}}*/
template<typename V> void testRSqrt()/*{{{*/
{
    typedef typename V::EntryType T;
    for (size_t i = 0; i < 1024 / V::Size; ++i) {
        const V x = V::Random() * T(1000);
        // RSQRTPS is documented as having a relative error <= 1.5 * 2^-12
        VERIFY(Vc::abs(Vc::rsqrt(x) * Vc::sqrt(x) - V::One()) < static_cast<T>(std::ldexp(1.5, -12)));
    }
}
/*}}}*/
template<typename V> void testSincos()/*{{{*/
{
    typedef typename V::EntryType T;
    setFuzzyness<float>(2);
    setFuzzyness<double>(1e7);
    Array<SincosReference<T> > reference = sincosReference<T>();
    for (size_t i = 0; i + V::Size - 1 < reference.size; i += V::Size) {
        V x, sref, cref;
        for (int j = 0; j < V::Size; ++j) {
            x[j] = reference.data[i + j].x;
            sref[j] = reference.data[i + j].s;
            cref[j] = reference.data[i + j].c;
        }
        V sin, cos;
        Vc::sincos(x, &sin, &cos);
        FUZZY_COMPARE(sin, sref) << " x = " << x << ", i = " << i;
        FUZZY_COMPARE(cos, cref) << " x = " << x << ", i = " << i;
        Vc::sincos(-x, &sin, &cos);
        FUZZY_COMPARE(sin, -sref) << " x = " << -x << ", i = " << i;
        FUZZY_COMPARE(cos, cref) << " x = " << -x << ", i = " << i;
    }
}
/*}}}*/
template<typename V> void testSin()/*{{{*/
{
    typedef typename V::EntryType T;
    setFuzzyness<float>(2);
    setFuzzyness<double>(1e7);
    Array<SincosReference<T> > reference = sincosReference<T>();
    for (size_t i = 0; i + V::Size - 1 < reference.size; i += V::Size) {
        V x, sref;
        for (int j = 0; j < V::Size; ++j) {
            x[j] = reference.data[i + j].x;
            sref[j] = reference.data[i + j].s;
        }
        FUZZY_COMPARE(Vc::sin(x), sref) << " x = " << x << ", i = " << i;
        FUZZY_COMPARE(Vc::sin(-x), -sref) << " x = " << x << ", i = " << i;
    }
}
/*}}}*/
template<typename V> void testCos()/*{{{*/
{
    typedef typename V::EntryType T;
    setFuzzyness<float>(2);
    setFuzzyness<double>(1e7);
    Array<SincosReference<T> > reference = sincosReference<T>();
    for (size_t i = 0; i + V::Size - 1 < reference.size; i += V::Size) {
        V x, cref;
        for (int j = 0; j < V::Size; ++j) {
            x[j] = reference.data[i + j].x;
            cref[j] = reference.data[i + j].c;
        }
        FUZZY_COMPARE(Vc::cos(x), cref) << " x = " << x << ", i = " << i;
        FUZZY_COMPARE(Vc::cos(-x), cref) << " x = " << x << ", i = " << i;
    }
}
/*}}}*/
template<typename V> void testAsin()/*{{{*/
{
    typedef typename V::EntryType T;
    setFuzzyness<float>(2);
    setFuzzyness<double>(36);
    Array<Reference<T> > reference = referenceData<T, Asin>();
    for (size_t i = 0; i + V::Size - 1 < reference.size; i += V::Size) {
        V x, ref;
        for (int j = 0; j < V::Size; ++j) {
            x[j] = reference.data[i + j].x;
            ref[j] = reference.data[i + j].ref;
        }
        FUZZY_COMPARE(Vc::asin(x), ref) << " x = " << x << ", i = " << i;
        FUZZY_COMPARE(Vc::asin(-x), -ref) << " -x = " << -x << ", i = " << i;
    }
}
/*}}}*/
const union {
    unsigned int hex;
    float value;
} INF = { 0x7f800000 };

#if defined(__APPLE__) && defined(VC_IMPL_Scalar)
#define ATAN_COMPARE FUZZY_COMPARE
#else
#define ATAN_COMPARE COMPARE
#endif

template<typename V> void testAtan()/*{{{*/
{
    typedef typename V::EntryType T;
    setFuzzyness<float>(3);
    setFuzzyness<double>(2);

    {
        const V Pi_2 = T(Vc_buildDouble(1, 0x921fb54442d18ull,  0));
        V nan; nan.setQnan();
        const V inf = T(INF.value);

        VERIFY(Vc::isnan(Vc::atan(nan)));
        ATAN_COMPARE(Vc::atan(+inf), +Pi_2);
#ifdef VC_MSVC
#pragma warning(suppress: 4756) // overflow in constant arithmetic
#endif
        ATAN_COMPARE(Vc::atan(-inf), -Pi_2);
    }

    Array<Reference<T> > reference = referenceData<T, Atan>();
    for (size_t i = 0; i + V::Size - 1 < reference.size; i += V::Size) {
        V x, ref;
        for (int j = 0; j < V::Size; ++j) {
            x[j] = reference.data[i + j].x;
            ref[j] = reference.data[i + j].ref;
        }
        FUZZY_COMPARE(Vc::atan(x), ref) << " x = " << x << ", i = " << i;
        FUZZY_COMPARE(Vc::atan(-x), -ref) << " -x = " << -x << ", i = " << i;
    }
}
/*}}}*/
template<typename V> void testAtan2()/*{{{*/
{
    typedef typename V::EntryType T;
    setFuzzyness<float>(3);
    setFuzzyness<double>(2);

    {
        const V Pi   = T(Vc_buildDouble(1, 0x921fb54442d18ull,  1));
        const V Pi_2 = T(Vc_buildDouble(1, 0x921fb54442d18ull,  0));
        V nan; nan.setQnan();
        const V inf = T(INF.value);

        // If y is +0 (-0) and x is less than 0, +pi (-pi) is returned.
        ATAN_COMPARE(Vc::atan2(V(T(+0.)), V(T(-3.))), +Pi);
        ATAN_COMPARE(Vc::atan2(V(T(-0.)), V(T(-3.))), -Pi);
        // If y is +0 (-0) and x is greater than 0, +0 (-0) is returned.
        COMPARE(Vc::atan2(V(T(+0.)), V(T(+3.))), V(T(+0.)));
        VERIFY(!Vc::atan2(V(T(+0.)), V(T(+3.))).isNegative());
        COMPARE(Vc::atan2(V(T(-0.)), V(T(+3.))), V(T(-0.)));
        VERIFY (Vc::atan2(V(T(-0.)), V(T(+3.))).isNegative());
        // If y is less than 0 and x is +0 or -0, -pi/2 is returned.
        COMPARE(Vc::atan2(V(T(-3.)), V(T(+0.))), -Pi_2);
        COMPARE(Vc::atan2(V(T(-3.)), V(T(-0.))), -Pi_2);
        // If y is greater than 0 and x is +0 or -0, pi/2 is returned.
        COMPARE(Vc::atan2(V(T(+3.)), V(T(+0.))), +Pi_2);
        COMPARE(Vc::atan2(V(T(+3.)), V(T(-0.))), +Pi_2);
        // If either x or y is NaN, a NaN is returned.
        VERIFY(Vc::isnan(Vc::atan2(nan, V(T(3.)))));
        VERIFY(Vc::isnan(Vc::atan2(V(T(3.)), nan)));
        VERIFY(Vc::isnan(Vc::atan2(nan, nan)));
        // If y is +0 (-0) and x is -0, +pi (-pi) is returned.
        ATAN_COMPARE(Vc::atan2(V(T(+0.)), V(T(-0.))), +Pi);
        ATAN_COMPARE(Vc::atan2(V(T(-0.)), V(T(-0.))), -Pi);
        // If y is +0 (-0) and x is +0, +0 (-0) is returned.
        COMPARE(Vc::atan2(V(T(+0.)), V(T(+0.))), V(T(+0.)));
        COMPARE(Vc::atan2(V(T(-0.)), V(T(+0.))), V(T(-0.)));
        VERIFY(!Vc::atan2(V(T(+0.)), V(T(+0.))).isNegative());
        VERIFY( Vc::atan2(V(T(-0.)), V(T(+0.))).isNegative());
        // If y is a finite value greater (less) than 0, and x is negative infinity, +pi (-pi) is returned.
        ATAN_COMPARE(Vc::atan2(V(T(+1.)), -inf), +Pi);
        ATAN_COMPARE(Vc::atan2(V(T(-1.)), -inf), -Pi);
        // If y is a finite value greater (less) than 0, and x is positive infinity, +0 (-0) is returned.
        COMPARE(Vc::atan2(V(T(+3.)), +inf), V(T(+0.)));
        VERIFY(!Vc::atan2(V(T(+3.)), +inf).isNegative());
        COMPARE(Vc::atan2(V(T(-3.)), +inf), V(T(-0.)));
        VERIFY (Vc::atan2(V(T(-3.)), +inf).isNegative());
        // If y is positive infinity (negative infinity), and x is finite, pi/2 (-pi/2) is returned.
        COMPARE(Vc::atan2(+inf, V(T(+3.))), +Pi_2);
        COMPARE(Vc::atan2(-inf, V(T(+3.))), -Pi_2);
        COMPARE(Vc::atan2(+inf, V(T(-3.))), +Pi_2);
        COMPARE(Vc::atan2(-inf, V(T(-3.))), -Pi_2);
#ifndef _WIN32 // the Microsoft implementation of atan2 fails this test
        const V Pi_4 = T(Vc_buildDouble(1, 0x921fb54442d18ull, -1));
        // If y is positive infinity (negative infinity) and x is negative infinity, +3*pi/4 (-3*pi/4) is returned.
        COMPARE(Vc::atan2(+inf, -inf), T(+3.) * Pi_4);
        COMPARE(Vc::atan2(-inf, -inf), T(-3.) * Pi_4);
        // If y is positive infinity (negative infinity) and x is positive infinity, +pi/4 (-pi/4) is returned.
        COMPARE(Vc::atan2(+inf, +inf), +Pi_4);
        COMPARE(Vc::atan2(-inf, +inf), -Pi_4);
#endif
    }

    for (int xoffset = -100; xoffset < 54613; xoffset += 47 * V::Size) {
        for (int yoffset = -100; yoffset < 54613; yoffset += 47 * V::Size) {
            FillHelperMemory(std::atan2((i + xoffset) * T(0.15), (i + yoffset) * T(0.15)));
            const V a(data);
            const V b(reference);

            const V x = (a + xoffset) * T(0.15);
            const V y = (a + yoffset) * T(0.15);
            FUZZY_COMPARE(Vc::atan2(x, y), b) << ", x = " << x << ", y = " << y;
        }
    }
}
/*}}}*/
template<typename Vec> void testReciprocal()/*{{{*/
{
    typedef typename Vec::EntryType T;
    setFuzzyness<float>(1.258295e+07);
    setFuzzyness<double>(0);
    const T one = 1;
    for (int offset = -1000; offset < 1000; offset += 10) {
        const T scale = T(0.1);
        typename Vec::Memory data;
        typename Vec::Memory reference;
        for (int ii = 0; ii < Vec::Size; ++ii) {
            const T i = static_cast<T>(ii);
            data[ii] = i;
            T tmp = (i + offset) * scale;
            reference[ii] = one / tmp;
        }
        Vec a(data);
        Vec b(reference);

        FUZZY_COMPARE(Vc::reciprocal((a + offset) * scale), b);
    }
}
/*}}}*/
template<typename V> void isNegative()/*{{{*/
{
    typedef typename V::EntryType T;
    VERIFY(V::One().isNegative().isEmpty());
    VERIFY(V::Zero().isNegative().isEmpty());
    VERIFY((-V::One()).isNegative().isFull());
    VERIFY(V(T(-0.)).isNegative().isFull());
}
/*}}}*/
template<typename Vec> void testInf()/*{{{*/
{
    typedef typename Vec::EntryType T;
    const T one = 1;
    const Vec zero(Zero);
    VERIFY(Vc::isfinite(zero));
    VERIFY(Vc::isfinite(Vec(one)));
    VERIFY(!Vc::isfinite(one / zero));
}
/*}}}*/
template<typename Vec> void testNaN()/*{{{*/
{
    typedef typename Vec::EntryType T;
    typedef typename Vec::IndexType I;
    typedef typename Vec::Mask M;
    const T one = 1;
    const Vec zero(Zero);
    VERIFY(!Vc::isnan(zero));
    VERIFY(!Vc::isnan(Vec(one)));
    const Vec inf = one / zero;
    VERIFY(Vc::isnan(Vec(inf * zero)));
    Vec nan = Vec::Zero();
    const M mask(I::IndexesFromZero() == I::Zero());
    nan.setQnan(mask);
    COMPARE(Vc::isnan(nan), mask);
    nan.setQnan();
    VERIFY(Vc::isnan(nan));
}
/*}}}*/
template<typename Vec> void testRound()/*{{{*/
{
    typedef typename Vec::EntryType T;
    enum {
        Count = (16 + Vec::Size) / Vec::Size
    };
    VectorMemoryHelper<Vec> mem1(Count);
    VectorMemoryHelper<Vec> mem2(Count);
    T *data = mem1;
    T *reference = mem2;
    for (int i = 0; i < Count * Vec::Size; ++i) {
        data[i] = i * 0.25 - 2.0;
        reference[i] = std::floor(i * 0.25 - 2.0 + 0.5);
        if (i % 8 == 2) {
            reference[i] -= 1.;
        }
        //std::cout << reference[i] << " ";
    }
    //std::cout << std::endl;
    for (int i = 0; i < Count; ++i) {
        const Vec a(&data[i * Vec::Size]);
        const Vec ref(&reference[i * Vec::Size]);
        //std::cout << a << ref << std::endl;
        COMPARE(Vc::round(a), ref);
    }
}
/*}}}*/
template<typename Vec> void testReduceMin()/*{{{*/
{
    typedef typename Vec::EntryType T;
    const T one = 1;
    VectorMemoryHelper<Vec> mem(Vec::Size);
    T *data = mem;
    for (int i = 0; i < Vec::Size * Vec::Size; ++i) {
        data[i] = i % (Vec::Size + 1) + one;
    }
    for (int i = 0; i < Vec::Size; ++i, data += Vec::Size) {
        const Vec a(&data[0]);
        //std::cout << a << std::endl;
        COMPARE(a.min(), one);
    }
}
/*}}}*/
template<typename Vec> void testReduceMax()/*{{{*/
{
    typedef typename Vec::EntryType T;
    const T max = Vec::Size + 1;
    VectorMemoryHelper<Vec> mem(Vec::Size);
    T *data = mem;
    for (int i = 0; i < Vec::Size * Vec::Size; ++i) {
        data[i] = (i + Vec::Size) % (Vec::Size + 1) + 1;
    }
    for (int i = 0; i < Vec::Size; ++i, data += Vec::Size) {
        const Vec a(&data[0]);
        //std::cout << a << std::endl;
        COMPARE(a.max(), max);
    }
}
/*}}}*/
template<typename Vec> void testReduceProduct()/*{{{*/
{
    enum {
        Max = Vec::Size > 8 ? Vec::Size / 2 : Vec::Size
    };
    typedef typename Vec::EntryType T;
    int _product = 1;
    for (int i = 1; i < Vec::Size; ++i) {
        _product *= (i % Max) + 1;
    }
    const T product = _product;
    VectorMemoryHelper<Vec> mem(Vec::Size);
    T *data = mem;
    for (int i = 0; i < Vec::Size * Vec::Size; ++i) {
        data[i] = ((i + (i / Vec::Size)) % Max) + 1;
    }
    for (int i = 0; i < Vec::Size; ++i, data += Vec::Size) {
        const Vec a(&data[0]);
        //std::cout << a << std::endl;
        COMPARE(a.product(), product);
    }
}
/*}}}*/
template<typename Vec> void testReduceSum()/*{{{*/
{
    typedef typename Vec::EntryType T;
    int _sum = 1;
    for (int i = 2; i <= Vec::Size; ++i) {
        _sum += i;
    }
    const T sum = _sum;
    VectorMemoryHelper<Vec> mem(Vec::Size);
    T *data = mem;
    for (int i = 0; i < Vec::Size * Vec::Size; ++i) {
        data[i] = (i + i / Vec::Size) % Vec::Size + 1;
    }
    for (int i = 0; i < Vec::Size; ++i, data += Vec::Size) {
        const Vec a(&data[0]);
        //std::cout << a << std::endl;
        COMPARE(a.sum(), sum);
    }
}
/*}}}*/
template<typename V> void testExponent()/*{{{*/
{
    typedef typename V::EntryType T;
    Vc::Memory<V, 32> input;
    Vc::Memory<V, 32> expected;
    input[ 0] = T(0.25); expected[ 0] = T(-2);
    input[ 1] = T(   1); expected[ 1] = T( 0);
    input[ 2] = T(   2); expected[ 2] = T( 1);
    input[ 3] = T(   3); expected[ 3] = T( 1);
    input[ 4] = T(   4); expected[ 4] = T( 2);
    input[ 5] = T( 0.5); expected[ 5] = T(-1);
    input[ 6] = T(   6); expected[ 6] = T( 2);
    input[ 7] = T(   7); expected[ 7] = T( 2);
    input[ 8] = T(   8); expected[ 8] = T( 3);
    input[ 9] = T(   9); expected[ 9] = T( 3);
    input[10] = T(  10); expected[10] = T( 3);
    input[11] = T(  11); expected[11] = T( 3);
    input[12] = T(  12); expected[12] = T( 3);
    input[13] = T(  13); expected[13] = T( 3);
    input[14] = T(  14); expected[14] = T( 3);
    input[15] = T(  15); expected[15] = T( 3);
    input[16] = T(  16); expected[16] = T( 4);
    input[17] = T(  17); expected[17] = T( 4);
    input[18] = T(  18); expected[18] = T( 4);
    input[19] = T(  19); expected[19] = T( 4);
    input[20] = T(  20); expected[20] = T( 4);
    input[21] = T(  21); expected[21] = T( 4);
    input[22] = T(  22); expected[22] = T( 4);
    input[23] = T(  23); expected[23] = T( 4);
    input[24] = T(  24); expected[24] = T( 4);
    input[25] = T(  25); expected[25] = T( 4);
    input[26] = T(  26); expected[26] = T( 4);
    input[27] = T(  27); expected[27] = T( 4);
    input[28] = T(  28); expected[28] = T( 4);
    input[29] = T(  29); expected[29] = T( 4);
    input[30] = T(  32); expected[30] = T( 5);
    input[31] = T(  31); expected[31] = T( 4);
    for (size_t i = 0; i < input.vectorsCount(); ++i) {
        COMPARE(V(input.vector(i)).exponent(), V(expected.vector(i)));
    }
}
/*}}}*/
template<typename T> struct _ExponentVector { typedef int_v Type; };
template<> struct _ExponentVector<sfloat_v> { typedef short_v Type; };

template<typename V> void testFrexp()/*{{{*/
{
    typedef typename V::EntryType T;
    typedef typename _ExponentVector<V>::Type ExpV;
    Vc::Memory<V, 32> input;
    Vc::Memory<V, 32> expectedFraction;
    Vc::Memory<ExpV, 32> expectedExponent;
    input[ 0] = T(0.25); expectedFraction[ 0] = T(.5     ); expectedExponent[ 0] = -1;
    input[ 1] = T(   1); expectedFraction[ 1] = T(.5     ); expectedExponent[ 1] =  1;
    input[ 2] = T(   0); expectedFraction[ 2] = T(0.     ); expectedExponent[ 2] =  0;
    input[ 3] = T(   3); expectedFraction[ 3] = T(.75    ); expectedExponent[ 3] =  2;
    input[ 4] = T(   4); expectedFraction[ 4] = T(.5     ); expectedExponent[ 4] =  3;
    input[ 5] = T( 0.5); expectedFraction[ 5] = T(.5     ); expectedExponent[ 5] =  0;
    input[ 6] = T(   6); expectedFraction[ 6] = T( 6./8. ); expectedExponent[ 6] =  3;
    input[ 7] = T(   7); expectedFraction[ 7] = T( 7./8. ); expectedExponent[ 7] =  3;
    input[ 8] = T(   8); expectedFraction[ 8] = T( 8./16.); expectedExponent[ 8] =  4;
    input[ 9] = T(   9); expectedFraction[ 9] = T( 9./16.); expectedExponent[ 9] =  4;
    input[10] = T(  10); expectedFraction[10] = T(10./16.); expectedExponent[10] =  4;
    input[11] = T(  11); expectedFraction[11] = T(11./16.); expectedExponent[11] =  4;
    input[12] = T(  12); expectedFraction[12] = T(12./16.); expectedExponent[12] =  4;
    input[13] = T(  13); expectedFraction[13] = T(13./16.); expectedExponent[13] =  4;
    input[14] = T(  14); expectedFraction[14] = T(14./16.); expectedExponent[14] =  4;
    input[15] = T(  15); expectedFraction[15] = T(15./16.); expectedExponent[15] =  4;
    input[16] = T(  16); expectedFraction[16] = T(16./32.); expectedExponent[16] =  5;
    input[17] = T(  17); expectedFraction[17] = T(17./32.); expectedExponent[17] =  5;
    input[18] = T(  18); expectedFraction[18] = T(18./32.); expectedExponent[18] =  5;
    input[19] = T(  19); expectedFraction[19] = T(19./32.); expectedExponent[19] =  5;
    input[20] = T(  20); expectedFraction[20] = T(20./32.); expectedExponent[20] =  5;
    input[21] = T(  21); expectedFraction[21] = T(21./32.); expectedExponent[21] =  5;
    input[22] = T(  22); expectedFraction[22] = T(22./32.); expectedExponent[22] =  5;
    input[23] = T(  23); expectedFraction[23] = T(23./32.); expectedExponent[23] =  5;
    input[24] = T(  24); expectedFraction[24] = T(24./32.); expectedExponent[24] =  5;
    input[25] = T(  25); expectedFraction[25] = T(25./32.); expectedExponent[25] =  5;
    input[26] = T(  26); expectedFraction[26] = T(26./32.); expectedExponent[26] =  5;
    input[27] = T(  27); expectedFraction[27] = T(27./32.); expectedExponent[27] =  5;
    input[28] = T(  28); expectedFraction[28] = T(28./32.); expectedExponent[28] =  5;
    input[29] = T(  29); expectedFraction[29] = T(29./32.); expectedExponent[29] =  5;
    input[30] = T(  32); expectedFraction[30] = T(32./64.); expectedExponent[30] =  6;
    input[31] = T(  31); expectedFraction[31] = T(31./32.); expectedExponent[31] =  5;
    for (size_t i = 0; i < input.vectorsCount(); ++i) {
        const V v = input.vector(i);
        ExpV exp;
        COMPARE(frexp(v, &exp), V(expectedFraction.vector(i)));
        if (V::Size * 2 == ExpV::Size) {
            for (size_t j = 0; j < V::Size; ++j) {
                COMPARE(exp[j * 2], expectedExponent[i * V::Size + j]);
            }
        } else {
            COMPARE(exp, ExpV(expectedExponent.vector(i)));
        }
    }
}
/*}}}*/
template<typename V> void testLdexp()/*{{{*/
{
    typedef typename V::EntryType T;
    typedef typename _ExponentVector<V>::Type ExpV;
    for (size_t i = 0; i < 1024 / V::Size; ++i) {
        const V v = (V::Random() - T(0.5)) * T(1000);
        ExpV e;
        const V m = frexp(v, &e);
        COMPARE(ldexp(m, e), v) << ", m = " << m << ", e = " << e;
    }
}
/*}}}*/
#include "ulp.h"
template<typename V> void testUlpDiff()/*{{{*/
{
    typedef typename V::EntryType T;

    COMPARE(ulpDiffToReference(V::Zero(), V::Zero()), V::Zero());
    COMPARE(ulpDiffToReference(std::numeric_limits<V>::min(), V::Zero()), V::One());
    COMPARE(ulpDiffToReference(V::Zero(), std::numeric_limits<V>::min()), V::One());
    for (size_t count = 0; count < 1024 / V::Size; ++count) {
        const V base = (V::Random() - T(0.5)) * T(1000);
        typename _Ulp_ExponentVector<V>::Type exp;
        Vc::frexp(base, &exp);
        const V eps = ldexp(V(std::numeric_limits<T>::epsilon()), exp - 1);
        //std::cout << base << ", " << exp << ", " << eps << std::endl;
        for (int i = -10000; i <= 10000; ++i) {
            const V i_v = V(T(i));
            const V diff = base + i_v * eps;

            // if diff and base have a different exponent then ulpDiffToReference has an uncertainty
            // of +/-1
            const V ulpDifference = ulpDiffToReference(diff, base);
            const V expectedDifference = Vc::abs(i_v);
            const V maxUncertainty = Vc::abs(abs(diff).exponent() - abs(base).exponent());

            VERIFY(Vc::abs(ulpDifference - expectedDifference) <= maxUncertainty)
                << ", base = " << base << ", epsilon = " << eps << ", diff = " << diff;
            for (int k = 0; k < V::Size; ++k) {
                VERIFY(std::abs(ulpDifference[k] - expectedDifference[k]) <= maxUncertainty[k]);
            }
        }
    }
}/*}}}*/

int main(int argc, char **argv)/*{{{*/
{
    initTest(argc, argv);

    Denormals<float>::data = Vc::malloc<float, Vc::AlignOnVector>(NDenormals);/*{{{*/
    Denormals<float>::data[0] = std::numeric_limits<float>::denorm_min();
    for (int i = 1; i < NDenormals; ++i) {
        Denormals<float>::data[i] = Denormals<float>::data[i - 1] * 2.173f;
    }
    Denormals<double>::data = Vc::malloc<double, Vc::AlignOnVector>(NDenormals);
    Denormals<double>::data[0] = std::numeric_limits<double>::denorm_min();
    for (int i = 1; i < NDenormals; ++i) {
        Denormals<double>::data[i] = Denormals<double>::data[i - 1] * 2.173;
    }/*}}}*/

    testRealTypes(isNegative);
    testRealTypes(testFrexp);
    testRealTypes(testLdexp);

    runTest(testAbs<int_v>);
    runTest(testAbs<float_v>);
    runTest(testAbs<double_v>);
    runTest(testAbs<short_v>);
    runTest(testAbs<sfloat_v>);

    testRealTypes(testUlpDiff);

    testRealTypes(testTrunc);
    testRealTypes(testFloor);
    testRealTypes(testCeil);
    testRealTypes(testExp);
    testRealTypes(testLog);
    testRealTypes(testLog2);
    testRealTypes(testLog10);

    runTest(testMax<int_v>);
    runTest(testMax<uint_v>);
    runTest(testMax<float_v>);
    runTest(testMax<double_v>);
    runTest(testMax<short_v>);
    runTest(testMax<ushort_v>);
    runTest(testMax<sfloat_v>);

    testRealTypes(testSqrt);
    testRealTypes(testRSqrt);
    testRealTypes(testSin);
    testRealTypes(testCos);
    testRealTypes(testAsin);
    testRealTypes(testAtan);
    testRealTypes(testAtan2);
    testRealTypes(testReciprocal);
    testRealTypes(testInf);
    testRealTypes(testNaN);
    testRealTypes(testRound);

    runTest(testReduceMin<float_v>);
    runTest(testReduceMin<sfloat_v>);
    runTest(testReduceMin<double_v>);
    runTest(testReduceMin<int_v>);
    runTest(testReduceMin<uint_v>);
    runTest(testReduceMin<short_v>);
    runTest(testReduceMin<ushort_v>);

    runTest(testReduceMax<float_v>);
    runTest(testReduceMax<sfloat_v>);
    runTest(testReduceMax<double_v>);
    runTest(testReduceMax<int_v>);
    runTest(testReduceMax<uint_v>);
    runTest(testReduceMax<short_v>);
    runTest(testReduceMax<ushort_v>);

    runTest(testReduceProduct<float_v>);
    runTest(testReduceProduct<sfloat_v>);
    runTest(testReduceProduct<double_v>);
    runTest(testReduceProduct<int_v>);
    runTest(testReduceProduct<uint_v>);
    runTest(testReduceProduct<short_v>);
    runTest(testReduceProduct<ushort_v>);

    runTest(testReduceSum<float_v>);
    runTest(testReduceSum<sfloat_v>);
    runTest(testReduceSum<double_v>);
    runTest(testReduceSum<int_v>);
    runTest(testReduceSum<uint_v>);
    runTest(testReduceSum<short_v>);
    runTest(testReduceSum<ushort_v>);

    testRealTypes(testSincos);
    testRealTypes(testExponent);

    return 0;
}/*}}}*/

// vim: foldmethod=marker
