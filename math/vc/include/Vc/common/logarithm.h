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

/* The log implementations are based on code from Julien Pommier which carries the following
   copyright information:
 */
/*
   Inspired by Intel Approximate Math library, and based on the
   corresponding algorithms of the cephes math library
*/
/* Copyright (C) 2007  Julien Pommier

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

#ifndef VC_COMMON_LOGARITHM_H
#define VC_COMMON_LOGARITHM_H

#include "macros.h"
namespace ROOT {
namespace Vc
{
namespace Common
{
#ifdef VC__USE_NAMESPACE
using Vc::VC__USE_NAMESPACE::Const;
using Vc::VC__USE_NAMESPACE::Vector;
namespace Internal
{
    using namespace Vc::VC__USE_NAMESPACE::Internal;
} // namespace Internal
#endif
enum LogarithmBase {
    BaseE, Base10, Base2
};

template<LogarithmBase Base>
struct LogImpl
{
    template<typename T> static Vc_ALWAYS_INLINE void log_series(Vector<T> &VC_RESTRICT x, typename Vector<T>::AsArg exponent) {
        typedef Vector<T> V;
        typedef Const<T> C;
        // Taylor series around x = 2^exponent
        //   f(x) = ln(x)   → exponent * ln(2) → C::ln2_small + C::ln2_large
        //  f'(x) =    x⁻¹  →  x               → 1
        // f''(x) = -  x⁻²  → -x² / 2          → C::_1_2()
        //        =  2!x⁻³  →  x³ / 3          → C::P(8)
        //        = -3!x⁻⁴  → -x⁴ / 4          → C::P(7)
        //        =  4!x⁻⁵  →  x⁵ / 5          → C::P(6)
        // ...
        // The high order coefficients are adjusted to reduce the error that occurs from ommission
        // of higher order terms.
        // P(0) is the smallest term and |x| < 1 ⇒ |xⁿ| > |xⁿ⁺¹|
        // The order of additions must go from smallest to largest terms
        const V x2 = x * x; // 0 → 4
#ifdef VC_LOG_ILP
        V y2 = (C::P(6) * /*4 →  8*/ x2 + /* 8 → 11*/ C::P(7) * /*1 → 5*/ x) + /*11 → 14*/ C::P(8);
        V y0 = (C::P(0) * /*5 →  9*/ x2 + /* 9 → 12*/ C::P(1) * /*2 → 6*/ x) + /*12 → 15*/ C::P(2);
        V y1 = (C::P(3) * /*6 → 10*/ x2 + /*10 → 13*/ C::P(4) * /*3 → 7*/ x) + /*13 → 16*/ C::P(5);
        const V x3 = x2 * x;  // 7 → 11
        const V x6 = x3 * x3; // 11 → 15
        const V x9 = x6 * x3; // 15 → 19
        V y = (y0 * /*19 → 23*/ x9 + /*23 → 26*/ y1 * /*16 → 20*/ x6) + /*26 → 29*/ y2 * /*14 → 18*/ x3;
#elif defined VC_LOG_ILP2
        /*
         *                            name start done
         *  movaps %xmm0, %xmm1     ; x     0     1
         *  movaps %xmm0, %xmm2     ; x     0     1
         *  mulps %xmm1, %xmm1      ; x2    1     5 *xmm1
         *  movaps <P8>, %xmm15     ; y8    1     2
         *  mulps %xmm1, %xmm2      ; x3    5     9 *xmm2
         *  movaps %xmm1, %xmm3     ; x2    5     6
         *  movaps %xmm1, %xmm4     ; x2    5     6
         *  mulps %xmm3, %xmm3      ; x4    6    10 *xmm3
         *  movaps %xmm2, %xmm5     ; x3    9    10
         *  movaps %xmm2, %xmm6     ; x3    9    10
         *  mulps %xmm2, %xmm4      ; x5    9    13 *xmm4
         *  movaps %xmm3, %xmm7     ; x4   10    11
         *  movaps %xmm3, %xmm8     ; x4   10    11
         *  movaps %xmm3, %xmm9     ; x4   10    11
         *  mulps %xmm5, %xmm5      ; x6   10    14 *xmm5
         *  mulps %xmm3, %xmm6      ; x7   11    15 *xmm6
         *  mulps %xmm7, %xmm7      ; x8   12    16 *xmm7
         *  movaps %xmm4, %xmm10    ; x5   13    14
         *  mulps %xmm4, %xmm8      ; x9   13    17 *xmm8
         *  mulps %xmm5, %xmm10     ; x11  14    18 *xmm10
         *  mulps %xmm5, %xmm9      ; x10  15    19 *xmm9
         *  mulps <P0>, %xmm10      ; y0   18    22
         *  mulps <P1>, %xmm9       ; y1   19    23
         *  mulps <P2>, %xmm8       ; y2   20    24
         *  mulps <P3>, %xmm7       ; y3   21    25
         *  addps %xmm10, %xmm9     ; y    23    26
         *  addps %xmm9, %xmm8      ; y    26    29
         *  addps %xmm8, %xmm7      ; y    29    32
         */
        const V x3 = x2 * x;  // 4  → 8
        const V x4 = x2 * x2; // 5  → 9
        const V x5 = x2 * x3; // 8  → 12
        const V x6 = x3 * x3; // 9  → 13
        const V x7 = x4 * x3; //
        const V x8 = x4 * x4;
        const V x9 = x5 * x4;
        const V x10 = x5 * x5;
        const V x11 = x5 * x6; // 13 → 17
        V y = C::P(0) * x11 + C::P(1) * x10 + C::P(2) * x9 + C::P(3) * x8 + C::P(4) * x7
            + C::P(5) * x6  + C::P(6) * x5  + C::P(7) * x4 + C::P(8) * x3;
#else
        V y = C::P(0);
        unrolled_loop16(i, 1, 9,
                y = y * x + C::P(i);
                );
        y *= x * x2;
#endif
        switch (Base) {
        case BaseE:
            // ln(2) is split in two parts to increase precision (i.e. ln2_small + ln2_large = ln(2))
            y += exponent * C::ln2_small();
            y -= x2 * C::_1_2(); // [0, 0.25[
            x += y;
            x += exponent * C::ln2_large();
            break;
        case Base10:
            y += exponent * C::ln2_small();
            y -= x2 * C::_1_2(); // [0, 0.25[
            x += y;
            x += exponent * C::ln2_large();
            x *= C::log10_e();
            break;
        case Base2:
            {
                const V x_ = x;
                x *= C::log2_e();
                y *= C::log2_e();
                y -= x_ * x * C::_1_2(); // [0, 0.25[
                x += y;
                x += exponent;
                break;
            }
        }
    }

    static Vc_ALWAYS_INLINE void log_series(Vector<double> &VC_RESTRICT x, Vector<double>::AsArg exponent) {
        typedef Vector<double> V;
        typedef Const<double> C;
        const V x2 = x * x;
        V y = C::P(0);
        V y2 = C::Q(0) + x;
        unrolled_loop16(i, 1, 5,
                y = y * x + C::P(i);
                y2 = y2 * x + C::Q(i);
                );
        y2 = x / y2;
        y = y * x + C::P(5);
        y = x2 * y * y2;
        // TODO: refactor the following with the float implementation:
        switch (Base) {
        case BaseE:
            // ln(2) is split in two parts to increase precision (i.e. ln2_small + ln2_large = ln(2))
            y += exponent * C::ln2_small();
            y -= x2 * C::_1_2(); // [0, 0.25[
            x += y;
            x += exponent * C::ln2_large();
            break;
        case Base10:
            y += exponent * C::ln2_small();
            y -= x2 * C::_1_2(); // [0, 0.25[
            x += y;
            x += exponent * C::ln2_large();
            x *= C::log10_e();
            break;
        case Base2:
            {
                const V x_ = x;
                x *= C::log2_e();
                y *= C::log2_e();
                y -= x_ * x * C::_1_2(); // [0, 0.25[
                x += y;
                x += exponent;
                break;
            }
        }
    }

    template<typename T> static inline Vector<T> calc(VC_ALIGNED_PARAMETER(Vector<T>) _x) {
        typedef Vector<T> V;
        typedef typename V::Mask M;
        typedef Const<T> C;

        V x(_x);

        const M invalidMask = x < V::Zero();
        const M infinityMask = x == V::Zero();
        const M denormal = x <= C::min();

        x(denormal) *= V(Vc_buildDouble(1, 0, 54)); // 2²⁵
        V exponent = Internal::exponent(x.data()); // = ⎣log₂(x)⎦
        exponent(denormal) -= 54;

        x.setZero(C::exponentMask()); // keep only the fractional part ⇒ x ∈ [1, 2[
        x |= C::_1_2();               // and set the exponent to 2⁻¹   ⇒ x ∈ [½, 1[

        // split calculation in two cases:
        // A: x ∈ [½, √½[
        // B: x ∈ [√½, 1[
        // √½ defines the point where Δe(x) := log₂(x) - ⎣log₂(x)⎦ = ½, i.e.
        // log₂(√½) - ⎣log₂(√½)⎦ = ½ * -1 - ⎣½ * -1⎦ = -½ + 1 = ½

        const M smallX = x < C::_1_sqrt2();
        x(smallX) += x; // => x ∈ [√½,     1[ ∪ [1.5, 1 + √½[
        x -= V::One();  // => x ∈ [√½ - 1, 0[ ∪ [0.5, √½[
        exponent(!smallX) += V::One();

        log_series(x, exponent); // A: (ˣ⁄₂ᵉ - 1, e)  B: (ˣ⁄₂ᵉ⁺¹ - 1, e + 1)

        x.setQnan(invalidMask);        // x < 0 → NaN
        x(infinityMask) = C::neginf(); // x = 0 → -∞

        return x;
    }
};

template<typename T> static Vc_ALWAYS_INLINE Vc_CONST Vector<T> log(VC_ALIGNED_PARAMETER(Vector<T>) x) {
    return LogImpl<BaseE>::calc(x);
}
template<typename T> static Vc_ALWAYS_INLINE Vc_CONST Vector<T> log10(VC_ALIGNED_PARAMETER(Vector<T>) x) {
    return LogImpl<Base10>::calc(x);
}
template<typename T> static Vc_ALWAYS_INLINE Vc_CONST Vector<T> log2(VC_ALIGNED_PARAMETER(Vector<T>) x) {
    return LogImpl<Base2>::calc(x);
}
} // namespace Common
#ifdef VC__USE_NAMESPACE
namespace VC__USE_NAMESPACE
{
    using Vc::Common::log;
    using Vc::Common::log10;
    using Vc::Common::log2;
} // namespace VC__USE_NAMESPACE
#undef VC__USE_NAMESPACE
#endif
} // namespace Vc
} // namespace ROOT
#include "undomacros.h"

#endif // VC_COMMON_LOGARITHM_H
