#ifndef COMMON_EXPONENTIAL_H
#define COMMON_EXPONENTIAL_H
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

    -------------------------------------------------------------------

    The exp implementation is derived from Cephes, which carries the
    following Copyright notice:

    Cephes Math Library Release 2.2:  June, 1992
    Copyright 1984, 1987, 1989 by Stephen L. Moshier
    Direct inquiries to 30 Frost Street, Cambridge, MA 02140

}}}*/

#ifndef VC_COMMON_EXPONENTIAL_H
#define VC_COMMON_EXPONENTIAL_H

#include "macros.h"
namespace ROOT {
namespace Vc
{
namespace Common
{
    using Vc::VC__USE_NAMESPACE::c_log;
    using Vc::VC__USE_NAMESPACE::Vector;
    using Vc::VC__USE_NAMESPACE::floor;
    using Vc::VC__USE_NAMESPACE::ldexp;

    static const float log2_e = 1.44269504088896341f;
    static const float MAXLOGF = 88.72283905206835f;
    static const float MINLOGF = -103.278929903431851103f; /* log(2^-149) */
    static const float MAXNUMF = 3.4028234663852885981170418348451692544e38f;

    template<typename T> struct TypenameForLdexp { typedef Vector<int> Type; };
    template<> struct TypenameForLdexp<Vc::sfloat> { typedef Vector<short> Type; };

    template<typename T> static inline Vector<T> exp(VC_ALIGNED_PARAMETER(Vector<T>) _x) {
        typedef Vector<T> V;
        typedef typename V::Mask M;
        typedef typename TypenameForLdexp<T>::Type I;
        typedef Const<T> C;

        V x(_x);

        const M overflow  = x > MAXLOGF;
        const M underflow = x < MINLOGF;

        // log₂(eˣ) = x * log₂(e) * log₂(2)
        //          = log₂(2^(x * log₂(e)))
        // => eˣ = 2^(x * log₂(e))
        // => n  = ⌊x * log₂(e) + ½⌋
        // => y  = x - n * ln(2)       | recall that: ln(2) * log₂(e) == 1
        // <=> eˣ = 2ⁿ * eʸ
        V z = floor(C::log2_e() * x + 0.5f);
        I n = static_cast<I>(z);
        x -= z * C::ln2_large();
        x -= z * C::ln2_small();

        /* Theoretical peak relative error in [-0.5, +0.5] is 4.2e-9. */
        z = ((((( 1.9875691500E-4f  * x
                + 1.3981999507E-3f) * x
                + 8.3334519073E-3f) * x
                + 4.1665795894E-2f) * x
                + 1.6666665459E-1f) * x
                + 5.0000001201E-1f) * (x * x)
                + x
                + 1.0f;

        x = ldexp(z, n); // == z * 2ⁿ

        x(overflow) = std::numeric_limits<typename V::EntryType>::infinity();
        x.setZero(underflow);

        return x;
    }
    static inline Vector<double> exp(Vector<double>::AsArg _x) {
        Vector<double> x = _x;
        typedef Vector<double> V;
        typedef V::Mask M;
        typedef Const<double> C;

        const M overflow  = x > Vc_buildDouble( 1, 0x0006232bdd7abcd2ull, 9); // max log
        const M underflow = x < Vc_buildDouble(-1, 0x0006232bdd7abcd2ull, 9); // min log

        V px = floor(C::log2_e() * x + 0.5);
#ifdef VC_IMPL_SSE
        Vector<int> n(px);
        n.data() = Mem::permute<X0, X2, X1, X3>(n.data());
#elif defined(VC_IMPL_AVX)
        __m128i tmp = _mm256_cvttpd_epi32(px.data());
        Vector<int> n = AVX::concat(_mm_unpacklo_epi32(tmp, tmp), _mm_unpackhi_epi32(tmp, tmp));
#endif
        x -= px * C::ln2_large(); //Vc_buildDouble(1, 0x00062e4000000000ull, -1);  // ln2
        x -= px * C::ln2_small(); //Vc_buildDouble(1, 0x0007f7d1cf79abcaull, -20); // ln2

        const double P[] = {
            Vc_buildDouble(1, 0x000089cdd5e44be8ull, -13),
            Vc_buildDouble(1, 0x000f06d10cca2c7eull,  -6),
            Vc_buildDouble(1, 0x0000000000000000ull,   0)
        };
        const double Q[] = {
            Vc_buildDouble(1, 0x00092eb6bc365fa0ull, -19),
            Vc_buildDouble(1, 0x0004ae39b508b6c0ull,  -9),
            Vc_buildDouble(1, 0x000d17099887e074ull,  -3),
            Vc_buildDouble(1, 0x0000000000000000ull,   1)
        };
        const V x2 = x * x;
        px = x * ((P[0] * x2 + P[1]) * x2 + P[2]);
        x =  px / ((((Q[0] * x2 + Q[1]) * x2 + Q[2]) * x2 + Q[3]) - px);
        x = V::One() + 2.0 * x;

        x = ldexp(x, n); // == x * 2ⁿ

        x(overflow) = std::numeric_limits<double>::infinity();
        x.setZero(underflow);

        return x;
    }
} // namespace Common
namespace VC__USE_NAMESPACE
{
    using Vc::Common::exp;
} // namespace VC__USE_NAMESPACE
} // namespace Vc
} // namespace ROOT
#include "undomacros.h"

#endif // VC_COMMON_EXPONENTIAL_H
#endif // COMMON_EXPONENTIAL_H
