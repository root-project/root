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

#ifndef VC_SSE_MATH_H
#define VC_SSE_MATH_H

#include "const.h"
#include "macros.h"

namespace ROOT {
namespace Vc
{
namespace SSE
{
    /**
     * splits \p v into exponent and mantissa, the sign is kept with the mantissa
     *
     * The return value will be in the range [0.5, 1.0[
     * The \p e value will be an integer defining the power-of-two exponent
     */
    inline double_v frexp(const double_v &v, int_v *e) {
        const __m128i exponentBits = Const<double>::exponentMask().dataI();
        const __m128i exponentPart = _mm_and_si128(_mm_castpd_si128(v.data()), exponentBits);
        *e = _mm_sub_epi32(_mm_srli_epi64(exponentPart, 52), _mm_set1_epi32(0x3fe));
        const __m128d exponentMaximized = _mm_or_pd(v.data(), _mm_castsi128_pd(exponentBits));
        double_v ret = _mm_and_pd(exponentMaximized, _mm_load_pd(reinterpret_cast<const double *>(&c_general::frexpMask[0])));
        double_m zeroMask = v == double_v::Zero();
        ret(isnan(v) || !isfinite(v) || zeroMask) = v;
        e->setZero(zeroMask.data());
        return ret;
    }
    inline float_v frexp(const float_v &v, int_v *e) {
        const __m128i exponentBits = Const<float>::exponentMask().dataI();
        const __m128i exponentPart = _mm_and_si128(_mm_castps_si128(v.data()), exponentBits);
        *e = _mm_sub_epi32(_mm_srli_epi32(exponentPart, 23), _mm_set1_epi32(0x7e));
        const __m128 exponentMaximized = _mm_or_ps(v.data(), _mm_castsi128_ps(exponentBits));
        float_v ret = _mm_and_ps(exponentMaximized, _mm_castsi128_ps(_mm_set1_epi32(0xbf7fffffu)));
        ret(isnan(v) || !isfinite(v) || v == float_v::Zero()) = v;
        e->setZero(v == float_v::Zero());
        return ret;
    }
    inline sfloat_v frexp(const sfloat_v &v, short_v *e) {
        const __m128i exponentBits = Const<float>::exponentMask().dataI();
        const __m128i exponentPart0 = _mm_and_si128(_mm_castps_si128(v.data()[0]), exponentBits);
        const __m128i exponentPart1 = _mm_and_si128(_mm_castps_si128(v.data()[1]), exponentBits);
        *e = _mm_sub_epi16(_mm_packs_epi32(_mm_srli_epi32(exponentPart0, 23), _mm_srli_epi32(exponentPart1, 23)),
                _mm_set1_epi16(0x7e));
        const __m128 exponentMaximized0 = _mm_or_ps(v.data()[0], _mm_castsi128_ps(exponentBits));
        const __m128 exponentMaximized1 = _mm_or_ps(v.data()[1], _mm_castsi128_ps(exponentBits));
        sfloat_v ret = M256::create(
                _mm_and_ps(exponentMaximized0, _mm_castsi128_ps(_mm_set1_epi32(0xbf7fffffu))),
                _mm_and_ps(exponentMaximized1, _mm_castsi128_ps(_mm_set1_epi32(0xbf7fffffu)))
                );
        sfloat_m zeroMask = v == sfloat_v::Zero();
        ret(isnan(v) || !isfinite(v) || zeroMask) = v;
        e->setZero(static_cast<short_m>(zeroMask));
        return ret;
    }

    /*             -> x * 2^e
     * x == NaN    -> NaN
     * x == (-)inf -> (-)inf
     */
    inline double_v ldexp(double_v::AsArg v, int_v::AsArg _e) {
        int_v e = _e;
        e.setZero((v == double_v::Zero()).dataI());
        const __m128i exponentBits = _mm_slli_epi64(e.data(), 52);
        return _mm_castsi128_pd(_mm_add_epi64(_mm_castpd_si128(v.data()), exponentBits));
    }
    inline float_v ldexp(float_v::AsArg v, int_v::AsArg _e) {
        int_v e = _e;
        e.setZero(static_cast<int_m>(v == float_v::Zero()));
        return (v.reinterpretCast<int_v>() + (e << 23)).reinterpretCast<float_v>();
    }
    inline sfloat_v ldexp(sfloat_v::AsArg v, short_v::AsArg _e) {
        short_v e = _e;
        e.setZero(static_cast<short_m>(v == sfloat_v::Zero()));
        e <<= (23 - 16);
        const __m128i exponentBits0 = _mm_unpacklo_epi16(_mm_setzero_si128(), e.data());
        const __m128i exponentBits1 = _mm_unpackhi_epi16(_mm_setzero_si128(), e.data());
        return M256::create(_mm_castsi128_ps(_mm_add_epi32(_mm_castps_si128(v.data()[0]), exponentBits0)),
                _mm_castsi128_ps(_mm_add_epi32(_mm_castps_si128(v.data()[1]), exponentBits1)));
    }

#ifdef VC_IMPL_SSE4_1
    inline double_v trunc(double_v::AsArg v) { return _mm_round_pd(v.data(), 0x3); }
    inline float_v trunc(float_v::AsArg v) { return _mm_round_ps(v.data(), 0x3); }
    inline sfloat_v trunc(sfloat_v::AsArg v) { return M256::create(_mm_round_ps(v.data()[0], 0x3),
            _mm_round_ps(v.data()[1], 0x3)); }

    inline double_v floor(double_v::AsArg v) { return _mm_floor_pd(v.data()); }
    inline float_v floor(float_v::AsArg v) { return _mm_floor_ps(v.data()); }
    inline sfloat_v floor(sfloat_v::AsArg v) { return M256::create(_mm_floor_ps(v.data()[0]),
            _mm_floor_ps(v.data()[1])); }

    inline double_v ceil(double_v::AsArg v) { return _mm_ceil_pd(v.data()); }
    inline float_v ceil(float_v::AsArg v) { return _mm_ceil_ps(v.data()); }
    inline sfloat_v ceil(sfloat_v::AsArg v) { return M256::create(_mm_ceil_ps(v.data()[0]),
            _mm_ceil_ps(v.data()[1])); }
#else
    static inline void floor_shift(float_v &v, float_v::AsArg e)
    {
        int_v x = _mm_setallone_si128();
        x <<= 23;
        x >>= static_cast<int_v>(e);
        v &= x.reinterpretCast<float_v>();
    }

    static inline void floor_shift(sfloat_v &v, sfloat_v::AsArg e)
    {
        int_v x = _mm_setallone_si128();
        x <<= 23;
        int_v y = x;
        x >>= _mm_cvttps_epi32(e.data()[0]);
        y >>= _mm_cvttps_epi32(e.data()[1]);
        v.data()[0] = _mm_and_ps(v.data()[0], _mm_castsi128_ps(x.data()));
        v.data()[1] = _mm_and_ps(v.data()[1], _mm_castsi128_ps(y.data()));
    }

    static inline void floor_shift(double_v &v, double_v::AsArg e)
    {
        const long long initialMask = 0xfff0000000000000ull;
        const uint_v shifts = static_cast<uint_v>(e);
        union d_ll {
            long long ll;
            double d;
        };
        d_ll mask0 = { initialMask >> shifts[0] };
        d_ll mask1 = { initialMask >> shifts[1] };
        v &= double_v(_mm_setr_pd(mask0.d, mask1.d));
    }

    template<typename T>
    inline Vector<T> trunc(VC_ALIGNED_PARAMETER(Vector<T>) _v) {
        typedef Vector<T> V;
        typedef typename V::Mask M;

        V v = _v;
        V e = abs(v).exponent();
        const M negativeExponent = e < 0;
        e.setZero(negativeExponent);
        //const M negativeInput = v < V::Zero();

        floor_shift(v, e);

        v.setZero(negativeExponent);
        //v(negativeInput && _v != v) -= V::One();
        return v;
    }

    template<typename T>
    inline Vector<T> floor(VC_ALIGNED_PARAMETER(Vector<T>) _v) {
        typedef Vector<T> V;
        typedef typename V::Mask M;

        V v = _v;
        V e = abs(v).exponent();
        const M negativeExponent = e < 0;
        e.setZero(negativeExponent);
        const M negativeInput = v < V::Zero();

        floor_shift(v, e);

        v.setZero(negativeExponent);
        v(negativeInput && _v != v) -= V::One();
        return v;
    }

    template<typename T>
    inline Vector<T> ceil(VC_ALIGNED_PARAMETER(Vector<T>) _v) {
        typedef Vector<T> V;
        typedef typename V::Mask M;

        V v = _v;
        V e = abs(v).exponent();
        const M negativeExponent = e < 0;
        e.setZero(negativeExponent);
        const M positiveInput = v > V::Zero();

        floor_shift(v, e);

        v.setZero(negativeExponent);
        v(positiveInput && _v != v) += V::One();
        return v;
    }
#endif
} // namespace SSE
} // namespace Vc
} // namespace ROOT

#include "undomacros.h"

#define VC__USE_NAMESPACE SSE
#include "../common/trigonometric.h"
#define VC__USE_NAMESPACE SSE
#include "../common/logarithm.h"
#define VC__USE_NAMESPACE SSE
#include "../common/exponential.h"
#undef VC__USE_NAMESPACE

#endif // VC_SSE_MATH_H
