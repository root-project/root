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

#ifndef VC_AVX_MATH_H
#define VC_AVX_MATH_H

#include "const.h"
#include "limits.h"
#include "macros.h"

namespace ROOT {
namespace Vc
{
namespace AVX
{
    /**
     * splits \p v into exponent and mantissa, the sign is kept with the mantissa
     *
     * The return value will be in the range [0.5, 1.0[
     * The \p e value will be an integer defining the power-of-two exponent
     */
    inline double_v frexp(double_v::AsArg v, int_v *e) {
        const m256d exponentBits = Const<double>::exponentMask().dataD();
        const m256d exponentPart = _mm256_and_pd(v.data(), exponentBits);
        e->data() = _mm256_sub_epi32(_mm256_srli_epi64(avx_cast<m256i>(exponentPart), 52), _mm256_set1_epi32(0x3fe));
        const m256d exponentMaximized = _mm256_or_pd(v.data(), exponentBits);
        double_v ret = _mm256_and_pd(exponentMaximized, _mm256_broadcast_sd(reinterpret_cast<const double *>(&c_general::frexpMask)));
        double_m zeroMask = v == double_v::Zero();
        ret(isnan(v) || !isfinite(v) || zeroMask) = v;
        e->setZero(zeroMask.data());
        return ret;
    }
    inline float_v frexp(float_v::AsArg v, int_v *e) {
        const m256 exponentBits = Const<float>::exponentMask().data();
        const m256 exponentPart = _mm256_and_ps(v.data(), exponentBits);
        e->data() = _mm256_sub_epi32(_mm256_srli_epi32(avx_cast<m256i>(exponentPart), 23), _mm256_set1_epi32(0x7e));
        const m256 exponentMaximized = _mm256_or_ps(v.data(), exponentBits);
        float_v ret = _mm256_and_ps(exponentMaximized, avx_cast<m256>(_mm256_set1_epi32(0xbf7fffffu)));
        ret(isnan(v) || !isfinite(v) || v == float_v::Zero()) = v;
        e->setZero(v == float_v::Zero());
        return ret;
    }
    inline sfloat_v frexp(sfloat_v::AsArg v, short_v *e) {
        const m256 exponentBits = Const<float>::exponentMask().data();
        const m256 exponentPart = _mm256_and_ps(v.data(), exponentBits);
        e->data() = _mm_sub_epi16(_mm_packs_epi32(_mm_srli_epi32(avx_cast<m128i>(exponentPart), 23),
                    _mm_srli_epi32(avx_cast<m128i>(hi128(exponentPart)), 23)), _mm_set1_epi16(0x7e));
        const m256 exponentMaximized = _mm256_or_ps(v.data(), exponentBits);
        sfloat_v ret = _mm256_and_ps(exponentMaximized, avx_cast<m256>(_mm256_set1_epi32(0xbf7fffffu)));
        ret(isnan(v) || !isfinite(v) || v == sfloat_v::Zero()) = v;
        e->setZero(v == sfloat_v::Zero());
        return ret;
    }

    /*             -> x * 2^e
     * x == NaN    -> NaN
     * x == (-)inf -> (-)inf
     */
    inline double_v ldexp(double_v::AsArg v, int_v::AsArg _e) {
        int_v e = _e;
        e.setZero((v == double_v::Zero()).dataI());
        const m256i exponentBits = _mm256_slli_epi64(e.data(), 52);
        return avx_cast<m256d>(_mm256_add_epi64(avx_cast<m256i>(v.data()), exponentBits));
    }
    inline float_v ldexp(float_v::AsArg v, int_v::AsArg _e) {
        int_v e = _e;
        e.setZero(static_cast<int_m>(v == float_v::Zero()));
        return (v.reinterpretCast<int_v>() + (e << 23)).reinterpretCast<float_v>();
    }
    inline sfloat_v ldexp(sfloat_v::AsArg v, short_v::AsArg _e) {
        short_v e = _e;
        e.setZero(static_cast<short_m>(v == sfloat_v::Zero()));
        e = e << (23 - 16);
        const m256i exponentBits = concat(_mm_unpacklo_epi16(_mm_setzero_si128(), e.data()),
                _mm_unpackhi_epi16(_mm_setzero_si128(), e.data()));
        return (v.reinterpretCast<int_v>() + int_v(exponentBits)).reinterpretCast<sfloat_v>();
    }

    static Vc_ALWAYS_INLINE  float_v trunc( float_v::AsArg v) { return _mm256_round_ps(v.data(), 0x3); }
    static Vc_ALWAYS_INLINE sfloat_v trunc(sfloat_v::AsArg v) { return _mm256_round_ps(v.data(), 0x3); }
    static Vc_ALWAYS_INLINE double_v trunc(double_v::AsArg v) { return _mm256_round_pd(v.data(), 0x3); }

    static Vc_ALWAYS_INLINE float_v floor(float_v::AsArg v) { return _mm256_floor_ps(v.data()); }
    static Vc_ALWAYS_INLINE sfloat_v floor(sfloat_v::AsArg v) { return _mm256_floor_ps(v.data()); }
    static Vc_ALWAYS_INLINE double_v floor(double_v::AsArg v) { return _mm256_floor_pd(v.data()); }

    static Vc_ALWAYS_INLINE float_v ceil(float_v::AsArg v) { return _mm256_ceil_ps(v.data()); }
    static Vc_ALWAYS_INLINE sfloat_v ceil(sfloat_v::AsArg v) { return _mm256_ceil_ps(v.data()); }
    static Vc_ALWAYS_INLINE double_v ceil(double_v::AsArg v) { return _mm256_ceil_pd(v.data()); }
} // namespace AVX
} // namespace Vc
} // namespace ROOT

#include "undomacros.h"
#define VC__USE_NAMESPACE AVX
#include "../common/trigonometric.h"
#define VC__USE_NAMESPACE AVX
#include "../common/logarithm.h"
#define VC__USE_NAMESPACE AVX
#include "../common/exponential.h"
#undef VC__USE_NAMESPACE

#endif // VC_AVX_MATH_H
