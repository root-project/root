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

#ifndef AVX_CASTS_H
#define AVX_CASTS_H

#include "intrinsics.h"
#include "types.h"
#include "macros.h"

namespace ROOT {
namespace Vc
{
namespace AVX
{
    template<typename T> static Vc_INTRINSIC_L T avx_cast(param128  v) Vc_INTRINSIC_R;
    template<typename T> static Vc_INTRINSIC_L T avx_cast(param128i v) Vc_INTRINSIC_R;
    template<typename T> static Vc_INTRINSIC_L T avx_cast(param128d v) Vc_INTRINSIC_R;
    template<typename T> static Vc_INTRINSIC_L T avx_cast(param256  v) Vc_INTRINSIC_R;
    template<typename T> static Vc_INTRINSIC_L T avx_cast(param256i v) Vc_INTRINSIC_R;
    template<typename T> static Vc_INTRINSIC_L T avx_cast(param256d v) Vc_INTRINSIC_R;

#ifdef VC_UNCONDITIONAL_AVX2_INTRINSICS
    template<typename T> static Vc_INTRINSIC T avx_cast(__m128  v) { return avx_cast<T>(param128 (v)); }
    template<typename T> static Vc_INTRINSIC T avx_cast(__m128i v) { return avx_cast<T>(param128i(v)); }
    template<typename T> static Vc_INTRINSIC T avx_cast(__m128d v) { return avx_cast<T>(param128d(v)); }
    template<typename T> static Vc_INTRINSIC T avx_cast(__m256  v) { return avx_cast<T>(param256 (v)); }
    template<typename T> static Vc_INTRINSIC T avx_cast(__m256i v) { return avx_cast<T>(param256i(v)); }
    template<typename T> static Vc_INTRINSIC T avx_cast(__m256d v) { return avx_cast<T>(param256d(v)); }
#endif

    // 128 -> 128
    template<> Vc_INTRINSIC m128  avx_cast(param128  v) { return v; }
    template<> Vc_INTRINSIC m128  avx_cast(param128i v) { return _mm_castsi128_ps(v); }
    template<> Vc_INTRINSIC m128  avx_cast(param128d v) { return _mm_castpd_ps(v); }
    template<> Vc_INTRINSIC m128i avx_cast(param128  v) { return _mm_castps_si128(v); }
    template<> Vc_INTRINSIC m128i avx_cast(param128i v) { return v; }
    template<> Vc_INTRINSIC m128i avx_cast(param128d v) { return _mm_castpd_si128(v); }
    template<> Vc_INTRINSIC m128d avx_cast(param128  v) { return _mm_castps_pd(v); }
    template<> Vc_INTRINSIC m128d avx_cast(param128i v) { return _mm_castsi128_pd(v); }
    template<> Vc_INTRINSIC m128d avx_cast(param128d v) { return v; }

    // 128 -> 256
    // FIXME: the following casts leave the upper 128bits undefined. With GCC and ICC I've never
    // seen the cast not do what I want though: after a VEX-coded SSE instruction the register's
    // upper 128bits are zero. Thus using the same register as AVX register will have the upper
    // 128bits zeroed. MSVC, though, implements _mm256_castxx128_xx256 with a 128bit move to memory
    // + 256bit load. Thus the upper 128bits are really undefined. But there is no intrinsic to do
    // what I want (i.e. alias the register, disallowing the move to memory in-between). I'm stuck,
    // do we really want to rely on specific compiler behavior here?
    template<> Vc_INTRINSIC m256  avx_cast(param128  v) { return _mm256_castps128_ps256(v); }
    template<> Vc_INTRINSIC m256  avx_cast(param128i v) { return _mm256_castps128_ps256(_mm_castsi128_ps(v)); }
    template<> Vc_INTRINSIC m256  avx_cast(param128d v) { return _mm256_castps128_ps256(_mm_castpd_ps(v)); }
    template<> Vc_INTRINSIC m256i avx_cast(param128  v) { return _mm256_castsi128_si256(_mm_castps_si128(v)); }
    template<> Vc_INTRINSIC m256i avx_cast(param128i v) { return _mm256_castsi128_si256(v); }
    template<> Vc_INTRINSIC m256i avx_cast(param128d v) { return _mm256_castsi128_si256(_mm_castpd_si128(v)); }
    template<> Vc_INTRINSIC m256d avx_cast(param128  v) { return _mm256_castpd128_pd256(_mm_castps_pd(v)); }
    template<> Vc_INTRINSIC m256d avx_cast(param128i v) { return _mm256_castpd128_pd256(_mm_castsi128_pd(v)); }
    template<> Vc_INTRINSIC m256d avx_cast(param128d v) { return _mm256_castpd128_pd256(v); }

#ifdef VC_MSVC
    static Vc_INTRINSIC Vc_CONST m256  zeroExtend(param128  v) { return _mm256_permute2f128_ps   (_mm256_castps128_ps256(v), _mm256_castps128_ps256(v), 0x80); }
    static Vc_INTRINSIC Vc_CONST m256i zeroExtend(param128i v) { return _mm256_permute2f128_si256(_mm256_castsi128_si256(v), _mm256_castsi128_si256(v), 0x80); }
    static Vc_INTRINSIC Vc_CONST m256d zeroExtend(param128d v) { return _mm256_permute2f128_pd   (_mm256_castpd128_pd256(v), _mm256_castpd128_pd256(v), 0x80); }
#else
    static Vc_INTRINSIC Vc_CONST m256  zeroExtend(param128  v) { return _mm256_castps128_ps256(v); }
    static Vc_INTRINSIC Vc_CONST m256i zeroExtend(param128i v) { return _mm256_castsi128_si256(v); }
    static Vc_INTRINSIC Vc_CONST m256d zeroExtend(param128d v) { return _mm256_castpd128_pd256(v); }
#ifdef VC_ICC
    static Vc_INTRINSIC Vc_CONST m256  zeroExtend(__m128  v) { return _mm256_castps128_ps256(v); }
    static Vc_INTRINSIC Vc_CONST m256i zeroExtend(__m128i v) { return _mm256_castsi128_si256(v); }
    static Vc_INTRINSIC Vc_CONST m256d zeroExtend(__m128d v) { return _mm256_castpd128_pd256(v); }
#endif
#endif

    // 256 -> 128
    template<> Vc_INTRINSIC m128  avx_cast(param256  v) { return _mm256_castps256_ps128(v); }
    template<> Vc_INTRINSIC m128  avx_cast(param256i v) { return _mm256_castps256_ps128(_mm256_castsi256_ps(v)); }
    template<> Vc_INTRINSIC m128  avx_cast(param256d v) { return _mm256_castps256_ps128(_mm256_castpd_ps(v)); }
    template<> Vc_INTRINSIC m128i avx_cast(param256  v) { return _mm256_castsi256_si128(_mm256_castps_si256(v)); }
    template<> Vc_INTRINSIC m128i avx_cast(param256i v) { return _mm256_castsi256_si128(v); }
    template<> Vc_INTRINSIC m128i avx_cast(param256d v) { return _mm256_castsi256_si128(_mm256_castpd_si256(v)); }
    template<> Vc_INTRINSIC m128d avx_cast(param256  v) { return _mm256_castpd256_pd128(_mm256_castps_pd(v)); }
    template<> Vc_INTRINSIC m128d avx_cast(param256i v) { return _mm256_castpd256_pd128(_mm256_castsi256_pd(v)); }
    template<> Vc_INTRINSIC m128d avx_cast(param256d v) { return _mm256_castpd256_pd128(v); }

    // 256 -> 256
    template<> Vc_INTRINSIC m256  avx_cast(param256  v) { return v; }
    template<> Vc_INTRINSIC m256  avx_cast(param256i v) { return _mm256_castsi256_ps(v); }
    template<> Vc_INTRINSIC m256  avx_cast(param256d v) { return _mm256_castpd_ps(v); }
    template<> Vc_INTRINSIC m256i avx_cast(param256  v) { return _mm256_castps_si256(v); }
    template<> Vc_INTRINSIC m256i avx_cast(param256i v) { return v; }
    template<> Vc_INTRINSIC m256i avx_cast(param256d v) { return _mm256_castpd_si256(v); }
    template<> Vc_INTRINSIC m256d avx_cast(param256  v) { return _mm256_castps_pd(v); }
    template<> Vc_INTRINSIC m256d avx_cast(param256i v) { return _mm256_castsi256_pd(v); }
    template<> Vc_INTRINSIC m256d avx_cast(param256d v) { return v; }

    // simplify splitting 256-bit registers in 128-bit registers
    Vc_INTRINSIC Vc_CONST m128  lo128(param256  v) { return avx_cast<m128>(v); }
    Vc_INTRINSIC Vc_CONST m128d lo128(param256d v) { return avx_cast<m128d>(v); }
    Vc_INTRINSIC Vc_CONST m128i lo128(param256i v) { return avx_cast<m128i>(v); }
    Vc_INTRINSIC Vc_CONST m128  hi128(param256  v) { return _mm256_extractf128_ps(v, 1); }
    Vc_INTRINSIC Vc_CONST m128d hi128(param256d v) { return _mm256_extractf128_pd(v, 1); }
    Vc_INTRINSIC Vc_CONST m128i hi128(param256i v) { return _mm256_extractf128_si256(v, 1); }

    // simplify combining 128-bit registers in 256-bit registers
    Vc_INTRINSIC Vc_CONST m256  concat(param128  a, param128  b) { return _mm256_insertf128_ps   (avx_cast<m256 >(a), b, 1); }
    Vc_INTRINSIC Vc_CONST m256d concat(param128d a, param128d b) { return _mm256_insertf128_pd   (avx_cast<m256d>(a), b, 1); }
    Vc_INTRINSIC Vc_CONST m256i concat(param128i a, param128i b) { return _mm256_insertf128_si256(avx_cast<m256i>(a), b, 1); }
#ifdef VC_UNCONDITIONAL_AVX2_INTRINSICS
    Vc_INTRINSIC Vc_CONST m256  concat(__m128  a, param128  b) { return _mm256_insertf128_ps   (avx_cast<m256 >(a), b, 1); }
    Vc_INTRINSIC Vc_CONST m256d concat(__m128d a, param128d b) { return _mm256_insertf128_pd   (avx_cast<m256d>(a), b, 1); }
    Vc_INTRINSIC Vc_CONST m256i concat(__m128i a, param128i b) { return _mm256_insertf128_si256(avx_cast<m256i>(a), b, 1); }
    Vc_INTRINSIC Vc_CONST m256  concat(param128  a, __m128  b) { return _mm256_insertf128_ps   (avx_cast<m256 >(a), b, 1); }
    Vc_INTRINSIC Vc_CONST m256d concat(param128d a, __m128d b) { return _mm256_insertf128_pd   (avx_cast<m256d>(a), b, 1); }
    Vc_INTRINSIC Vc_CONST m256i concat(param128i a, __m128i b) { return _mm256_insertf128_si256(avx_cast<m256i>(a), b, 1); }
    Vc_INTRINSIC Vc_CONST m256  concat(__m128  a, __m128  b) { return _mm256_insertf128_ps   (avx_cast<m256 >(a), b, 1); }
    Vc_INTRINSIC Vc_CONST m256d concat(__m128d a, __m128d b) { return _mm256_insertf128_pd   (avx_cast<m256d>(a), b, 1); }
    Vc_INTRINSIC Vc_CONST m256i concat(__m128i a, __m128i b) { return _mm256_insertf128_si256(avx_cast<m256i>(a), b, 1); }
#endif

    template<typename From, typename To> struct StaticCastHelper {};
    template<> struct StaticCastHelper<float         , int           > { static Vc_ALWAYS_INLINE Vc_CONST m256i  cast(param256  v) { return _mm256_cvttps_epi32(v); } };
    template<> struct StaticCastHelper<double        , int           > { static Vc_ALWAYS_INLINE Vc_CONST m256i  cast(param256d v) { return avx_cast<m256i>(_mm256_cvttpd_epi32(v)); } };
    template<> struct StaticCastHelper<int           , int           > { static Vc_ALWAYS_INLINE Vc_CONST m256i  cast(param256i v) { return v; } };
    template<> struct StaticCastHelper<unsigned int  , int           > { static Vc_ALWAYS_INLINE Vc_CONST m256i  cast(param256i v) { return v; } };
    template<> struct StaticCastHelper<short         , int           > { static Vc_ALWAYS_INLINE Vc_CONST m256i  cast(param128i   v) { return concat(_mm_srai_epi32(_mm_unpacklo_epi16(v, v), 16), _mm_srai_epi32(_mm_unpackhi_epi16(v, v), 16)); } };
    template<> struct StaticCastHelper<float         , unsigned int  > { static inline Vc_CONST m256i  cast(param256  v) {
        return _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(_mm256_cvttps_epi32(v)),
                _mm256_castsi256_ps(_mm256_add_epi32(m256i(_mm256_cvttps_epi32(_mm256_sub_ps(v, _mm256_set2power31_ps()))), _mm256_set2power31_epu32())),
                _mm256_cmpge_ps(v, _mm256_set2power31_ps())
                ));

    } };
    template<> struct StaticCastHelper<double        , unsigned int  > { static Vc_ALWAYS_INLINE Vc_CONST m256i  cast(param256d v) { return avx_cast<m256i>(_mm256_cvttpd_epi32(v)); } };
    template<> struct StaticCastHelper<int           , unsigned int  > { static Vc_ALWAYS_INLINE Vc_CONST m256i  cast(param256i v) { return v; } };
    template<> struct StaticCastHelper<unsigned int  , unsigned int  > { static Vc_ALWAYS_INLINE Vc_CONST m256i  cast(param256i v) { return v; } };
    template<> struct StaticCastHelper<unsigned short, unsigned int  > { static Vc_ALWAYS_INLINE Vc_CONST m256i  cast(param128i   v) { return concat(_mm_srli_epi32(_mm_unpacklo_epi16(v, v), 16), _mm_srli_epi32(_mm_unpackhi_epi16(v, v), 16)); } };
    template<> struct StaticCastHelper<float         , float         > { static Vc_ALWAYS_INLINE Vc_CONST m256   cast(param256  v) { return v; } };
    template<> struct StaticCastHelper<double        , float         > { static Vc_ALWAYS_INLINE Vc_CONST m256   cast(param256d v) { return avx_cast<m256>(_mm256_cvtpd_ps(v)); } };
    template<> struct StaticCastHelper<int           , float         > { static Vc_ALWAYS_INLINE Vc_CONST m256   cast(param256i v) { return _mm256_cvtepi32_ps(v); } };
    template<> struct StaticCastHelper<unsigned int  , float         > { static inline Vc_CONST m256   cast(param256i v) {
        return _mm256_blendv_ps(
                _mm256_cvtepi32_ps(v),
                _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(v, _mm256_set2power31_epu32())), _mm256_set2power31_ps()),
                _mm256_castsi256_ps(_mm256_cmplt_epi32(v, _mm256_setzero_si256()))
                );
    } };
    template<> struct StaticCastHelper<short         , float         > { static Vc_ALWAYS_INLINE Vc_CONST m256  cast(param128i v) { return _mm256_cvtepi32_ps(StaticCastHelper<short, int>::cast(v)); } };
    template<> struct StaticCastHelper<unsigned short, float         > { static Vc_ALWAYS_INLINE Vc_CONST m256  cast(param128i v) { return _mm256_cvtepi32_ps(StaticCastHelper<unsigned short, unsigned int>::cast(v)); } };
    template<> struct StaticCastHelper<float         , double        > { static Vc_ALWAYS_INLINE Vc_CONST m256d cast(param256  v) { return _mm256_cvtps_pd(avx_cast<m128>(v)); } };
    template<> struct StaticCastHelper<double        , double        > { static Vc_ALWAYS_INLINE Vc_CONST m256d cast(param256d v) { return v; } };
    template<> struct StaticCastHelper<int           , double        > { static Vc_ALWAYS_INLINE Vc_CONST m256d cast(param256i v) { return _mm256_cvtepi32_pd(avx_cast<m128i>(v)); } };
    template<> struct StaticCastHelper<unsigned int  , double        > { static Vc_ALWAYS_INLINE Vc_CONST m256d cast(param256i v) { return _mm256_cvtepi32_pd(avx_cast<m128i>(v)); } };
    template<> struct StaticCastHelper<int           , short         > { static Vc_ALWAYS_INLINE Vc_CONST m128i cast(param256i v) { return _mm_packs_epi32(lo128(v), hi128(v)); } };
    template<> struct StaticCastHelper<float         , short         > { static Vc_ALWAYS_INLINE Vc_CONST m128i cast(param256  v) { return StaticCastHelper<int, short>::cast(StaticCastHelper<float, int>::cast(v)); } };
    template<> struct StaticCastHelper<short         , short         > { static Vc_ALWAYS_INLINE Vc_CONST m128i cast(param128i v) { return v; } };
    template<> struct StaticCastHelper<unsigned short, short         > { static Vc_ALWAYS_INLINE Vc_CONST m128i cast(param128i v) { return v; } };
    template<> struct StaticCastHelper<unsigned int  , unsigned short> { static Vc_ALWAYS_INLINE Vc_CONST m128i cast(param256i v) { return _mm_packus_epi32(lo128(v), hi128(v)); } };
    template<> struct StaticCastHelper<float         , unsigned short> { static Vc_ALWAYS_INLINE Vc_CONST m128i cast(param256  v) { return StaticCastHelper<unsigned int, unsigned short>::cast(StaticCastHelper<float, unsigned int>::cast(v)); } };
    template<> struct StaticCastHelper<short         , unsigned short> { static Vc_ALWAYS_INLINE Vc_CONST m128i cast(param128i v) { return v; } };
    template<> struct StaticCastHelper<unsigned short, unsigned short> { static Vc_ALWAYS_INLINE Vc_CONST m128i cast(param128i v) { return v; } };
    template<> struct StaticCastHelper<sfloat        , short         > { static Vc_ALWAYS_INLINE Vc_CONST m128i cast(param256  v) { return StaticCastHelper<int, short>::cast(StaticCastHelper<float, int>::cast(v)); } };
    template<> struct StaticCastHelper<sfloat        , unsigned short> { static Vc_ALWAYS_INLINE Vc_CONST m128i cast(param256  v) { return StaticCastHelper<unsigned int, unsigned short>::cast(StaticCastHelper<float, unsigned int>::cast(v)); } };
    template<> struct StaticCastHelper<short         , sfloat        > { static Vc_ALWAYS_INLINE Vc_CONST m256  cast(param128i v) { return _mm256_cvtepi32_ps(StaticCastHelper<short, int>::cast(v)); } };
    template<> struct StaticCastHelper<unsigned short, sfloat        > { static Vc_ALWAYS_INLINE Vc_CONST m256  cast(param128i v) { return _mm256_cvtepi32_ps(StaticCastHelper<unsigned short, unsigned int>::cast(v)); } };
} // namespace AVX
} // namespace Vc
} // namespace ROOT

#include "undomacros.h"

#endif // AVX_CASTS_H
