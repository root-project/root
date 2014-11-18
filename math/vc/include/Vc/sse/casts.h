/*  This file is part of the Vc library.

    Copyright (C) 2009-2011 Matthias Kretz <kretz@kde.org>

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

#ifndef SSE_CASTS_H
#define SSE_CASTS_H

#include "intrinsics.h"
#include "types.h"

namespace ROOT {
namespace Vc
{
namespace SSE
{
    template<typename To, typename From> static Vc_ALWAYS_INLINE To Vc_CONST mm128_reinterpret_cast(VC_ALIGNED_PARAMETER(From) v) { return v; }
    template<> Vc_ALWAYS_INLINE _M128I Vc_CONST mm128_reinterpret_cast<_M128I, _M128 >(VC_ALIGNED_PARAMETER(_M128 ) v) { return _mm_castps_si128(v); }
    template<> Vc_ALWAYS_INLINE _M128I Vc_CONST mm128_reinterpret_cast<_M128I, _M128D>(VC_ALIGNED_PARAMETER(_M128D) v) { return _mm_castpd_si128(v); }
    template<> Vc_ALWAYS_INLINE _M128  Vc_CONST mm128_reinterpret_cast<_M128 , _M128D>(VC_ALIGNED_PARAMETER(_M128D) v) { return _mm_castpd_ps(v);    }
    template<> Vc_ALWAYS_INLINE _M128  Vc_CONST mm128_reinterpret_cast<_M128 , _M128I>(VC_ALIGNED_PARAMETER(_M128I) v) { return _mm_castsi128_ps(v); }
    template<> Vc_ALWAYS_INLINE _M128D Vc_CONST mm128_reinterpret_cast<_M128D, _M128I>(VC_ALIGNED_PARAMETER(_M128I) v) { return _mm_castsi128_pd(v); }
    template<> Vc_ALWAYS_INLINE _M128D Vc_CONST mm128_reinterpret_cast<_M128D, _M128 >(VC_ALIGNED_PARAMETER(_M128 ) v) { return _mm_castps_pd(v);    }
    template<typename To, typename From> static Vc_ALWAYS_INLINE To Vc_CONST sse_cast(VC_ALIGNED_PARAMETER(From) v) { return mm128_reinterpret_cast<To, From>(v); }

    template<typename From, typename To> struct StaticCastHelper {};
    template<> struct StaticCastHelper<float       , int         > { static Vc_ALWAYS_INLINE _M128I cast(const _M128  &v) { return _mm_cvttps_epi32(v); } };
    template<> struct StaticCastHelper<double      , int         > { static Vc_ALWAYS_INLINE _M128I cast(const _M128D &v) { return _mm_cvttpd_epi32(v); } };
    template<> struct StaticCastHelper<int         , int         > { static Vc_ALWAYS_INLINE _M128I cast(const _M128I &v) { return v; } };
    template<> struct StaticCastHelper<unsigned int, int         > { static Vc_ALWAYS_INLINE _M128I cast(const _M128I &v) { return v; } };
    template<> struct StaticCastHelper<float       , unsigned int> { static Vc_ALWAYS_INLINE _M128I cast(const _M128  &v) {
        return _mm_castps_si128(mm_blendv_ps(
                _mm_castsi128_ps(_mm_cvttps_epi32(v)),
                _mm_castsi128_ps(_mm_add_epi32(_mm_cvttps_epi32(_mm_sub_ps(v, _mm_set1_ps(1u << 31))), _mm_set1_epi32(1 << 31))),
                _mm_cmpge_ps(v, _mm_set1_ps(1u << 31))
                ));

    } };
    template<> struct StaticCastHelper<double      , unsigned int> { static Vc_ALWAYS_INLINE _M128I cast(const _M128D &v) { return _mm_cvttpd_epi32(v); } };
    template<> struct StaticCastHelper<int         , unsigned int> { static Vc_ALWAYS_INLINE _M128I cast(const _M128I &v) { return v; } };
    template<> struct StaticCastHelper<unsigned int, unsigned int> { static Vc_ALWAYS_INLINE _M128I cast(const _M128I &v) { return v; } };
    template<> struct StaticCastHelper<float       , float       > { static Vc_ALWAYS_INLINE _M128  cast(const _M128  &v) { return v; } };
    template<> struct StaticCastHelper<double      , float       > { static Vc_ALWAYS_INLINE _M128  cast(const _M128D &v) { return _mm_cvtpd_ps(v); } };
    template<> struct StaticCastHelper<int         , float       > { static Vc_ALWAYS_INLINE _M128  cast(const _M128I &v) { return _mm_cvtepi32_ps(v); } };
    template<> struct StaticCastHelper<unsigned int, float       > { static Vc_ALWAYS_INLINE _M128  cast(const _M128I &v) {
        return mm_blendv_ps(
                _mm_cvtepi32_ps(v),
                _mm_add_ps(_mm_cvtepi32_ps(_mm_sub_epi32(v, _mm_set1_epi32(1 << 31))), _mm_set1_ps(1u << 31)),
                _mm_castsi128_ps(_mm_cmplt_epi32(v, _mm_setzero_si128()))
                );
    } };
    template<> struct StaticCastHelper<float       , double      > { static Vc_ALWAYS_INLINE _M128D cast(const _M128  &v) { return _mm_cvtps_pd(v); } };
    template<> struct StaticCastHelper<double      , double      > { static Vc_ALWAYS_INLINE _M128D cast(const _M128D &v) { return v; } };
    template<> struct StaticCastHelper<int         , double      > { static Vc_ALWAYS_INLINE _M128D cast(const _M128I &v) { return _mm_cvtepi32_pd(v); } };
    template<> struct StaticCastHelper<unsigned int, double      > { static Vc_ALWAYS_INLINE _M128D cast(const _M128I &v) { return _mm_cvtepi32_pd(v); } };

    template<> struct StaticCastHelper<unsigned short, float8        > { static Vc_ALWAYS_INLINE  M256  cast(const _M128I &v) {
        return M256::create(_mm_cvtepi32_ps(_mm_unpacklo_epi16(v, _mm_setzero_si128())),
                    _mm_cvtepi32_ps(_mm_unpackhi_epi16(v, _mm_setzero_si128())));
    } };
    template<> struct StaticCastHelper<short         , float8        > { static Vc_ALWAYS_INLINE  M256  cast(const _M128I &v) {
        const _M128I neg = _mm_cmplt_epi16(v, _mm_setzero_si128());
        return M256::create(_mm_cvtepi32_ps(_mm_unpacklo_epi16(v, neg)),
                    _mm_cvtepi32_ps(_mm_unpackhi_epi16(v, neg)));
    } };
    template<> struct StaticCastHelper<float8        , short         > { static Vc_ALWAYS_INLINE _M128I cast(const  M256  &v) { return _mm_packs_epi32(_mm_cvttps_epi32(v[0]), _mm_cvttps_epi32(v[1])); } };
#ifdef VC_IMPL_SSE4_1
    template<> struct StaticCastHelper<float8        , unsigned short> { static Vc_ALWAYS_INLINE _M128I cast(const  M256  &v) { return _mm_packus_epi32(_mm_cvttps_epi32(v[0]), _mm_cvttps_epi32(v[1])); } };
#else
    template<> struct StaticCastHelper<float8        , unsigned short> { static Vc_ALWAYS_INLINE _M128I cast(const  M256  &v) {
        return _mm_add_epi16(_mm_set1_epi16(-32768),
                _mm_packs_epi32(
                    _mm_add_epi32(_mm_set1_epi32(-32768), _mm_cvttps_epi32(v[0])),
                    _mm_add_epi32(_mm_set1_epi32(-32768), _mm_cvttps_epi32(v[1]))
                    )
                );
    } };
#endif

    template<> struct StaticCastHelper<float         , short         > { static Vc_ALWAYS_INLINE _M128I cast(const _M128  &v) { return _mm_packs_epi32(_mm_cvttps_epi32(v), _mm_setzero_si128()); } };
    template<> struct StaticCastHelper<short         , short         > { static Vc_ALWAYS_INLINE _M128I cast(const _M128I &v) { return v; } };
    template<> struct StaticCastHelper<unsigned short, short         > { static Vc_ALWAYS_INLINE _M128I cast(const _M128I &v) { return v; } };
    template<> struct StaticCastHelper<float         , unsigned short> { static Vc_ALWAYS_INLINE _M128I cast(const _M128  &v) { return _mm_packs_epi32(_mm_cvttps_epi32(v), _mm_setzero_si128()); } };
    template<> struct StaticCastHelper<short         , unsigned short> { static Vc_ALWAYS_INLINE _M128I cast(const _M128I &v) { return v; } };
    template<> struct StaticCastHelper<unsigned short, unsigned short> { static Vc_ALWAYS_INLINE _M128I cast(const _M128I &v) { return v; } };
} // namespace SSE
} // namespace Vc
} // namespace ROOT

#endif // SSE_CASTS_H
