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

#ifndef VC_SSE_INTERLEAVEDMEMORY_TCC
#define VC_SSE_INTERLEAVEDMEMORY_TCC

#include "macros.h"

namespace ROOT {
namespace Vc
{
namespace Common
{

namespace
{
template<typename V, int Size> struct InterleaveImpl;
template<> struct InterleaveImpl<SSE::sfloat_v, 8> {
    static inline void interleave(float *const data, const SSE::sfloat_v::IndexType &i,/*{{{*/
            const SSE::sfloat_v::AsArg v0, const SSE::sfloat_v::AsArg v1)
    {
        const __m128 tmp0 = _mm_unpacklo_ps(v0.data()[0], v1.data()[0]);
        const __m128 tmp1 = _mm_unpackhi_ps(v0.data()[0], v1.data()[0]);
        const __m128 tmp2 = _mm_unpacklo_ps(v0.data()[1], v1.data()[1]);
        const __m128 tmp3 = _mm_unpackhi_ps(v0.data()[1], v1.data()[1]);

        _mm_storel_pi(reinterpret_cast<__m64 *>(&data[i[0]]), tmp0);
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[1]]), tmp0);
        _mm_storel_pi(reinterpret_cast<__m64 *>(&data[i[2]]), tmp1);
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[3]]), tmp1);
        _mm_storel_pi(reinterpret_cast<__m64 *>(&data[i[4]]), tmp2);
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[5]]), tmp2);
        _mm_storel_pi(reinterpret_cast<__m64 *>(&data[i[6]]), tmp3);
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[7]]), tmp3);
    }/*}}}*/
    static inline void interleave(float *const data, const SSE::sfloat_v::IndexType &i,/*{{{*/
            const SSE::sfloat_v::AsArg v0, const SSE::sfloat_v::AsArg v1, const SSE::sfloat_v::AsArg v2)
    {
#ifdef VC_USE_MASKMOV_SCATTER
        const __m128i mask = _mm_set_epi32(0, -1, -1, -1);

        const __m128 tmp0 = _mm_unpacklo_ps(v0.data()[0], v1.data()[0]);
        const __m128 tmp1 = _mm_unpackhi_ps(v0.data()[0], v1.data()[0]);
        const __m128 tmp2 = _mm_unpacklo_ps(v2.data()[0], v2.data()[0]);
        const __m128 tmp3 = _mm_unpackhi_ps(v2.data()[0], v2.data()[0]);
        _mm_maskmoveu_si128(_mm_castps_si128(_mm_movelh_ps(tmp0, tmp2)), mask, reinterpret_cast<char *>(&data[i[0]]));
        _mm_maskmoveu_si128(_mm_castps_si128(_mm_movehl_ps(tmp2, tmp0)), mask, reinterpret_cast<char *>(&data[i[1]]));
        _mm_maskmoveu_si128(_mm_castps_si128(_mm_movelh_ps(tmp1, tmp3)), mask, reinterpret_cast<char *>(&data[i[2]]));
        _mm_maskmoveu_si128(_mm_castps_si128(_mm_movehl_ps(tmp3, tmp1)), mask, reinterpret_cast<char *>(&data[i[3]]));

        const __m128 tmp8 = _mm_unpacklo_ps(v0.data()[1], v1.data()[1]);
        const __m128 tmp9 = _mm_unpackhi_ps(v0.data()[1], v1.data()[1]);
        const __m128 tmp10 = _mm_unpacklo_ps(v2.data()[1], v2.data()[1]);
        const __m128 tmp11 = _mm_unpackhi_ps(v2.data()[1], v2.data()[1]);
        _mm_maskmoveu_si128(_mm_castps_si128(_mm_movelh_ps(tmp8, tmp10)), mask, reinterpret_cast<char *>(&data[i[4]]));
        _mm_maskmoveu_si128(_mm_castps_si128(_mm_movehl_ps(tmp10, tmp8)), mask, reinterpret_cast<char *>(&data[i[5]]));
        _mm_maskmoveu_si128(_mm_castps_si128(_mm_movelh_ps(tmp9, tmp11)), mask, reinterpret_cast<char *>(&data[i[6]]));
        _mm_maskmoveu_si128(_mm_castps_si128(_mm_movehl_ps(tmp11, tmp9)), mask, reinterpret_cast<char *>(&data[i[7]]));
#else
        interleave(data, i, v0, v1);
        v2.scatter(data + 2, i);
#endif
    }/*}}}*/
    static inline void interleave(float *const data, const SSE::sfloat_v::IndexType &i,/*{{{*/
            const SSE::sfloat_v::AsArg v0, const SSE::sfloat_v::AsArg v1,
            const SSE::sfloat_v::AsArg v2, const SSE::sfloat_v::AsArg v3)
    {
        const __m128 tmp0 = _mm_unpacklo_ps(v0.data()[0], v1.data()[0]);
        const __m128 tmp1 = _mm_unpackhi_ps(v0.data()[0], v1.data()[0]);
        const __m128 tmp2 = _mm_unpacklo_ps(v2.data()[0], v3.data()[0]);
        const __m128 tmp3 = _mm_unpackhi_ps(v2.data()[0], v3.data()[0]);
        _mm_storeu_ps(&data[i[0]], _mm_movelh_ps(tmp0, tmp2));
        _mm_storeu_ps(&data[i[1]], _mm_movehl_ps(tmp2, tmp0));
        _mm_storeu_ps(&data[i[2]], _mm_movelh_ps(tmp1, tmp3));
        _mm_storeu_ps(&data[i[3]], _mm_movehl_ps(tmp3, tmp1));

        const __m128 tmp8 = _mm_unpacklo_ps(v0.data()[1], v1.data()[1]);
        const __m128 tmp9 = _mm_unpackhi_ps(v0.data()[1], v1.data()[1]);
        const __m128 tmp10 = _mm_unpacklo_ps(v2.data()[1], v3.data()[1]);
        const __m128 tmp11 = _mm_unpackhi_ps(v2.data()[1], v3.data()[1]);
        _mm_storeu_ps(&data[i[4]], _mm_movelh_ps(tmp8, tmp10));
        _mm_storeu_ps(&data[i[5]], _mm_movehl_ps(tmp10, tmp8));
        _mm_storeu_ps(&data[i[6]], _mm_movelh_ps(tmp9, tmp11));
        _mm_storeu_ps(&data[i[7]], _mm_movehl_ps(tmp11, tmp9));
    }/*}}}*/
};
template<typename V> struct InterleaveImpl<V, 8> {
    static inline void interleave(typename V::EntryType *const data, const typename V::IndexType &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1)
    {
        const __m128i tmp0 = _mm_unpacklo_epi16(v0.data(), v1.data());
        const __m128i tmp1 = _mm_unpackhi_epi16(v0.data(), v1.data());
#ifdef __x86_64__
        const long long tmp00 = _mm_cvtsi128_si64(tmp0);
        const long long tmp01 = _mm_cvtsi128_si64(_mm_unpackhi_epi64(tmp0, tmp0));
        const long long tmp10 = _mm_cvtsi128_si64(tmp1);
        const long long tmp11 = _mm_cvtsi128_si64(_mm_unpackhi_epi64(tmp1, tmp1));
        *reinterpret_cast<int *>(&data[i[0]]) = tmp00;
        *reinterpret_cast<int *>(&data[i[1]]) = tmp00 >> 32;
        *reinterpret_cast<int *>(&data[i[2]]) = tmp01;
        *reinterpret_cast<int *>(&data[i[3]]) = tmp01 >> 32;
        *reinterpret_cast<int *>(&data[i[4]]) = tmp10;
        *reinterpret_cast<int *>(&data[i[5]]) = tmp10 >> 32;
        *reinterpret_cast<int *>(&data[i[6]]) = tmp11;
        *reinterpret_cast<int *>(&data[i[7]]) = tmp11 >> 32;
#elif defined(VC_IMPL_SSE4_1)
        *reinterpret_cast<int *>(&data[i[0]]) = _mm_cvtsi128_si32(tmp0);
        *reinterpret_cast<int *>(&data[i[1]]) = _mm_extract_epi32(tmp0, 1);
        *reinterpret_cast<int *>(&data[i[2]]) = _mm_extract_epi32(tmp0, 2);
        *reinterpret_cast<int *>(&data[i[3]]) = _mm_extract_epi32(tmp0, 3);
        *reinterpret_cast<int *>(&data[i[4]]) = _mm_cvtsi128_si32(tmp1);
        *reinterpret_cast<int *>(&data[i[5]]) = _mm_extract_epi32(tmp1, 1);
        *reinterpret_cast<int *>(&data[i[6]]) = _mm_extract_epi32(tmp1, 2);
        *reinterpret_cast<int *>(&data[i[7]]) = _mm_extract_epi32(tmp1, 3);
#else
        *reinterpret_cast<int *>(&data[i[0]]) = _mm_cvtsi128_si32(tmp0);
        *reinterpret_cast<int *>(&data[i[1]]) = _mm_cvtsi128_si32(_mm_srli_si128(tmp0, 4));
        *reinterpret_cast<int *>(&data[i[2]]) = _mm_cvtsi128_si32(_mm_srli_si128(tmp0, 8));
        *reinterpret_cast<int *>(&data[i[3]]) = _mm_cvtsi128_si32(_mm_srli_si128(tmp0, 12));
        *reinterpret_cast<int *>(&data[i[4]]) = _mm_cvtsi128_si32(tmp1);
        *reinterpret_cast<int *>(&data[i[5]]) = _mm_cvtsi128_si32(_mm_srli_si128(tmp1, 4));
        *reinterpret_cast<int *>(&data[i[6]]) = _mm_cvtsi128_si32(_mm_srli_si128(tmp1, 8));
        *reinterpret_cast<int *>(&data[i[7]]) = _mm_cvtsi128_si32(_mm_srli_si128(tmp1, 12));
#endif
    }/*}}}*/
    static inline void interleave(typename V::EntryType *const data, const typename V::IndexType &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2)
    {
#ifdef VC_USE_MASKMOV_SCATTER
        const __m128i maskLo = _mm_set_epi16(0, 0, 0, 0, 0, -1, -1, -1);
        const __m128i maskHi = _mm_set_epi16(0, -1, -1, -1, 0, 0, 0, 0);
        typename V::EntryType *const dataHi = data - 4;
        const __m128i tmp0 = _mm_unpacklo_epi16(v0.data(), v2.data());
        const __m128i tmp1 = _mm_unpackhi_epi16(v0.data(), v2.data());
        const __m128i tmp2 = _mm_unpacklo_epi16(v1.data(), v1.data());
        const __m128i tmp3 = _mm_unpackhi_epi16(v1.data(), v1.data());

        const __m128i tmp4 = _mm_unpacklo_epi16(tmp0, tmp2);
        const __m128i tmp5 = _mm_unpackhi_epi16(tmp0, tmp2);
        const __m128i tmp6 = _mm_unpacklo_epi16(tmp1, tmp3);
        const __m128i tmp7 = _mm_unpackhi_epi16(tmp1, tmp3);
        _mm_maskmoveu_si128(tmp4, maskLo, reinterpret_cast<char *>(&data[i[0]]));
        _mm_maskmoveu_si128(tmp4, maskHi, reinterpret_cast<char *>(&dataHi[i[1]]));
        _mm_maskmoveu_si128(tmp5, maskLo, reinterpret_cast<char *>(&data[i[2]]));
        _mm_maskmoveu_si128(tmp5, maskHi, reinterpret_cast<char *>(&dataHi[i[3]]));
        _mm_maskmoveu_si128(tmp6, maskLo, reinterpret_cast<char *>(&data[i[4]]));
        _mm_maskmoveu_si128(tmp6, maskHi, reinterpret_cast<char *>(&dataHi[i[5]]));
        _mm_maskmoveu_si128(tmp7, maskLo, reinterpret_cast<char *>(&data[i[6]]));
        _mm_maskmoveu_si128(tmp7, maskHi, reinterpret_cast<char *>(&dataHi[i[7]]));
#else
        interleave(data, i, v0, v1);
        v2.scatter(data + 2, i);
#endif
    }/*}}}*/
    static inline void interleave(typename V::EntryType *const data, const typename V::IndexType &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1,
            const typename V::AsArg v2, const typename V::AsArg v3)
    {
        const __m128i tmp0 = _mm_unpacklo_epi16(v0.data(), v2.data());
        const __m128i tmp1 = _mm_unpackhi_epi16(v0.data(), v2.data());
        const __m128i tmp2 = _mm_unpacklo_epi16(v1.data(), v3.data());
        const __m128i tmp3 = _mm_unpackhi_epi16(v1.data(), v3.data());

        const __m128i tmp4 = _mm_unpacklo_epi16(tmp0, tmp2);
        const __m128i tmp5 = _mm_unpackhi_epi16(tmp0, tmp2);
        const __m128i tmp6 = _mm_unpacklo_epi16(tmp1, tmp3);
        const __m128i tmp7 = _mm_unpackhi_epi16(tmp1, tmp3);

        _mm_storel_epi64(reinterpret_cast<__m128i *>(&data[i[0]]), tmp4);
        _mm_storel_epi64(reinterpret_cast<__m128i *>(&data[i[2]]), tmp5);
        _mm_storel_epi64(reinterpret_cast<__m128i *>(&data[i[4]]), tmp6);
        _mm_storel_epi64(reinterpret_cast<__m128i *>(&data[i[6]]), tmp7);
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[1]]), _mm_castsi128_ps(tmp4));
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[3]]), _mm_castsi128_ps(tmp5));
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[5]]), _mm_castsi128_ps(tmp6));
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[7]]), _mm_castsi128_ps(tmp7));
    }/*}}}*/
};
template<typename V> struct InterleaveImpl<V, 4> {
    static inline void interleave(typename V::EntryType *const data, const typename V::IndexType &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1)
    {
        const __m128 tmp0 = _mm_unpacklo_ps(SSE::sse_cast<__m128>(v0.data()),SSE::sse_cast<__m128>(v1.data()));
        const __m128 tmp1 = _mm_unpackhi_ps(SSE::sse_cast<__m128>(v0.data()),SSE::sse_cast<__m128>(v1.data()));
        _mm_storel_pi(reinterpret_cast<__m64 *>(&data[i[0]]), tmp0);
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[1]]), tmp0);
        _mm_storel_pi(reinterpret_cast<__m64 *>(&data[i[2]]), tmp1);
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[3]]), tmp1);
    }/*}}}*/
    static inline void interleave(typename V::EntryType *const data, const typename V::IndexType &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2)
    {
#ifdef VC_USE_MASKMOV_SCATTER
        const __m128 tmp0 = _mm_unpacklo_ps(SSE::sse_cast<__m128>(v0.data()), SSE::sse_cast<__m128>(v1.data()));
        const __m128 tmp1 = _mm_unpackhi_ps(SSE::sse_cast<__m128>(v0.data()), SSE::sse_cast<__m128>(v1.data()));
        const __m128 tmp2 = _mm_unpacklo_ps(SSE::sse_cast<__m128>(v2.data()), SSE::sse_cast<__m128>(v2.data()));
        const __m128 tmp3 = _mm_unpackhi_ps(SSE::sse_cast<__m128>(v2.data()), SSE::sse_cast<__m128>(v2.data()));
        const __m128i mask = _mm_set_epi32(0, -1, -1, -1);
        _mm_maskmoveu_si128(_mm_castps_si128(_mm_movelh_ps(tmp0, tmp2)), mask, reinterpret_cast<char *>(&data[i[0]]));
        _mm_maskmoveu_si128(_mm_castps_si128(_mm_movehl_ps(tmp2, tmp0)), mask, reinterpret_cast<char *>(&data[i[1]]));
        _mm_maskmoveu_si128(_mm_castps_si128(_mm_movelh_ps(tmp1, tmp3)), mask, reinterpret_cast<char *>(&data[i[2]]));
        _mm_maskmoveu_si128(_mm_castps_si128(_mm_movehl_ps(tmp3, tmp1)), mask, reinterpret_cast<char *>(&data[i[3]]));
#else
        const __m128 tmp0 = _mm_unpacklo_ps(SSE::sse_cast<__m128>(v0.data()),SSE::sse_cast<__m128>(v1.data()));
        const __m128 tmp1 = _mm_unpackhi_ps(SSE::sse_cast<__m128>(v0.data()),SSE::sse_cast<__m128>(v1.data()));
        _mm_storel_pi(reinterpret_cast<__m64 *>(&data[i[0]]), tmp0);
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[1]]), tmp0);
        _mm_storel_pi(reinterpret_cast<__m64 *>(&data[i[2]]), tmp1);
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[3]]), tmp1);
        v2.scatter(data + 2, i);
#endif
    }/*}}}*/
    static inline void interleave(typename V::EntryType *const data, const typename V::IndexType &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1,
            const typename V::AsArg v2, const typename V::AsArg v3)
    {
        const __m128 tmp0 = _mm_unpacklo_ps(SSE::sse_cast<__m128>(v0.data()),SSE::sse_cast<__m128>(v1.data()));
        const __m128 tmp1 = _mm_unpackhi_ps(SSE::sse_cast<__m128>(v0.data()),SSE::sse_cast<__m128>(v1.data()));
        const __m128 tmp2 = _mm_unpacklo_ps(SSE::sse_cast<__m128>(v2.data()),SSE::sse_cast<__m128>(v3.data()));
        const __m128 tmp3 = _mm_unpackhi_ps(SSE::sse_cast<__m128>(v2.data()),SSE::sse_cast<__m128>(v3.data()));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[0]]), _mm_movelh_ps(tmp0, tmp2));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[1]]), _mm_movehl_ps(tmp2, tmp0));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[2]]), _mm_movelh_ps(tmp1, tmp3));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[3]]), _mm_movehl_ps(tmp3, tmp1));
    }/*}}}*/
};
template<typename V> struct InterleaveImpl<V, 2> {
    static inline void interleave(typename V::EntryType *const data, const typename V::IndexType &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1)
    {
        const __m128d tmp0 = _mm_unpacklo_pd(v0.data(), v1.data());
        const __m128d tmp1 = _mm_unpackhi_pd(v0.data(), v1.data());
        _mm_storeu_pd(&data[i[0]], tmp0);
        _mm_storeu_pd(&data[i[1]], tmp1);
    }/*}}}*/
    static inline void interleave(typename V::EntryType *const data, const typename V::IndexType &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2)
    {
        interleave(data, i, v0, v1);
        v2.scatter(data + 2, i);
    }/*}}}*/
    static inline void interleave(typename V::EntryType *const data, const typename V::IndexType &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1,
            const typename V::AsArg v2, const typename V::AsArg v3)
    {
        interleave(data, i, v0, v1);
        interleave(data + 2, i, v2, v3);
    }/*}}}*/
};
} // anonymous namespace

template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1)
{
    InterleaveImpl<V, V::Size>::interleave(m_data, m_indexes, v0, v1);
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2)
{
    InterleaveImpl<V, V::Size>::interleave(m_data, m_indexes, v0, v1, v2);
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2, const typename V::AsArg v3)
{
    InterleaveImpl<V, V::Size>::interleave(m_data, m_indexes, v0, v1, v2, v3);
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2, const typename V::AsArg v3, const typename V::AsArg v4)
{
    InterleaveImpl<V, V::Size>::interleave(m_data, m_indexes, v0, v1, v2, v3);
    v4.scatter(m_data + 4, m_indexes);
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2, const typename V::AsArg v3, const typename V::AsArg v4,
        const typename V::AsArg v5)
{
    InterleaveImpl<V, V::Size>::interleave(m_data    , m_indexes, v0, v1, v2, v3);
    InterleaveImpl<V, V::Size>::interleave(m_data + 4, m_indexes, v4, v5);
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2, const typename V::AsArg v3, const typename V::AsArg v4,
        const typename V::AsArg v5, const typename V::AsArg v6)
{
    InterleaveImpl<V, V::Size>::interleave(m_data + 0, m_indexes, v0, v1, v2, v3);
    InterleaveImpl<V, V::Size>::interleave(m_data + 4, m_indexes, v4, v5, v6);
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2, const typename V::AsArg v3, const typename V::AsArg v4,
        const typename V::AsArg v5, const typename V::AsArg v6, const typename V::AsArg v7)
{
    InterleaveImpl<V, V::Size>::interleave(m_data + 0, m_indexes, v0, v1, v2, v3);
    InterleaveImpl<V, V::Size>::interleave(m_data + 4, m_indexes, v4, v5, v6, v7);
}/*}}}*/

template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1) const/*{{{*/
{
    const __m128 a = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&m_data[m_indexes[0]])));
    const __m128 b = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&m_data[m_indexes[1]])));
    const __m128 c = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&m_data[m_indexes[2]])));
    const __m128 d = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&m_data[m_indexes[3]])));

    const __m128 tmp0 = _mm_unpacklo_ps(a, b); // [a0 a1 b0 b1]
    const __m128 tmp1 = _mm_unpacklo_ps(c, d); // [a2 a3 b2 b3]

    v0.data() = _mm_movelh_ps(tmp0, tmp1);
    v1.data() = _mm_movehl_ps(tmp1, tmp0);
}
/*}}}*/
template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2) const/*{{{*/
{
    const __m128 a = _mm_loadu_ps(&m_data[m_indexes[0]]);
    const __m128 b = _mm_loadu_ps(&m_data[m_indexes[1]]);
    const __m128 c = _mm_loadu_ps(&m_data[m_indexes[2]]);
    const __m128 d = _mm_loadu_ps(&m_data[m_indexes[3]]);

    const __m128 tmp0 = _mm_unpacklo_ps(a, b); // [a0 a1 b0 b1]
    const __m128 tmp1 = _mm_unpacklo_ps(c, d); // [a2 a3 b2 b3]
    const __m128 tmp2 = _mm_unpackhi_ps(a, b); // [c0 c1 XX XX]
    const __m128 tmp3 = _mm_unpackhi_ps(c, d); // [c2 c3 XX XX]

    v0.data() = _mm_movelh_ps(tmp0, tmp1);
    v1.data() = _mm_movehl_ps(tmp1, tmp0);
    v2.data() = _mm_movelh_ps(tmp2, tmp3);
}
/*}}}*/
template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3) const/*{{{*/
{
    const __m128 a = _mm_loadu_ps(&m_data[m_indexes[0]]);
    const __m128 b = _mm_loadu_ps(&m_data[m_indexes[1]]);
    const __m128 c = _mm_loadu_ps(&m_data[m_indexes[2]]);
    const __m128 d = _mm_loadu_ps(&m_data[m_indexes[3]]);

    const __m128 tmp0 = _mm_unpacklo_ps(a, b); // [a0 a1 b0 b1]
    const __m128 tmp1 = _mm_unpacklo_ps(c, d); // [a2 a3 b2 b3]
    const __m128 tmp2 = _mm_unpackhi_ps(a, b); // [c0 c1 d0 d1]
    const __m128 tmp3 = _mm_unpackhi_ps(c, d); // [c2 c3 d2 d3]

    v0.data() = _mm_movelh_ps(tmp0, tmp1);
    v1.data() = _mm_movehl_ps(tmp1, tmp0);
    v2.data() = _mm_movelh_ps(tmp2, tmp3);
    v3.data() = _mm_movehl_ps(tmp3, tmp2);
}
/*}}}*/
template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3, float_v &v4) const/*{{{*/
{
    v4.gather(m_data, m_indexes + I(4));
    deinterleave(v0, v1, v2, v3);
}
/*}}}*/
template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3, float_v &v4, float_v &v5) const/*{{{*/
{
    const __m128 a = _mm_loadu_ps(&m_data[m_indexes[0]]);
    const __m128 e = _mm_loadu_ps(&m_data[4 + m_indexes[0]]);
    const __m128 b = _mm_loadu_ps(&m_data[m_indexes[1]]);
    const __m128 f = _mm_loadu_ps(&m_data[4 + m_indexes[1]]);

    const __m128 tmp0 = _mm_unpacklo_ps(a, b); // [a0 a1 b0 b1]
    const __m128 tmp2 = _mm_unpackhi_ps(a, b); // [c0 c1 d0 d1]
    const __m128 tmp4 = _mm_unpacklo_ps(e, f); // [a0 a1 b0 b1]

    const __m128 c = _mm_loadu_ps(&m_data[m_indexes[2]]);
    const __m128 g = _mm_loadu_ps(&m_data[4 + m_indexes[2]]);
    const __m128 d = _mm_loadu_ps(&m_data[m_indexes[3]]);
    const __m128 h = _mm_loadu_ps(&m_data[4 + m_indexes[3]]);

    const __m128 tmp1 = _mm_unpacklo_ps(c, d); // [a2 a3 b2 b3]
    v0.data() = _mm_movelh_ps(tmp0, tmp1);
    v1.data() = _mm_movehl_ps(tmp1, tmp0);

    const __m128 tmp3 = _mm_unpackhi_ps(c, d); // [c2 c3 d2 d3]
    v2.data() = _mm_movelh_ps(tmp2, tmp3);
    v3.data() = _mm_movehl_ps(tmp3, tmp2);

    const __m128 tmp5 = _mm_unpacklo_ps(g, h); // [a2 a3 b2 b3]
    v4.data() = _mm_movelh_ps(tmp4, tmp5);
    v5.data() = _mm_movehl_ps(tmp5, tmp4);
}
/*}}}*/
template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3, float_v &v4, float_v &v5, float_v &v6) const/*{{{*/
{
    const __m128 a = _mm_loadu_ps(&m_data[m_indexes[0]]);
    const __m128 e = _mm_loadu_ps(&m_data[4 + m_indexes[0]]);
    const __m128 b = _mm_loadu_ps(&m_data[m_indexes[1]]);
    const __m128 f = _mm_loadu_ps(&m_data[4 + m_indexes[1]]);

    const __m128 tmp0 = _mm_unpacklo_ps(a, b); // [a0 a1 b0 b1]
    const __m128 tmp2 = _mm_unpackhi_ps(a, b); // [c0 c1 d0 d1]
    const __m128 tmp4 = _mm_unpacklo_ps(e, f); // [a0 a1 b0 b1]
    const __m128 tmp6 = _mm_unpackhi_ps(e, f); // [c0 c1 d0 d1]

    const __m128 c = _mm_loadu_ps(&m_data[m_indexes[2]]);
    const __m128 g = _mm_loadu_ps(&m_data[4 + m_indexes[2]]);
    const __m128 d = _mm_loadu_ps(&m_data[m_indexes[3]]);
    const __m128 h = _mm_loadu_ps(&m_data[4 + m_indexes[3]]);

    const __m128 tmp1 = _mm_unpacklo_ps(c, d); // [a2 a3 b2 b3]
    v0.data() = _mm_movelh_ps(tmp0, tmp1);
    v1.data() = _mm_movehl_ps(tmp1, tmp0);

    const __m128 tmp3 = _mm_unpackhi_ps(c, d); // [c2 c3 d2 d3]
    v2.data() = _mm_movelh_ps(tmp2, tmp3);
    v3.data() = _mm_movehl_ps(tmp3, tmp2);

    const __m128 tmp5 = _mm_unpacklo_ps(g, h); // [a2 a3 b2 b3]
    v4.data() = _mm_movelh_ps(tmp4, tmp5);
    v5.data() = _mm_movehl_ps(tmp5, tmp4);

    const __m128 tmp7 = _mm_unpackhi_ps(g, h); // [c2 c3 d2 d3]
    v6.data() = _mm_movelh_ps(tmp6, tmp7);
}
/*}}}*/
template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3, float_v &v4, float_v &v5, float_v &v6, float_v &v7) const/*{{{*/
{
    const __m128 a = _mm_loadu_ps(&m_data[m_indexes[0]]);
    const __m128 e = _mm_loadu_ps(&m_data[4 + m_indexes[0]]);
    const __m128 b = _mm_loadu_ps(&m_data[m_indexes[1]]);
    const __m128 f = _mm_loadu_ps(&m_data[4 + m_indexes[1]]);

    const __m128 tmp0 = _mm_unpacklo_ps(a, b); // [a0 a1 b0 b1]
    const __m128 tmp2 = _mm_unpackhi_ps(a, b); // [c0 c1 d0 d1]
    const __m128 tmp4 = _mm_unpacklo_ps(e, f); // [a0 a1 b0 b1]
    const __m128 tmp6 = _mm_unpackhi_ps(e, f); // [c0 c1 d0 d1]

    const __m128 c = _mm_loadu_ps(&m_data[m_indexes[2]]);
    const __m128 g = _mm_loadu_ps(&m_data[4 + m_indexes[2]]);
    const __m128 d = _mm_loadu_ps(&m_data[m_indexes[3]]);
    const __m128 h = _mm_loadu_ps(&m_data[4 + m_indexes[3]]);

    const __m128 tmp1 = _mm_unpacklo_ps(c, d); // [a2 a3 b2 b3]
    v0.data() = _mm_movelh_ps(tmp0, tmp1);
    v1.data() = _mm_movehl_ps(tmp1, tmp0);

    const __m128 tmp3 = _mm_unpackhi_ps(c, d); // [c2 c3 d2 d3]
    v2.data() = _mm_movelh_ps(tmp2, tmp3);
    v3.data() = _mm_movehl_ps(tmp3, tmp2);

    const __m128 tmp5 = _mm_unpacklo_ps(g, h); // [a2 a3 b2 b3]
    v4.data() = _mm_movelh_ps(tmp4, tmp5);
    v5.data() = _mm_movehl_ps(tmp5, tmp4);

    const __m128 tmp7 = _mm_unpackhi_ps(g, h); // [c2 c3 d2 d3]
    v6.data() = _mm_movelh_ps(tmp6, tmp7);
    v7.data() = _mm_movehl_ps(tmp7, tmp6);
}/*}}}*/

static inline void _sse_deinterleave_double(const double *VC_RESTRICT data, const uint_v &indexes, double_v &v0, double_v &v1)/*{{{*/
{
    const __m128d a = _mm_loadu_pd(&data[indexes[0]]);
    const __m128d b = _mm_loadu_pd(&data[indexes[1]]);

    v0.data() = _mm_unpacklo_pd(a, b);
    v1.data() = _mm_unpackhi_pd(a, b);
}/*}}}*/
template<> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<double_v>::deinterleave(double_v &v0, double_v &v1) const {/*{{{*/
    _sse_deinterleave_double(m_data, m_indexes, v0, v1);
}
/*}}}*/
template<> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<double_v>::deinterleave(double_v &v0, double_v &v1,/*{{{*/
        double_v &v2) const {
    v2.gather(m_data + 2, m_indexes);
    _sse_deinterleave_double(m_data, m_indexes, v0, v1);
}
/*}}}*/
template<> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<double_v>::deinterleave(double_v &v0, double_v &v1,/*{{{*/
        double_v &v2, double_v &v3) const {
    _sse_deinterleave_double(m_data    , m_indexes, v0, v1);
    _sse_deinterleave_double(m_data + 2, m_indexes, v2, v3);
}
/*}}}*/
template<> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<double_v>::deinterleave(double_v &v0, double_v &v1,/*{{{*/
        double_v &v2, double_v &v3, double_v &v4) const {
    v4.gather(m_data + 4, m_indexes);
    _sse_deinterleave_double(m_data    , m_indexes, v0, v1);
    _sse_deinterleave_double(m_data + 2, m_indexes, v2, v3);
}
/*}}}*/
template<> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<double_v>::deinterleave(double_v &v0, double_v &v1,/*{{{*/
        double_v &v2, double_v &v3, double_v &v4, double_v &v5) const {
    _sse_deinterleave_double(m_data    , m_indexes, v0, v1);
    _sse_deinterleave_double(m_data + 2, m_indexes, v2, v3);
    _sse_deinterleave_double(m_data + 4, m_indexes, v4, v5);
}
/*}}}*/
template<> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<double_v>::deinterleave(double_v &v0, double_v &v1,/*{{{*/
        double_v &v2, double_v &v3, double_v &v4, double_v &v5, double_v &v6) const {
    v6.gather(m_data + 6, m_indexes);
    _sse_deinterleave_double(m_data    , m_indexes, v0, v1);
    _sse_deinterleave_double(m_data + 2, m_indexes, v2, v3);
    _sse_deinterleave_double(m_data + 4, m_indexes, v4, v5);
}
/*}}}*/
template<> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<double_v>::deinterleave(double_v &v0, double_v &v1,/*{{{*/
        double_v &v2, double_v &v3, double_v &v4, double_v &v5, double_v &v6, double_v &v7) const {
    _sse_deinterleave_double(m_data    , m_indexes, v0, v1);
    _sse_deinterleave_double(m_data + 2, m_indexes, v2, v3);
    _sse_deinterleave_double(m_data + 4, m_indexes, v4, v5);
    _sse_deinterleave_double(m_data + 6, m_indexes, v6, v7);
}/*}}}*/

template<> inline void InterleavedMemoryAccessBase<short_v>::deinterleave(short_v &v0, short_v &v1) const {/*{{{*/
    const __m128i a = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&m_data[m_indexes[0]]));
    const __m128i b = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&m_data[m_indexes[1]]));
    const __m128i c = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&m_data[m_indexes[2]]));
    const __m128i d = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&m_data[m_indexes[3]]));
    const __m128i e = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&m_data[m_indexes[4]]));
    const __m128i f = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&m_data[m_indexes[5]]));
    const __m128i g = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&m_data[m_indexes[6]]));
    const __m128i h = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&m_data[m_indexes[7]]));

    const __m128i tmp2  = _mm_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4
    const __m128i tmp3  = _mm_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
    const __m128i tmp4  = _mm_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
    const __m128i tmp5  = _mm_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7

    const __m128i tmp0  = _mm_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6
    const __m128i tmp1  = _mm_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7

    v0.data() = _mm_unpacklo_epi16(tmp0, tmp1);
    v1.data() = _mm_unpackhi_epi16(tmp0, tmp1);
}
/*}}}*/
template<> inline void InterleavedMemoryAccessBase<short_v>::deinterleave(short_v &v0, short_v &v1,/*{{{*/
        short_v &v2) const {
    const __m128i a = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[0]]));
    const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[1]]));
    const __m128i c = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[2]]));
    const __m128i d = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[3]]));
    const __m128i e = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[4]]));
    const __m128i f = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[5]]));
    const __m128i g = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[6]]));
    const __m128i h = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[7]]));

    const __m128i tmp2  = _mm_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4
    const __m128i tmp4  = _mm_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
    const __m128i tmp3  = _mm_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
    const __m128i tmp5  = _mm_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7

    const __m128i tmp0  = _mm_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6
    const __m128i tmp1  = _mm_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7
    const __m128i tmp6  = _mm_unpackhi_epi16(tmp2, tmp3); // c0 c2 c4 c6 d0 d2 d4 d6
    const __m128i tmp7  = _mm_unpackhi_epi16(tmp4, tmp5); // c1 c3 c5 c7 d1 d3 d5 d7

    v0.data() = _mm_unpacklo_epi16(tmp0, tmp1);
    v1.data() = _mm_unpackhi_epi16(tmp0, tmp1);
    v2.data() = _mm_unpacklo_epi16(tmp6, tmp7);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<short_v>::deinterleave(short_v &v0, short_v &v1,/*{{{*/
        short_v &v2, short_v &v3) const {
    const __m128i a = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[0]]));
    const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[1]]));
    const __m128i c = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[2]]));
    const __m128i d = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[3]]));
    const __m128i e = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[4]]));
    const __m128i f = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[5]]));
    const __m128i g = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[6]]));
    const __m128i h = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[7]]));

    const __m128i tmp2  = _mm_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4
    const __m128i tmp4  = _mm_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
    const __m128i tmp3  = _mm_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
    const __m128i tmp5  = _mm_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7

    const __m128i tmp0  = _mm_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6
    const __m128i tmp1  = _mm_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7
    const __m128i tmp6  = _mm_unpackhi_epi16(tmp2, tmp3); // c0 c2 c4 c6 d0 d2 d4 d6
    const __m128i tmp7  = _mm_unpackhi_epi16(tmp4, tmp5); // c1 c3 c5 c7 d1 d3 d5 d7

    v0.data() = _mm_unpacklo_epi16(tmp0, tmp1);
    v1.data() = _mm_unpackhi_epi16(tmp0, tmp1);
    v2.data() = _mm_unpacklo_epi16(tmp6, tmp7);
    v3.data() = _mm_unpackhi_epi16(tmp6, tmp7);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<short_v>::deinterleave(short_v &v0, short_v &v1,/*{{{*/
        short_v &v2, short_v &v3, short_v &v4) const {
    const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[0]]));
    const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[1]]));
    const __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[2]]));
    const __m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[3]]));
    const __m128i e = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[4]]));
    const __m128i f = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[5]]));
    const __m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[6]]));
    const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[7]]));

    const __m128i tmp2  = _mm_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4
    const __m128i tmp4  = _mm_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
    const __m128i tmp3  = _mm_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
    const __m128i tmp5  = _mm_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7
    const __m128i tmp10 = _mm_unpackhi_epi16(a, e); // e0 e4 f0 f4 g0 g4 h0 h4
    const __m128i tmp11 = _mm_unpackhi_epi16(c, g); // e1 e5 f1 f5 g1 g5 h1 h5
    const __m128i tmp12 = _mm_unpackhi_epi16(b, f); // e2 e6 f2 f6 g2 g6 h2 h6
    const __m128i tmp13 = _mm_unpackhi_epi16(d, h); // e3 e7 f3 f7 g3 g7 h3 h7

    const __m128i tmp0  = _mm_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6
    const __m128i tmp1  = _mm_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7
    const __m128i tmp6  = _mm_unpackhi_epi16(tmp2, tmp3); // c0 c2 c4 c6 d0 d2 d4 d6
    const __m128i tmp7  = _mm_unpackhi_epi16(tmp4, tmp5); // c1 c3 c5 c7 d1 d3 d5 d7
    const __m128i tmp8  = _mm_unpacklo_epi16(tmp10, tmp11); // e0 e2 e4 e6 f0 f2 f4 f6
    const __m128i tmp9  = _mm_unpacklo_epi16(tmp12, tmp13); // e1 e3 e5 e7 f1 f3 f5 f7

    v0.data() = _mm_unpacklo_epi16(tmp0, tmp1);
    v1.data() = _mm_unpackhi_epi16(tmp0, tmp1);
    v2.data() = _mm_unpacklo_epi16(tmp6, tmp7);
    v3.data() = _mm_unpackhi_epi16(tmp6, tmp7);
    v4.data() = _mm_unpacklo_epi16(tmp8, tmp9);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<short_v>::deinterleave(short_v &v0, short_v &v1,/*{{{*/
        short_v &v2, short_v &v3, short_v &v4, short_v &v5) const {
    const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[0]]));
    const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[1]]));
    const __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[2]]));
    const __m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[3]]));
    const __m128i e = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[4]]));
    const __m128i f = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[5]]));
    const __m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[6]]));
    const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[7]]));

    const __m128i tmp2  = _mm_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4
    const __m128i tmp4  = _mm_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
    const __m128i tmp3  = _mm_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
    const __m128i tmp5  = _mm_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7
    const __m128i tmp10 = _mm_unpackhi_epi16(a, e); // e0 e4 f0 f4 g0 g4 h0 h4
    const __m128i tmp11 = _mm_unpackhi_epi16(c, g); // e1 e5 f1 f5 g1 g5 h1 h5
    const __m128i tmp12 = _mm_unpackhi_epi16(b, f); // e2 e6 f2 f6 g2 g6 h2 h6
    const __m128i tmp13 = _mm_unpackhi_epi16(d, h); // e3 e7 f3 f7 g3 g7 h3 h7

    const __m128i tmp0  = _mm_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6
    const __m128i tmp1  = _mm_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7
    const __m128i tmp6  = _mm_unpackhi_epi16(tmp2, tmp3); // c0 c2 c4 c6 d0 d2 d4 d6
    const __m128i tmp7  = _mm_unpackhi_epi16(tmp4, tmp5); // c1 c3 c5 c7 d1 d3 d5 d7
    const __m128i tmp8  = _mm_unpacklo_epi16(tmp10, tmp11); // e0 e2 e4 e6 f0 f2 f4 f6
    const __m128i tmp9  = _mm_unpacklo_epi16(tmp12, tmp13); // e1 e3 e5 e7 f1 f3 f5 f7

    v0.data() = _mm_unpacklo_epi16(tmp0, tmp1);
    v1.data() = _mm_unpackhi_epi16(tmp0, tmp1);
    v2.data() = _mm_unpacklo_epi16(tmp6, tmp7);
    v3.data() = _mm_unpackhi_epi16(tmp6, tmp7);
    v4.data() = _mm_unpacklo_epi16(tmp8, tmp9);
    v5.data() = _mm_unpackhi_epi16(tmp8, tmp9);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<short_v>::deinterleave(short_v &v0, short_v &v1,/*{{{*/
        short_v &v2, short_v &v3, short_v &v4, short_v &v5, short_v &v6) const {
    const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[0]]));
    const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[1]]));
    const __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[2]]));
    const __m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[3]]));
    const __m128i e = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[4]]));
    const __m128i f = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[5]]));
    const __m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[6]]));
    const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[7]]));

    const __m128i tmp2  = _mm_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4
    const __m128i tmp4  = _mm_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
    const __m128i tmp3  = _mm_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
    const __m128i tmp5  = _mm_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7
    const __m128i tmp10 = _mm_unpackhi_epi16(a, e); // e0 e4 f0 f4 g0 g4 h0 h4
    const __m128i tmp11 = _mm_unpackhi_epi16(c, g); // e1 e5 f1 f5 g1 g5 h1 h5
    const __m128i tmp12 = _mm_unpackhi_epi16(b, f); // e2 e6 f2 f6 g2 g6 h2 h6
    const __m128i tmp13 = _mm_unpackhi_epi16(d, h); // e3 e7 f3 f7 g3 g7 h3 h7

    const __m128i tmp0  = _mm_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6
    const __m128i tmp1  = _mm_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7
    const __m128i tmp6  = _mm_unpackhi_epi16(tmp2, tmp3); // c0 c2 c4 c6 d0 d2 d4 d6
    const __m128i tmp7  = _mm_unpackhi_epi16(tmp4, tmp5); // c1 c3 c5 c7 d1 d3 d5 d7
    const __m128i tmp8  = _mm_unpacklo_epi16(tmp10, tmp11); // e0 e2 e4 e6 f0 f2 f4 f6
    const __m128i tmp9  = _mm_unpacklo_epi16(tmp12, tmp13); // e1 e3 e5 e7 f1 f3 f5 f7
    const __m128i tmp14 = _mm_unpackhi_epi16(tmp10, tmp11); // g0 g2 g4 g6 h0 h2 h4 h6
    const __m128i tmp15 = _mm_unpackhi_epi16(tmp12, tmp13); // g1 g3 g5 g7 h1 h3 h5 h7

    v0.data() = _mm_unpacklo_epi16(tmp0, tmp1);
    v1.data() = _mm_unpackhi_epi16(tmp0, tmp1);
    v2.data() = _mm_unpacklo_epi16(tmp6, tmp7);
    v3.data() = _mm_unpackhi_epi16(tmp6, tmp7);
    v4.data() = _mm_unpacklo_epi16(tmp8, tmp9);
    v5.data() = _mm_unpackhi_epi16(tmp8, tmp9);
    v6.data() = _mm_unpacklo_epi16(tmp14, tmp15);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<short_v>::deinterleave(short_v &v0, short_v &v1,/*{{{*/
        short_v &v2, short_v &v3, short_v &v4, short_v &v5, short_v &v6, short_v &v7) const {
    const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[0]]));
    const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[1]]));
    const __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[2]]));
    const __m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[3]]));
    const __m128i e = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[4]]));
    const __m128i f = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[5]]));
    const __m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[6]]));
    const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[7]]));

    const __m128i tmp2  = _mm_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4
    const __m128i tmp4  = _mm_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
    const __m128i tmp3  = _mm_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
    const __m128i tmp5  = _mm_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7
    const __m128i tmp10 = _mm_unpackhi_epi16(a, e); // e0 e4 f0 f4 g0 g4 h0 h4
    const __m128i tmp11 = _mm_unpackhi_epi16(c, g); // e1 e5 f1 f5 g1 g5 h1 h5
    const __m128i tmp12 = _mm_unpackhi_epi16(b, f); // e2 e6 f2 f6 g2 g6 h2 h6
    const __m128i tmp13 = _mm_unpackhi_epi16(d, h); // e3 e7 f3 f7 g3 g7 h3 h7

    const __m128i tmp0  = _mm_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6
    const __m128i tmp1  = _mm_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7
    const __m128i tmp6  = _mm_unpackhi_epi16(tmp2, tmp3); // c0 c2 c4 c6 d0 d2 d4 d6
    const __m128i tmp7  = _mm_unpackhi_epi16(tmp4, tmp5); // c1 c3 c5 c7 d1 d3 d5 d7
    const __m128i tmp8  = _mm_unpacklo_epi16(tmp10, tmp11); // e0 e2 e4 e6 f0 f2 f4 f6
    const __m128i tmp9  = _mm_unpacklo_epi16(tmp12, tmp13); // e1 e3 e5 e7 f1 f3 f5 f7
    const __m128i tmp14 = _mm_unpackhi_epi16(tmp10, tmp11); // g0 g2 g4 g6 h0 h2 h4 h6
    const __m128i tmp15 = _mm_unpackhi_epi16(tmp12, tmp13); // g1 g3 g5 g7 h1 h3 h5 h7

    v0.data() = _mm_unpacklo_epi16(tmp0, tmp1);
    v1.data() = _mm_unpackhi_epi16(tmp0, tmp1);
    v2.data() = _mm_unpacklo_epi16(tmp6, tmp7);
    v3.data() = _mm_unpackhi_epi16(tmp6, tmp7);
    v4.data() = _mm_unpacklo_epi16(tmp8, tmp9);
    v5.data() = _mm_unpackhi_epi16(tmp8, tmp9);
    v6.data() = _mm_unpacklo_epi16(tmp14, tmp15);
    v7.data() = _mm_unpackhi_epi16(tmp14, tmp15);
}/*}}}*/

template<> inline void InterleavedMemoryAccessBase<sfloat_v>::deinterleave(sfloat_v &v0, sfloat_v &v1) const/*{{{*/
{
    const __m128 i0a = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&m_data[m_indexes[0]])));
    const __m128 i1a = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&m_data[m_indexes[1]])));
    const __m128 i2a = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&m_data[m_indexes[2]])));
    const __m128 i3a = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&m_data[m_indexes[3]])));
    const __m128 i4a = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&m_data[m_indexes[4]])));
    const __m128 i5a = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&m_data[m_indexes[5]])));
    const __m128 i6a = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&m_data[m_indexes[6]])));
    const __m128 i7a = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&m_data[m_indexes[7]])));

    const __m128 ab01 = _mm_unpacklo_ps(i0a, i1a); // [a0 a1 b0 b1]
    const __m128 ab23 = _mm_unpacklo_ps(i2a, i3a); // [a2 a3 b2 b3]
    const __m128 ab45 = _mm_unpacklo_ps(i4a, i5a); // [a4 a5 b4 b5]
    const __m128 ab67 = _mm_unpacklo_ps(i6a, i7a); // [a6 a7 b6 b7]
    v0.data() = Vc::SSE::M256::create(_mm_movelh_ps(ab01, ab23), _mm_movelh_ps(ab45, ab67));
    v1.data() = Vc::SSE::M256::create(_mm_movehl_ps(ab23, ab01), _mm_movehl_ps(ab67, ab45));
}
/*}}}*/
template<> inline void InterleavedMemoryAccessBase<sfloat_v>::deinterleave(sfloat_v &v0, sfloat_v &v1, sfloat_v &v2) const/*{{{*/
{
    const __m128 i0a = _mm_loadu_ps(&m_data[m_indexes[0]]);
    const __m128 i1a = _mm_loadu_ps(&m_data[m_indexes[1]]);
    const __m128 i2a = _mm_loadu_ps(&m_data[m_indexes[2]]);
    const __m128 i3a = _mm_loadu_ps(&m_data[m_indexes[3]]);
    const __m128 i4a = _mm_loadu_ps(&m_data[m_indexes[4]]);
    const __m128 i5a = _mm_loadu_ps(&m_data[m_indexes[5]]);
    const __m128 i6a = _mm_loadu_ps(&m_data[m_indexes[6]]);
    const __m128 i7a = _mm_loadu_ps(&m_data[m_indexes[7]]);

    const __m128 ab01 = _mm_unpacklo_ps(i0a, i1a); // [a0 a1 b0 b1]
    const __m128 ab23 = _mm_unpacklo_ps(i2a, i3a); // [a2 a3 b2 b3]
    const __m128 ab45 = _mm_unpacklo_ps(i4a, i5a); // [a4 a5 b4 b5]
    const __m128 ab67 = _mm_unpacklo_ps(i6a, i7a); // [a6 a7 b6 b7]
    v0.data() = Vc::SSE::M256::create(_mm_movelh_ps(ab01, ab23), _mm_movelh_ps(ab45, ab67));
    v1.data() = Vc::SSE::M256::create(_mm_movehl_ps(ab23, ab01), _mm_movehl_ps(ab67, ab45));

    const __m128 cd01 = _mm_unpackhi_ps(i0a, i1a); // [c0 c1 d0 d1]
    const __m128 cd23 = _mm_unpackhi_ps(i2a, i3a); // [c2 c3 d2 d3]
    const __m128 cd45 = _mm_unpackhi_ps(i4a, i5a); // [c4 c5 d4 d5]
    const __m128 cd67 = _mm_unpackhi_ps(i6a, i7a); // [c6 c7 d6 d7]
    v2.data() = Vc::SSE::M256::create(_mm_movelh_ps(cd01, cd23), _mm_movelh_ps(cd45, cd67));
}
/*}}}*/
template<> inline void InterleavedMemoryAccessBase<sfloat_v>::deinterleave(sfloat_v &v0, sfloat_v &v1, sfloat_v &v2, sfloat_v &v3) const/*{{{*/
{
    const __m128 i0a = _mm_loadu_ps(&m_data[m_indexes[0]]);
    const __m128 i1a = _mm_loadu_ps(&m_data[m_indexes[1]]);
    const __m128 i2a = _mm_loadu_ps(&m_data[m_indexes[2]]);
    const __m128 i3a = _mm_loadu_ps(&m_data[m_indexes[3]]);
    const __m128 i4a = _mm_loadu_ps(&m_data[m_indexes[4]]);
    const __m128 i5a = _mm_loadu_ps(&m_data[m_indexes[5]]);
    const __m128 i6a = _mm_loadu_ps(&m_data[m_indexes[6]]);
    const __m128 i7a = _mm_loadu_ps(&m_data[m_indexes[7]]);

    const __m128 ab01 = _mm_unpacklo_ps(i0a, i1a); // [a0 a1 b0 b1]
    const __m128 ab23 = _mm_unpacklo_ps(i2a, i3a); // [a2 a3 b2 b3]
    const __m128 ab45 = _mm_unpacklo_ps(i4a, i5a); // [a4 a5 b4 b5]
    const __m128 ab67 = _mm_unpacklo_ps(i6a, i7a); // [a6 a7 b6 b7]
    v0.data() = Vc::SSE::M256::create(_mm_movelh_ps(ab01, ab23), _mm_movelh_ps(ab45, ab67));
    v1.data() = Vc::SSE::M256::create(_mm_movehl_ps(ab23, ab01), _mm_movehl_ps(ab67, ab45));

    const __m128 cd01 = _mm_unpackhi_ps(i0a, i1a); // [c0 c1 d0 d1]
    const __m128 cd23 = _mm_unpackhi_ps(i2a, i3a); // [c2 c3 d2 d3]
    const __m128 cd45 = _mm_unpackhi_ps(i4a, i5a); // [c4 c5 d4 d5]
    const __m128 cd67 = _mm_unpackhi_ps(i6a, i7a); // [c6 c7 d6 d7]
    v2.data() = Vc::SSE::M256::create(_mm_movelh_ps(cd01, cd23), _mm_movelh_ps(cd45, cd67));
    v3.data() = Vc::SSE::M256::create(_mm_movehl_ps(cd23, cd01), _mm_movehl_ps(cd67, cd45));
}
/*}}}*/
template<> inline void InterleavedMemoryAccessBase<sfloat_v>::deinterleave(sfloat_v &v0, sfloat_v &v1, sfloat_v &v2, sfloat_v &v3, sfloat_v &v4) const/*{{{*/
{
    const __m128 i0a = _mm_loadu_ps(&m_data[m_indexes[0]]);
    const __m128 i1a = _mm_loadu_ps(&m_data[m_indexes[1]]);
    const __m128 i2a = _mm_loadu_ps(&m_data[m_indexes[2]]);
    const __m128 i3a = _mm_loadu_ps(&m_data[m_indexes[3]]);
    const __m128 i4a = _mm_loadu_ps(&m_data[m_indexes[4]]);
    const __m128 i5a = _mm_loadu_ps(&m_data[m_indexes[5]]);
    const __m128 i6a = _mm_loadu_ps(&m_data[m_indexes[6]]);
    const __m128 i7a = _mm_loadu_ps(&m_data[m_indexes[7]]);
    v4.gather(m_data + float_v::Size, m_indexes);

    const __m128 ab01 = _mm_unpacklo_ps(i0a, i1a); // [a0 a1 b0 b1]
    const __m128 ab23 = _mm_unpacklo_ps(i2a, i3a); // [a2 a3 b2 b3]
    const __m128 ab45 = _mm_unpacklo_ps(i4a, i5a); // [a4 a5 b4 b5]
    const __m128 ab67 = _mm_unpacklo_ps(i6a, i7a); // [a6 a7 b6 b7]
    v0.data() = Vc::SSE::M256::create(_mm_movelh_ps(ab01, ab23), _mm_movelh_ps(ab45, ab67));
    v1.data() = Vc::SSE::M256::create(_mm_movehl_ps(ab23, ab01), _mm_movehl_ps(ab67, ab45));

    const __m128 cd01 = _mm_unpackhi_ps(i0a, i1a); // [c0 c1 d0 d1]
    const __m128 cd23 = _mm_unpackhi_ps(i2a, i3a); // [c2 c3 d2 d3]
    const __m128 cd45 = _mm_unpackhi_ps(i4a, i5a); // [c4 c5 d4 d5]
    const __m128 cd67 = _mm_unpackhi_ps(i6a, i7a); // [c6 c7 d6 d7]
    v2.data() = Vc::SSE::M256::create(_mm_movelh_ps(cd01, cd23), _mm_movelh_ps(cd45, cd67));
    v3.data() = Vc::SSE::M256::create(_mm_movehl_ps(cd23, cd01), _mm_movehl_ps(cd67, cd45));
}
/*}}}*/
template<> inline void InterleavedMemoryAccessBase<sfloat_v>::deinterleave(sfloat_v &v0, sfloat_v &v1, sfloat_v &v2, sfloat_v &v3, sfloat_v &v4, sfloat_v &v5) const/*{{{*/
{
    const __m128 i0a = _mm_loadu_ps(&m_data[m_indexes[0]]);
    const __m128 i1a = _mm_loadu_ps(&m_data[m_indexes[1]]);
    const __m128 i2a = _mm_loadu_ps(&m_data[m_indexes[2]]);
    const __m128 i3a = _mm_loadu_ps(&m_data[m_indexes[3]]);
    const __m128 i4a = _mm_loadu_ps(&m_data[m_indexes[4]]);
    const __m128 i5a = _mm_loadu_ps(&m_data[m_indexes[5]]);
    const __m128 i6a = _mm_loadu_ps(&m_data[m_indexes[6]]);
    const __m128 i7a = _mm_loadu_ps(&m_data[m_indexes[7]]);
    const __m128 i0b = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&m_data[m_indexes[0] + float_v::Size])));
    const __m128 i1b = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&m_data[m_indexes[1] + float_v::Size])));
    const __m128 i2b = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&m_data[m_indexes[2] + float_v::Size])));
    const __m128 i3b = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&m_data[m_indexes[3] + float_v::Size])));
    const __m128 i4b = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&m_data[m_indexes[4] + float_v::Size])));
    const __m128 i5b = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&m_data[m_indexes[5] + float_v::Size])));
    const __m128 i6b = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&m_data[m_indexes[6] + float_v::Size])));
    const __m128 i7b = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&m_data[m_indexes[7] + float_v::Size])));

    const __m128 ab01 = _mm_unpacklo_ps(i0a, i1a); // [a0 a1 b0 b1]
    const __m128 ab23 = _mm_unpacklo_ps(i2a, i3a); // [a2 a3 b2 b3]
    const __m128 ab45 = _mm_unpacklo_ps(i4a, i5a); // [a4 a5 b4 b5]
    const __m128 ab67 = _mm_unpacklo_ps(i6a, i7a); // [a6 a7 b6 b7]
    v0.data() = Vc::SSE::M256::create(_mm_movelh_ps(ab01, ab23), _mm_movelh_ps(ab45, ab67));
    v1.data() = Vc::SSE::M256::create(_mm_movehl_ps(ab23, ab01), _mm_movehl_ps(ab67, ab45));

    const __m128 cd01 = _mm_unpackhi_ps(i0a, i1a); // [c0 c1 d0 d1]
    const __m128 cd23 = _mm_unpackhi_ps(i2a, i3a); // [c2 c3 d2 d3]
    const __m128 cd45 = _mm_unpackhi_ps(i4a, i5a); // [c4 c5 d4 d5]
    const __m128 cd67 = _mm_unpackhi_ps(i6a, i7a); // [c6 c7 d6 d7]
    v2.data() = Vc::SSE::M256::create(_mm_movelh_ps(cd01, cd23), _mm_movelh_ps(cd45, cd67));
    v3.data() = Vc::SSE::M256::create(_mm_movehl_ps(cd23, cd01), _mm_movehl_ps(cd67, cd45));

    const __m128 ef01 = _mm_unpacklo_ps(i0b, i1b); // [e0 e1 f0 f1]
    const __m128 ef23 = _mm_unpacklo_ps(i2b, i3b); // [e2 e3 f2 f3]
    const __m128 ef45 = _mm_unpacklo_ps(i4b, i5b); // [e4 e5 f4 f5]
    const __m128 ef67 = _mm_unpacklo_ps(i6b, i7b); // [e6 e7 f6 f7]
    v4.data() = Vc::SSE::M256::create(_mm_movelh_ps(ef01, ef23), _mm_movelh_ps(ef45, ef67));
    v5.data() = Vc::SSE::M256::create(_mm_movehl_ps(ef23, ef01), _mm_movehl_ps(ef67, ef45));
}
/*}}}*/
template<> inline void InterleavedMemoryAccessBase<sfloat_v>::deinterleave(sfloat_v &v0, sfloat_v &v1, sfloat_v &v2, sfloat_v &v3, sfloat_v &v4, sfloat_v &v5, sfloat_v &v6) const/*{{{*/
{
    const __m128 i0a = _mm_loadu_ps(&m_data[m_indexes[0]]);
    const __m128 i0b = _mm_loadu_ps(&m_data[m_indexes[0] + float_v::Size]);
    const __m128 i1a = _mm_loadu_ps(&m_data[m_indexes[1]]);
    const __m128 i1b = _mm_loadu_ps(&m_data[m_indexes[1] + float_v::Size]);
    const __m128 i2a = _mm_loadu_ps(&m_data[m_indexes[2]]);
    const __m128 i2b = _mm_loadu_ps(&m_data[m_indexes[2] + float_v::Size]);
    const __m128 i3a = _mm_loadu_ps(&m_data[m_indexes[3]]);
    const __m128 i3b = _mm_loadu_ps(&m_data[m_indexes[3] + float_v::Size]);
    const __m128 i4a = _mm_loadu_ps(&m_data[m_indexes[4]]);
    const __m128 i4b = _mm_loadu_ps(&m_data[m_indexes[4] + float_v::Size]);
    const __m128 i5a = _mm_loadu_ps(&m_data[m_indexes[5]]);
    const __m128 i5b = _mm_loadu_ps(&m_data[m_indexes[5] + float_v::Size]);
    const __m128 i6a = _mm_loadu_ps(&m_data[m_indexes[6]]);
    const __m128 i6b = _mm_loadu_ps(&m_data[m_indexes[6] + float_v::Size]);
    const __m128 i7a = _mm_loadu_ps(&m_data[m_indexes[7]]);
    const __m128 i7b = _mm_loadu_ps(&m_data[m_indexes[7] + float_v::Size]);

    const __m128 ab01 = _mm_unpacklo_ps(i0a, i1a); // [a0 a1 b0 b1]
    const __m128 ab23 = _mm_unpacklo_ps(i2a, i3a); // [a2 a3 b2 b3]
    const __m128 ab45 = _mm_unpacklo_ps(i4a, i5a); // [a4 a5 b4 b5]
    const __m128 ab67 = _mm_unpacklo_ps(i6a, i7a); // [a6 a7 b6 b7]
    v0.data() = Vc::SSE::M256::create(_mm_movelh_ps(ab01, ab23), _mm_movelh_ps(ab45, ab67));
    v1.data() = Vc::SSE::M256::create(_mm_movehl_ps(ab23, ab01), _mm_movehl_ps(ab67, ab45));

    const __m128 cd01 = _mm_unpackhi_ps(i0a, i1a); // [c0 c1 d0 d1]
    const __m128 cd23 = _mm_unpackhi_ps(i2a, i3a); // [c2 c3 d2 d3]
    const __m128 cd45 = _mm_unpackhi_ps(i4a, i5a); // [c4 c5 d4 d5]
    const __m128 cd67 = _mm_unpackhi_ps(i6a, i7a); // [c6 c7 d6 d7]
    v2.data() = Vc::SSE::M256::create(_mm_movelh_ps(cd01, cd23), _mm_movelh_ps(cd45, cd67));
    v3.data() = Vc::SSE::M256::create(_mm_movehl_ps(cd23, cd01), _mm_movehl_ps(cd67, cd45));

    const __m128 ef01 = _mm_unpacklo_ps(i0b, i1b); // [e0 e1 f0 f1]
    const __m128 ef23 = _mm_unpacklo_ps(i2b, i3b); // [e2 e3 f2 f3]
    const __m128 ef45 = _mm_unpacklo_ps(i4b, i5b); // [e4 e5 f4 f5]
    const __m128 ef67 = _mm_unpacklo_ps(i6b, i7b); // [e6 e7 f6 f7]
    v4.data() = Vc::SSE::M256::create(_mm_movelh_ps(ef01, ef23), _mm_movelh_ps(ef45, ef67));
    v5.data() = Vc::SSE::M256::create(_mm_movehl_ps(ef23, ef01), _mm_movehl_ps(ef67, ef45));

    const __m128 gh01 = _mm_unpackhi_ps(i0b, i1b); // [g0 g1 h0 h1]
    const __m128 gh23 = _mm_unpackhi_ps(i2b, i3b); // [g2 g3 h2 h3]
    const __m128 gh45 = _mm_unpackhi_ps(i4b, i5b); // [g4 g5 h4 h5]
    const __m128 gh67 = _mm_unpackhi_ps(i6b, i7b); // [g6 g7 h6 h7]
    v6.data() = Vc::SSE::M256::create(_mm_movelh_ps(gh01, gh23), _mm_movelh_ps(gh45, gh67));
}
/*}}}*/
template<> inline void InterleavedMemoryAccessBase<sfloat_v>::deinterleave(sfloat_v &v0, sfloat_v &v1, sfloat_v &v2, sfloat_v &v3, sfloat_v &v4, sfloat_v &v5, sfloat_v &v6, sfloat_v &v7) const/*{{{*/
{
    const __m128 i0a = _mm_loadu_ps(&m_data[m_indexes[0]]);
    const __m128 i0b = _mm_loadu_ps(&m_data[m_indexes[0] + float_v::Size]);
    const __m128 i1a = _mm_loadu_ps(&m_data[m_indexes[1]]);
    const __m128 i1b = _mm_loadu_ps(&m_data[m_indexes[1] + float_v::Size]);
    const __m128 i2a = _mm_loadu_ps(&m_data[m_indexes[2]]);
    const __m128 i2b = _mm_loadu_ps(&m_data[m_indexes[2] + float_v::Size]);
    const __m128 i3a = _mm_loadu_ps(&m_data[m_indexes[3]]);
    const __m128 i3b = _mm_loadu_ps(&m_data[m_indexes[3] + float_v::Size]);
    const __m128 i4a = _mm_loadu_ps(&m_data[m_indexes[4]]);
    const __m128 i4b = _mm_loadu_ps(&m_data[m_indexes[4] + float_v::Size]);
    const __m128 i5a = _mm_loadu_ps(&m_data[m_indexes[5]]);
    const __m128 i5b = _mm_loadu_ps(&m_data[m_indexes[5] + float_v::Size]);
    const __m128 i6a = _mm_loadu_ps(&m_data[m_indexes[6]]);
    const __m128 i6b = _mm_loadu_ps(&m_data[m_indexes[6] + float_v::Size]);
    const __m128 i7a = _mm_loadu_ps(&m_data[m_indexes[7]]);
    const __m128 i7b = _mm_loadu_ps(&m_data[m_indexes[7] + float_v::Size]);

    const __m128 ab01 = _mm_unpacklo_ps(i0a, i1a); // [a0 a1 b0 b1]
    const __m128 ab23 = _mm_unpacklo_ps(i2a, i3a); // [a2 a3 b2 b3]
    const __m128 ab45 = _mm_unpacklo_ps(i4a, i5a); // [a4 a5 b4 b5]
    const __m128 ab67 = _mm_unpacklo_ps(i6a, i7a); // [a6 a7 b6 b7]
    v0.data() = Vc::SSE::M256::create(_mm_movelh_ps(ab01, ab23), _mm_movelh_ps(ab45, ab67));
    v1.data() = Vc::SSE::M256::create(_mm_movehl_ps(ab23, ab01), _mm_movehl_ps(ab67, ab45));

    const __m128 cd01 = _mm_unpackhi_ps(i0a, i1a); // [c0 c1 d0 d1]
    const __m128 cd23 = _mm_unpackhi_ps(i2a, i3a); // [c2 c3 d2 d3]
    const __m128 cd45 = _mm_unpackhi_ps(i4a, i5a); // [c4 c5 d4 d5]
    const __m128 cd67 = _mm_unpackhi_ps(i6a, i7a); // [c6 c7 d6 d7]
    v2.data() = Vc::SSE::M256::create(_mm_movelh_ps(cd01, cd23), _mm_movelh_ps(cd45, cd67));
    v3.data() = Vc::SSE::M256::create(_mm_movehl_ps(cd23, cd01), _mm_movehl_ps(cd67, cd45));

    const __m128 ef01 = _mm_unpacklo_ps(i0b, i1b); // [e0 e1 f0 f1]
    const __m128 ef23 = _mm_unpacklo_ps(i2b, i3b); // [e2 e3 f2 f3]
    const __m128 ef45 = _mm_unpacklo_ps(i4b, i5b); // [e4 e5 f4 f5]
    const __m128 ef67 = _mm_unpacklo_ps(i6b, i7b); // [e6 e7 f6 f7]
    v4.data() = Vc::SSE::M256::create(_mm_movelh_ps(ef01, ef23), _mm_movelh_ps(ef45, ef67));
    v5.data() = Vc::SSE::M256::create(_mm_movehl_ps(ef23, ef01), _mm_movehl_ps(ef67, ef45));

    const __m128 gh01 = _mm_unpackhi_ps(i0b, i1b); // [g0 g1 h0 h1]
    const __m128 gh23 = _mm_unpackhi_ps(i2b, i3b); // [g2 g3 h2 h3]
    const __m128 gh45 = _mm_unpackhi_ps(i4b, i5b); // [g4 g5 h4 h5]
    const __m128 gh67 = _mm_unpackhi_ps(i6b, i7b); // [g6 g7 h6 h7]
    v6.data() = Vc::SSE::M256::create(_mm_movelh_ps(gh01, gh23), _mm_movelh_ps(gh45, gh67));
    v7.data() = Vc::SSE::M256::create(_mm_movehl_ps(gh23, gh01), _mm_movehl_ps(gh67, gh45));
}/*}}}*/

// forward types of equal size - ugly, but it works/*{{{*/
#define _forward(V, V2) \
template<> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::deinterleave(V &v0, V &v1) const { \
    reinterpret_cast<const InterleavedMemoryAccessBase<V2> *>(this)->deinterleave(reinterpret_cast<V2 &>(v0), reinterpret_cast<V2 &>(v1)); \
} \
template<> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::deinterleave(V &v0, V &v1, V &v2) const { \
    reinterpret_cast<const InterleavedMemoryAccessBase<V2> *>(this)->deinterleave(reinterpret_cast<V2 &>(v0), reinterpret_cast<V2 &>(v1), \
            reinterpret_cast<V2 &>(v2)); \
} \
template<> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::deinterleave(V &v0, V &v1, V &v2, V &v3) const { \
    reinterpret_cast<const InterleavedMemoryAccessBase<V2> *>(this)->deinterleave(reinterpret_cast<V2 &>(v0), reinterpret_cast<V2 &>(v1), \
            reinterpret_cast<V2 &>(v2), reinterpret_cast<V2 &>(v3)); \
} \
template<> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::deinterleave(V &v0, V &v1, V &v2, V &v3, \
        V &v4) const { \
    reinterpret_cast<const InterleavedMemoryAccessBase<V2> *>(this)->deinterleave(reinterpret_cast<V2 &>(v0), reinterpret_cast<V2 &>(v1), \
            reinterpret_cast<V2 &>(v2), reinterpret_cast<V2 &>(v3), reinterpret_cast<V2 &>(v4)); \
} \
template<> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::deinterleave(V &v0, V &v1, V &v2, V &v3, \
        V &v4, V &v5) const { \
    reinterpret_cast<const InterleavedMemoryAccessBase<V2> *>(this)->deinterleave(reinterpret_cast<V2 &>(v0), reinterpret_cast<V2 &>(v1), \
            reinterpret_cast<V2 &>(v2), reinterpret_cast<V2 &>(v3), reinterpret_cast<V2 &>(v4), \
            reinterpret_cast<V2 &>(v5)); \
} \
template<> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::deinterleave(V &v0, V &v1, V &v2, V &v3, \
        V &v4, V &v5, V &v6) const { \
    reinterpret_cast<const InterleavedMemoryAccessBase<V2> *>(this)->deinterleave(reinterpret_cast<V2 &>(v0), reinterpret_cast<V2 &>(v1), \
            reinterpret_cast<V2 &>(v2), reinterpret_cast<V2 &>(v3), reinterpret_cast<V2 &>(v4), \
            reinterpret_cast<V2 &>(v5), reinterpret_cast<V2 &>(v6)); \
} \
template<> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::deinterleave(V &v0, V &v1, V &v2, V &v3, \
        V &v4, V &v5, V &v6, V &v7) const { \
    reinterpret_cast<const InterleavedMemoryAccessBase<V2> *>(this)->deinterleave(reinterpret_cast<V2 &>(v0), reinterpret_cast<V2 &>(v1), \
            reinterpret_cast<V2 &>(v2), reinterpret_cast<V2 &>(v3), reinterpret_cast<V2 &>(v4), \
            reinterpret_cast<V2 &>(v5), reinterpret_cast<V2 &>(v6), reinterpret_cast<V2 &>(v7)); \
}
_forward( int_v, float_v)
_forward(uint_v, float_v)
_forward(ushort_v, short_v)
#undef _forward/*}}}*/

} // namespace Common
} // namespace Vc
} // namespace ROOT

#include "undomacros.h"

#endif // VC_SSE_INTERLEAVEDMEMORY_TCC

// vim: foldmethod=marker
