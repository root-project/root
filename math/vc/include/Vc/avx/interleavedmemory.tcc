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

#ifndef VC_AVX_INTERLEAVEDMEMORY_TCC
#define VC_AVX_INTERLEAVEDMEMORY_TCC

#include "macros.h"

namespace ROOT {
namespace Vc
{
namespace Common
{

namespace
{
template<typename V, int Size, size_t VSize> struct InterleaveImpl;
template<typename V> struct InterleaveImpl<V, 8, 16> {
    static inline void interleave(typename V::EntryType *const data, const typename V::IndexType &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1)
    {
        const m128i tmp0 = _mm_unpacklo_epi16(v0.data(), v1.data());
        const m128i tmp1 = _mm_unpackhi_epi16(v0.data(), v1.data());
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
#else
        *reinterpret_cast<int *>(&data[i[0]]) = _mm_cvtsi128_si32(tmp0);
        *reinterpret_cast<int *>(&data[i[1]]) = _mm_extract_epi32(tmp0, 1);
        *reinterpret_cast<int *>(&data[i[2]]) = _mm_extract_epi32(tmp0, 2);
        *reinterpret_cast<int *>(&data[i[3]]) = _mm_extract_epi32(tmp0, 3);
        *reinterpret_cast<int *>(&data[i[4]]) = _mm_cvtsi128_si32(tmp1);
        *reinterpret_cast<int *>(&data[i[5]]) = _mm_extract_epi32(tmp1, 1);
        *reinterpret_cast<int *>(&data[i[6]]) = _mm_extract_epi32(tmp1, 2);
        *reinterpret_cast<int *>(&data[i[7]]) = _mm_extract_epi32(tmp1, 3);
#endif
    }/*}}}*/
    static inline void interleave(typename V::EntryType *const data, const typename V::IndexType &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2)
    {
#ifdef VC_USE_MASKMOV_SCATTER
        const m128i maskLo = _mm_set_epi16(0, 0, 0, 0, 0, -1, -1, -1);
        const m128i maskHi = _mm_set_epi16(0, -1, -1, -1, 0, 0, 0, 0);
        typename V::EntryType *const dataHi = data - 4;
        const m128i tmp0 = _mm_unpacklo_epi16(v0.data(), v2.data());
        const m128i tmp1 = _mm_unpackhi_epi16(v0.data(), v2.data());
        const m128i tmp2 = _mm_unpacklo_epi16(v1.data(), v1.data());
        const m128i tmp3 = _mm_unpackhi_epi16(v1.data(), v1.data());

        const m128i tmp4 = _mm_unpacklo_epi16(tmp0, tmp2);
        const m128i tmp5 = _mm_unpackhi_epi16(tmp0, tmp2);
        const m128i tmp6 = _mm_unpacklo_epi16(tmp1, tmp3);
        const m128i tmp7 = _mm_unpackhi_epi16(tmp1, tmp3);
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
        const m128i tmp0 = _mm_unpacklo_epi16(v0.data(), v2.data());
        const m128i tmp1 = _mm_unpackhi_epi16(v0.data(), v2.data());
        const m128i tmp2 = _mm_unpacklo_epi16(v1.data(), v3.data());
        const m128i tmp3 = _mm_unpackhi_epi16(v1.data(), v3.data());

        const m128i tmp4 = _mm_unpacklo_epi16(tmp0, tmp2);
        const m128i tmp5 = _mm_unpackhi_epi16(tmp0, tmp2);
        const m128i tmp6 = _mm_unpacklo_epi16(tmp1, tmp3);
        const m128i tmp7 = _mm_unpackhi_epi16(tmp1, tmp3);

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
template<typename V> struct InterleaveImpl<V, 8, 32> {
    static inline void interleave(typename V::EntryType *const data, const typename V::IndexType &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1)
    {
        using namespace Vc::AVX;
        // [0a 1a 0b 1b 0e 1e 0f 1f]:
        const m256 tmp0 = _mm256_unpacklo_ps(AVX::avx_cast<m256>(v0.data()), AVX::avx_cast<m256>(v1.data()));
        // [0c 1c 0d 1d 0g 1g 0h 1h]:
        const m256 tmp1 = _mm256_unpackhi_ps(AVX::avx_cast<m256>(v0.data()), AVX::avx_cast<m256>(v1.data()));
        _mm_storel_pi(reinterpret_cast<__m64 *>(&data[i[0]]), lo128(tmp0));
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[1]]), lo128(tmp0));
        _mm_storel_pi(reinterpret_cast<__m64 *>(&data[i[2]]), lo128(tmp1));
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[3]]), lo128(tmp1));
        _mm_storel_pi(reinterpret_cast<__m64 *>(&data[i[4]]), hi128(tmp0));
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[5]]), hi128(tmp0));
        _mm_storel_pi(reinterpret_cast<__m64 *>(&data[i[6]]), hi128(tmp1));
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[7]]), hi128(tmp1));
    }/*}}}*/
    static inline void interleave(typename V::EntryType *const data, const typename V::IndexType &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2)
    {
        using namespace Vc::AVX;
#ifdef VC_USE_MASKMOV_SCATTER
        // [0a 2a 0b 2b 0e 2e 0f 2f]:
        const m256 tmp0 = _mm256_unpacklo_ps(AVX::avx_cast<m256>(v0.data()), AVX::avx_cast<m256>(v2.data()));
        // [0c 2c 0d 2d 0g 2g 0h 2h]:
        const m256 tmp1 = _mm256_unpackhi_ps(AVX::avx_cast<m256>(v0.data()), AVX::avx_cast<m256>(v2.data()));
        // [1a __ 1b __ 1e __ 1f __]:
        const m256 tmp2 = _mm256_unpacklo_ps(AVX::avx_cast<m256>(v1.data()), AVX::avx_cast<m256>(v1.data()));
        // [1c __ 1d __ 1g __ 1h __]:
        const m256 tmp3 = _mm256_unpackhi_ps(AVX::avx_cast<m256>(v1.data()), AVX::avx_cast<m256>(v1.data()));
        const m256 tmp4 = _mm256_unpacklo_ps(tmp0, tmp2);
        const m256 tmp5 = _mm256_unpackhi_ps(tmp0, tmp2);
        const m256 tmp6 = _mm256_unpacklo_ps(tmp1, tmp3);
        const m256 tmp7 = _mm256_unpackhi_ps(tmp1, tmp3);
        const m128i mask = _mm_set_epi32(0, -1, -1, -1);
        _mm_maskstore_ps(reinterpret_cast<float *>(&data[i[0]]), mask, lo128(tmp4));
        _mm_maskstore_ps(reinterpret_cast<float *>(&data[i[1]]), mask, lo128(tmp5));
        _mm_maskstore_ps(reinterpret_cast<float *>(&data[i[2]]), mask, lo128(tmp6));
        _mm_maskstore_ps(reinterpret_cast<float *>(&data[i[3]]), mask, lo128(tmp7));
        _mm_maskstore_ps(reinterpret_cast<float *>(&data[i[4]]), mask, hi128(tmp4));
        _mm_maskstore_ps(reinterpret_cast<float *>(&data[i[5]]), mask, hi128(tmp5));
        _mm_maskstore_ps(reinterpret_cast<float *>(&data[i[6]]), mask, hi128(tmp6));
        _mm_maskstore_ps(reinterpret_cast<float *>(&data[i[7]]), mask, hi128(tmp7));
#else
        interleave(data, i, v0, v1);
        v2.scatter(data + 2, i);
#endif
    }/*}}}*/
    static inline void interleave(typename V::EntryType *const data, const typename V::IndexType &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1,
            const typename V::AsArg v2, const typename V::AsArg v3)
    {
        using namespace Vc::AVX;
        const m256 tmp0 = _mm256_unpacklo_ps(AVX::avx_cast<m256>(v0.data()), AVX::avx_cast<m256>(v2.data()));
        const m256 tmp1 = _mm256_unpackhi_ps(AVX::avx_cast<m256>(v0.data()), AVX::avx_cast<m256>(v2.data()));
        const m256 tmp2 = _mm256_unpacklo_ps(AVX::avx_cast<m256>(v1.data()), AVX::avx_cast<m256>(v3.data()));
        const m256 tmp3 = _mm256_unpackhi_ps(AVX::avx_cast<m256>(v1.data()), AVX::avx_cast<m256>(v3.data()));
        const m256 tmp4 = _mm256_unpacklo_ps(tmp0, tmp2);
        const m256 tmp5 = _mm256_unpackhi_ps(tmp0, tmp2);
        const m256 tmp6 = _mm256_unpacklo_ps(tmp1, tmp3);
        const m256 tmp7 = _mm256_unpackhi_ps(tmp1, tmp3);
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[0]]), lo128(tmp4));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[1]]), lo128(tmp5));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[2]]), lo128(tmp6));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[3]]), lo128(tmp7));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[4]]), hi128(tmp4));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[5]]), hi128(tmp5));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[6]]), hi128(tmp6));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[7]]), hi128(tmp7));
    }/*}}}*/
};
template<typename V> struct InterleaveImpl<V, 4, 32> {
    static inline void interleave(typename V::EntryType *const data, const typename V::IndexType &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1)
    {
        using namespace Vc::AVX;
        const m256d tmp0 = _mm256_unpacklo_pd(v0.data(), v1.data());
        const m256d tmp1 = _mm256_unpackhi_pd(v0.data(), v1.data());
        _mm_storeu_pd(&data[i[0]], lo128(tmp0));
        _mm_storeu_pd(&data[i[1]], lo128(tmp1));
        _mm_storeu_pd(&data[i[2]], hi128(tmp0));
        _mm_storeu_pd(&data[i[3]], hi128(tmp1));
    }/*}}}*/
    static inline void interleave(typename V::EntryType *const data, const typename V::IndexType &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2)
    {
        using namespace Vc::AVX;
#ifdef VC_USE_MASKMOV_SCATTER
        const m256d tmp0 = _mm256_unpacklo_pd(v0.data(), v1.data());
        const m256d tmp1 = _mm256_unpackhi_pd(v0.data(), v1.data());
        const m256d tmp2 = _mm256_unpacklo_pd(v2.data(), v2.data());
        const m256d tmp3 = _mm256_unpackhi_pd(v2.data(), v2.data());

#if defined(VC_MSVC) && (VC_MSVC < 170000000 || !defined(_WIN64))
        // MSVC needs to be at Version 2012 before _mm256_set_epi64x works
        const m256i mask = AVX::concat(_mm_setallone_si128(), _mm_set_epi32(0, 0, -1, -1));
#else
        const m256i mask = _mm256_set_epi64x(0, -1, -1, -1);
#endif
        _mm256_maskstore_pd(&data[i[0]], mask, Mem::shuffle128<X0, Y0>(tmp0, tmp2));
        _mm256_maskstore_pd(&data[i[1]], mask, Mem::shuffle128<X0, Y0>(tmp1, tmp3));
        _mm256_maskstore_pd(&data[i[2]], mask, Mem::shuffle128<X1, Y1>(tmp0, tmp2));
        _mm256_maskstore_pd(&data[i[3]], mask, Mem::shuffle128<X1, Y1>(tmp1, tmp3));
#else
        interleave(data, i, v0, v1);
        v2.scatter(data + 2, i);
#endif
    }/*}}}*/
    static inline void interleave(typename V::EntryType *const data, const typename V::IndexType &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1,
            const typename V::AsArg v2, const typename V::AsArg v3)
    {
        using namespace Vc::AVX;
        // 0a 1a 0c 1c:
        const m256d tmp0 = _mm256_unpacklo_pd(v0.data(), v1.data());
        // 0b 1b 0b 1b:
        const m256d tmp1 = _mm256_unpackhi_pd(v0.data(), v1.data());
        // 2a 3a 2c 3c:
        const m256d tmp2 = _mm256_unpacklo_pd(v2.data(), v3.data());
        // 2b 3b 2b 3b:
        const m256d tmp3 = _mm256_unpackhi_pd(v2.data(), v3.data());
        _mm256_storeu_pd(&data[i[0]], Mem::shuffle128<X0, Y0>(tmp0, tmp2));
        _mm256_storeu_pd(&data[i[1]], Mem::shuffle128<X0, Y0>(tmp1, tmp3));
        _mm256_storeu_pd(&data[i[2]], Mem::shuffle128<X1, Y1>(tmp0, tmp2));
        _mm256_storeu_pd(&data[i[3]], Mem::shuffle128<X1, Y1>(tmp1, tmp3));
    }/*}}}*/
};
} // anonymous namespace

template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1)
{
    InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data, m_indexes, v0, v1);
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2)
{
    InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data, m_indexes, v0, v1, v2);
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2, const typename V::AsArg v3)
{
    InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data, m_indexes, v0, v1, v2, v3);
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2, const typename V::AsArg v3, const typename V::AsArg v4)
{
    InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data, m_indexes, v0, v1, v2, v3);
    v4.scatter(m_data + 4, m_indexes);
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2, const typename V::AsArg v3, const typename V::AsArg v4,
        const typename V::AsArg v5)
{
    InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data    , m_indexes, v0, v1, v2, v3);
    InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data + 4, m_indexes, v4, v5);
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2, const typename V::AsArg v3, const typename V::AsArg v4,
        const typename V::AsArg v5, const typename V::AsArg v6)
{
    InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data + 0, m_indexes, v0, v1, v2, v3);
    InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data + 4, m_indexes, v4, v5, v6);
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2, const typename V::AsArg v3, const typename V::AsArg v4,
        const typename V::AsArg v5, const typename V::AsArg v6, const typename V::AsArg v7)
{
    InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data + 0, m_indexes, v0, v1, v2, v3);
    InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data + 4, m_indexes, v4, v5, v6, v7);
}/*}}}*/

template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1) const/*{{{*/
{
    const m128  il0 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[0]])); // a0 b0
    const m128  il2 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[2]])); // a2 b2
    const m128  il4 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[4]])); // a4 b4
    const m128  il6 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[6]])); // a6 b6
    const m128 il01 = _mm_loadh_pi(             il0, reinterpret_cast<__m64 const *>(&m_data[m_indexes[1]])); // a0 b0 a1 b1
    const m128 il23 = _mm_loadh_pi(             il2, reinterpret_cast<__m64 const *>(&m_data[m_indexes[3]])); // a2 b2 a3 b3
    const m128 il45 = _mm_loadh_pi(             il4, reinterpret_cast<__m64 const *>(&m_data[m_indexes[5]])); // a4 b4 a5 b5
    const m128 il67 = _mm_loadh_pi(             il6, reinterpret_cast<__m64 const *>(&m_data[m_indexes[7]])); // a6 b6 a7 b7

    const m256 tmp2 = AVX::concat(il01, il45);
    const m256 tmp3 = AVX::concat(il23, il67);

    const m256 tmp0 = _mm256_unpacklo_ps(tmp2, tmp3);
    const m256 tmp1 = _mm256_unpackhi_ps(tmp2, tmp3);

    v0.data() = _mm256_unpacklo_ps(tmp0, tmp1);
    v1.data() = _mm256_unpackhi_ps(tmp0, tmp1);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2) const/*{{{*/
{
    const m128  il0 = _mm_loadu_ps(&m_data[m_indexes[0]]); // a0 b0 c0 d0
    const m128  il1 = _mm_loadu_ps(&m_data[m_indexes[1]]); // a1 b1 c1 d1
    const m128  il2 = _mm_loadu_ps(&m_data[m_indexes[2]]); // a2 b2 c2 d2
    const m128  il3 = _mm_loadu_ps(&m_data[m_indexes[3]]); // a3 b3 c3 d3
    const m128  il4 = _mm_loadu_ps(&m_data[m_indexes[4]]); // a4 b4 c4 d4
    const m128  il5 = _mm_loadu_ps(&m_data[m_indexes[5]]); // a5 b5 c5 d5
    const m128  il6 = _mm_loadu_ps(&m_data[m_indexes[6]]); // a6 b6 c6 d6
    const m128  il7 = _mm_loadu_ps(&m_data[m_indexes[7]]); // a7 b7 c7 d7

    const m256 il04 = AVX::concat(il0, il4);
    const m256 il15 = AVX::concat(il1, il5);
    const m256 il26 = AVX::concat(il2, il6);
    const m256 il37 = AVX::concat(il3, il7);
    const m256 ab0246 = _mm256_unpacklo_ps(il04, il26);
    const m256 ab1357 = _mm256_unpacklo_ps(il15, il37);
    const m256 cd0246 = _mm256_unpackhi_ps(il04, il26);
    const m256 cd1357 = _mm256_unpackhi_ps(il15, il37);
    v0.data() = _mm256_unpacklo_ps(ab0246, ab1357);
    v1.data() = _mm256_unpackhi_ps(ab0246, ab1357);
    v2.data() = _mm256_unpacklo_ps(cd0246, cd1357);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3) const/*{{{*/
{
    const m128  il0 = _mm_loadu_ps(&m_data[m_indexes[0]]); // a0 b0 c0 d0
    const m128  il1 = _mm_loadu_ps(&m_data[m_indexes[1]]); // a1 b1 c1 d1
    const m128  il2 = _mm_loadu_ps(&m_data[m_indexes[2]]); // a2 b2 c2 d2
    const m128  il3 = _mm_loadu_ps(&m_data[m_indexes[3]]); // a3 b3 c3 d3
    const m128  il4 = _mm_loadu_ps(&m_data[m_indexes[4]]); // a4 b4 c4 d4
    const m128  il5 = _mm_loadu_ps(&m_data[m_indexes[5]]); // a5 b5 c5 d5
    const m128  il6 = _mm_loadu_ps(&m_data[m_indexes[6]]); // a6 b6 c6 d6
    const m128  il7 = _mm_loadu_ps(&m_data[m_indexes[7]]); // a7 b7 c7 d7

    const m256 il04 = AVX::concat(il0, il4);
    const m256 il15 = AVX::concat(il1, il5);
    const m256 il26 = AVX::concat(il2, il6);
    const m256 il37 = AVX::concat(il3, il7);
    const m256 ab0246 = _mm256_unpacklo_ps(il04, il26);
    const m256 ab1357 = _mm256_unpacklo_ps(il15, il37);
    const m256 cd0246 = _mm256_unpackhi_ps(il04, il26);
    const m256 cd1357 = _mm256_unpackhi_ps(il15, il37);
    v0.data() = _mm256_unpacklo_ps(ab0246, ab1357);
    v1.data() = _mm256_unpackhi_ps(ab0246, ab1357);
    v2.data() = _mm256_unpacklo_ps(cd0246, cd1357);
    v3.data() = _mm256_unpackhi_ps(cd0246, cd1357);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3, float_v &v4) const/*{{{*/
{
    v4.gather(m_data, m_indexes + I(4));
    deinterleave(v0, v1, v2, v3);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3, float_v &v4, float_v &v5) const/*{{{*/
{
    deinterleave(v0, v1, v2, v3);
    const m128  il0 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[0] + 4])); // a0 b0
    const m128  il2 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[2] + 4])); // a2 b2
    const m128  il4 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[4] + 4])); // a4 b4
    const m128  il6 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[6] + 4])); // a6 b6
    const m128 il01 = _mm_loadh_pi(             il0, reinterpret_cast<__m64 const *>(&m_data[m_indexes[1] + 4])); // a0 b0 a1 b1
    const m128 il23 = _mm_loadh_pi(             il2, reinterpret_cast<__m64 const *>(&m_data[m_indexes[3] + 4])); // a2 b2 a3 b3
    const m128 il45 = _mm_loadh_pi(             il4, reinterpret_cast<__m64 const *>(&m_data[m_indexes[5] + 4])); // a4 b4 a5 b5
    const m128 il67 = _mm_loadh_pi(             il6, reinterpret_cast<__m64 const *>(&m_data[m_indexes[7] + 4])); // a6 b6 a7 b7

    const m256 tmp2 = AVX::concat(il01, il45);
    const m256 tmp3 = AVX::concat(il23, il67);

    const m256 tmp0 = _mm256_unpacklo_ps(tmp2, tmp3);
    const m256 tmp1 = _mm256_unpackhi_ps(tmp2, tmp3);

    v4.data() = _mm256_unpacklo_ps(tmp0, tmp1);
    v5.data() = _mm256_unpackhi_ps(tmp0, tmp1);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3, float_v &v4, float_v &v5, float_v &v6) const/*{{{*/
{
    deinterleave(v0, v1, v2, v3);
    const m128  il0 = _mm_loadu_ps(&m_data[m_indexes[0] + 4]); // a0 b0 c0 d0
    const m128  il1 = _mm_loadu_ps(&m_data[m_indexes[1] + 4]); // a1 b1 c1 d1
    const m128  il2 = _mm_loadu_ps(&m_data[m_indexes[2] + 4]); // a2 b2 c2 d2
    const m128  il3 = _mm_loadu_ps(&m_data[m_indexes[3] + 4]); // a3 b3 c3 d3
    const m128  il4 = _mm_loadu_ps(&m_data[m_indexes[4] + 4]); // a4 b4 c4 d4
    const m128  il5 = _mm_loadu_ps(&m_data[m_indexes[5] + 4]); // a5 b5 c5 d5
    const m128  il6 = _mm_loadu_ps(&m_data[m_indexes[6] + 4]); // a6 b6 c6 d6
    const m128  il7 = _mm_loadu_ps(&m_data[m_indexes[7] + 4]); // a7 b7 c7 d7

    const m256 il04 = AVX::concat(il0, il4);
    const m256 il15 = AVX::concat(il1, il5);
    const m256 il26 = AVX::concat(il2, il6);
    const m256 il37 = AVX::concat(il3, il7);
    const m256 ab0246 = _mm256_unpacklo_ps(il04, il26);
    const m256 ab1357 = _mm256_unpacklo_ps(il15, il37);
    const m256 cd0246 = _mm256_unpackhi_ps(il04, il26);
    const m256 cd1357 = _mm256_unpackhi_ps(il15, il37);
    v4.data() = _mm256_unpacklo_ps(ab0246, ab1357);
    v5.data() = _mm256_unpackhi_ps(ab0246, ab1357);
    v6.data() = _mm256_unpacklo_ps(cd0246, cd1357);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3, float_v &v4, float_v &v5, float_v &v6, float_v &v7) const/*{{{*/
{
    deinterleave(v0, v1, v2, v3);
    const m128  il0 = _mm_loadu_ps(&m_data[m_indexes[0] + 4]); // a0 b0 c0 d0
    const m128  il1 = _mm_loadu_ps(&m_data[m_indexes[1] + 4]); // a1 b1 c1 d1
    const m128  il2 = _mm_loadu_ps(&m_data[m_indexes[2] + 4]); // a2 b2 c2 d2
    const m128  il3 = _mm_loadu_ps(&m_data[m_indexes[3] + 4]); // a3 b3 c3 d3
    const m128  il4 = _mm_loadu_ps(&m_data[m_indexes[4] + 4]); // a4 b4 c4 d4
    const m128  il5 = _mm_loadu_ps(&m_data[m_indexes[5] + 4]); // a5 b5 c5 d5
    const m128  il6 = _mm_loadu_ps(&m_data[m_indexes[6] + 4]); // a6 b6 c6 d6
    const m128  il7 = _mm_loadu_ps(&m_data[m_indexes[7] + 4]); // a7 b7 c7 d7

    const m256 il04 = AVX::concat(il0, il4);
    const m256 il15 = AVX::concat(il1, il5);
    const m256 il26 = AVX::concat(il2, il6);
    const m256 il37 = AVX::concat(il3, il7);
    const m256 ab0246 = _mm256_unpacklo_ps(il04, il26);
    const m256 ab1357 = _mm256_unpacklo_ps(il15, il37);
    const m256 cd0246 = _mm256_unpackhi_ps(il04, il26);
    const m256 cd1357 = _mm256_unpackhi_ps(il15, il37);
    v4.data() = _mm256_unpacklo_ps(ab0246, ab1357);
    v5.data() = _mm256_unpackhi_ps(ab0246, ab1357);
    v6.data() = _mm256_unpacklo_ps(cd0246, cd1357);
    v7.data() = _mm256_unpackhi_ps(cd0246, cd1357);
}/*}}}*/

template<> inline void InterleavedMemoryAccessBase<sfloat_v>::deinterleave(sfloat_v &v0, sfloat_v &v1) const/*{{{*/
{
    const m128  il0 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[0]])); // a0 b0
    const m128  il2 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[2]])); // a2 b2
    const m128  il4 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[4]])); // a4 b4
    const m128  il6 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[6]])); // a6 b6
    const m128 il01 = _mm_loadh_pi(             il0, reinterpret_cast<__m64 const *>(&m_data[m_indexes[1]])); // a0 b0 a1 b1
    const m128 il23 = _mm_loadh_pi(             il2, reinterpret_cast<__m64 const *>(&m_data[m_indexes[3]])); // a2 b2 a3 b3
    const m128 il45 = _mm_loadh_pi(             il4, reinterpret_cast<__m64 const *>(&m_data[m_indexes[5]])); // a4 b4 a5 b5
    const m128 il67 = _mm_loadh_pi(             il6, reinterpret_cast<__m64 const *>(&m_data[m_indexes[7]])); // a6 b6 a7 b7

    const m256 tmp2 = AVX::concat(il01, il45);
    const m256 tmp3 = AVX::concat(il23, il67);

    const m256 tmp0 = _mm256_unpacklo_ps(tmp2, tmp3);
    const m256 tmp1 = _mm256_unpackhi_ps(tmp2, tmp3);

    v0.data() = _mm256_unpacklo_ps(tmp0, tmp1);
    v1.data() = _mm256_unpackhi_ps(tmp0, tmp1);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<sfloat_v>::deinterleave(sfloat_v &v0, sfloat_v &v1, sfloat_v &v2) const/*{{{*/
{
    const m128  il0 = _mm_loadu_ps(&m_data[m_indexes[0]]); // a0 b0 c0 d0
    const m128  il1 = _mm_loadu_ps(&m_data[m_indexes[1]]); // a1 b1 c1 d1
    const m128  il2 = _mm_loadu_ps(&m_data[m_indexes[2]]); // a2 b2 c2 d2
    const m128  il3 = _mm_loadu_ps(&m_data[m_indexes[3]]); // a3 b3 c3 d3
    const m128  il4 = _mm_loadu_ps(&m_data[m_indexes[4]]); // a4 b4 c4 d4
    const m128  il5 = _mm_loadu_ps(&m_data[m_indexes[5]]); // a5 b5 c5 d5
    const m128  il6 = _mm_loadu_ps(&m_data[m_indexes[6]]); // a6 b6 c6 d6
    const m128  il7 = _mm_loadu_ps(&m_data[m_indexes[7]]); // a7 b7 c7 d7

    const m256 il04 = AVX::concat(il0, il4);
    const m256 il15 = AVX::concat(il1, il5);
    const m256 il26 = AVX::concat(il2, il6);
    const m256 il37 = AVX::concat(il3, il7);
    const m256 ab0246 = _mm256_unpacklo_ps(il04, il26);
    const m256 ab1357 = _mm256_unpacklo_ps(il15, il37);
    const m256 cd0246 = _mm256_unpackhi_ps(il04, il26);
    const m256 cd1357 = _mm256_unpackhi_ps(il15, il37);
    v0.data() = _mm256_unpacklo_ps(ab0246, ab1357);
    v1.data() = _mm256_unpackhi_ps(ab0246, ab1357);
    v2.data() = _mm256_unpacklo_ps(cd0246, cd1357);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<sfloat_v>::deinterleave(sfloat_v &v0, sfloat_v &v1, sfloat_v &v2, sfloat_v &v3) const/*{{{*/
{
    const m128  il0 = _mm_loadu_ps(&m_data[m_indexes[0]]); // a0 b0 c0 d0
    const m128  il1 = _mm_loadu_ps(&m_data[m_indexes[1]]); // a1 b1 c1 d1
    const m128  il2 = _mm_loadu_ps(&m_data[m_indexes[2]]); // a2 b2 c2 d2
    const m128  il3 = _mm_loadu_ps(&m_data[m_indexes[3]]); // a3 b3 c3 d3
    const m128  il4 = _mm_loadu_ps(&m_data[m_indexes[4]]); // a4 b4 c4 d4
    const m128  il5 = _mm_loadu_ps(&m_data[m_indexes[5]]); // a5 b5 c5 d5
    const m128  il6 = _mm_loadu_ps(&m_data[m_indexes[6]]); // a6 b6 c6 d6
    const m128  il7 = _mm_loadu_ps(&m_data[m_indexes[7]]); // a7 b7 c7 d7

    const m256 il04 = AVX::concat(il0, il4);
    const m256 il15 = AVX::concat(il1, il5);
    const m256 il26 = AVX::concat(il2, il6);
    const m256 il37 = AVX::concat(il3, il7);
    const m256 ab0246 = _mm256_unpacklo_ps(il04, il26);
    const m256 ab1357 = _mm256_unpacklo_ps(il15, il37);
    const m256 cd0246 = _mm256_unpackhi_ps(il04, il26);
    const m256 cd1357 = _mm256_unpackhi_ps(il15, il37);
    v0.data() = _mm256_unpacklo_ps(ab0246, ab1357);
    v1.data() = _mm256_unpackhi_ps(ab0246, ab1357);
    v2.data() = _mm256_unpacklo_ps(cd0246, cd1357);
    v3.data() = _mm256_unpackhi_ps(cd0246, cd1357);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<sfloat_v>::deinterleave(sfloat_v &v0, sfloat_v &v1, sfloat_v &v2, sfloat_v &v3, sfloat_v &v4) const/*{{{*/
{
    v4.gather(m_data, m_indexes + I(4));
    deinterleave(v0, v1, v2, v3);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<sfloat_v>::deinterleave(sfloat_v &v0, sfloat_v &v1, sfloat_v &v2, sfloat_v &v3, sfloat_v &v4, sfloat_v &v5) const/*{{{*/
{
    deinterleave(v0, v1, v2, v3);
    const m128  il0 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[0] + 4])); // a0 b0
    const m128  il2 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[2] + 4])); // a2 b2
    const m128  il4 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[4] + 4])); // a4 b4
    const m128  il6 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[6] + 4])); // a6 b6
    const m128 il01 = _mm_loadh_pi(             il0, reinterpret_cast<__m64 const *>(&m_data[m_indexes[1] + 4])); // a0 b0 a1 b1
    const m128 il23 = _mm_loadh_pi(             il2, reinterpret_cast<__m64 const *>(&m_data[m_indexes[3] + 4])); // a2 b2 a3 b3
    const m128 il45 = _mm_loadh_pi(             il4, reinterpret_cast<__m64 const *>(&m_data[m_indexes[5] + 4])); // a4 b4 a5 b5
    const m128 il67 = _mm_loadh_pi(             il6, reinterpret_cast<__m64 const *>(&m_data[m_indexes[7] + 4])); // a6 b6 a7 b7

    const m256 tmp2 = AVX::concat(il01, il45);
    const m256 tmp3 = AVX::concat(il23, il67);

    const m256 tmp0 = _mm256_unpacklo_ps(tmp2, tmp3);
    const m256 tmp1 = _mm256_unpackhi_ps(tmp2, tmp3);

    v4.data() = _mm256_unpacklo_ps(tmp0, tmp1);
    v5.data() = _mm256_unpackhi_ps(tmp0, tmp1);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<sfloat_v>::deinterleave(sfloat_v &v0, sfloat_v &v1, sfloat_v &v2, sfloat_v &v3, sfloat_v &v4, sfloat_v &v5, sfloat_v &v6) const/*{{{*/
{
    deinterleave(v0, v1, v2, v3);
    const m128  il0 = _mm_loadu_ps(&m_data[m_indexes[0] + 4]); // a0 b0 c0 d0
    const m128  il1 = _mm_loadu_ps(&m_data[m_indexes[1] + 4]); // a1 b1 c1 d1
    const m128  il2 = _mm_loadu_ps(&m_data[m_indexes[2] + 4]); // a2 b2 c2 d2
    const m128  il3 = _mm_loadu_ps(&m_data[m_indexes[3] + 4]); // a3 b3 c3 d3
    const m128  il4 = _mm_loadu_ps(&m_data[m_indexes[4] + 4]); // a4 b4 c4 d4
    const m128  il5 = _mm_loadu_ps(&m_data[m_indexes[5] + 4]); // a5 b5 c5 d5
    const m128  il6 = _mm_loadu_ps(&m_data[m_indexes[6] + 4]); // a6 b6 c6 d6
    const m128  il7 = _mm_loadu_ps(&m_data[m_indexes[7] + 4]); // a7 b7 c7 d7

    const m256 il04 = AVX::concat(il0, il4);
    const m256 il15 = AVX::concat(il1, il5);
    const m256 il26 = AVX::concat(il2, il6);
    const m256 il37 = AVX::concat(il3, il7);
    const m256 ab0246 = _mm256_unpacklo_ps(il04, il26);
    const m256 ab1357 = _mm256_unpacklo_ps(il15, il37);
    const m256 cd0246 = _mm256_unpackhi_ps(il04, il26);
    const m256 cd1357 = _mm256_unpackhi_ps(il15, il37);
    v4.data() = _mm256_unpacklo_ps(ab0246, ab1357);
    v5.data() = _mm256_unpackhi_ps(ab0246, ab1357);
    v6.data() = _mm256_unpacklo_ps(cd0246, cd1357);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<sfloat_v>::deinterleave(sfloat_v &v0, sfloat_v &v1, sfloat_v &v2, sfloat_v &v3, sfloat_v &v4, sfloat_v &v5, sfloat_v &v6, sfloat_v &v7) const/*{{{*/
{
    deinterleave(v0, v1, v2, v3);
    const m128  il0 = _mm_loadu_ps(&m_data[m_indexes[0] + 4]); // a0 b0 c0 d0
    const m128  il1 = _mm_loadu_ps(&m_data[m_indexes[1] + 4]); // a1 b1 c1 d1
    const m128  il2 = _mm_loadu_ps(&m_data[m_indexes[2] + 4]); // a2 b2 c2 d2
    const m128  il3 = _mm_loadu_ps(&m_data[m_indexes[3] + 4]); // a3 b3 c3 d3
    const m128  il4 = _mm_loadu_ps(&m_data[m_indexes[4] + 4]); // a4 b4 c4 d4
    const m128  il5 = _mm_loadu_ps(&m_data[m_indexes[5] + 4]); // a5 b5 c5 d5
    const m128  il6 = _mm_loadu_ps(&m_data[m_indexes[6] + 4]); // a6 b6 c6 d6
    const m128  il7 = _mm_loadu_ps(&m_data[m_indexes[7] + 4]); // a7 b7 c7 d7

    const m256 il04 = AVX::concat(il0, il4);
    const m256 il15 = AVX::concat(il1, il5);
    const m256 il26 = AVX::concat(il2, il6);
    const m256 il37 = AVX::concat(il3, il7);
    const m256 ab0246 = _mm256_unpacklo_ps(il04, il26);
    const m256 ab1357 = _mm256_unpacklo_ps(il15, il37);
    const m256 cd0246 = _mm256_unpackhi_ps(il04, il26);
    const m256 cd1357 = _mm256_unpackhi_ps(il15, il37);
    v4.data() = _mm256_unpacklo_ps(ab0246, ab1357);
    v5.data() = _mm256_unpackhi_ps(ab0246, ab1357);
    v6.data() = _mm256_unpacklo_ps(cd0246, cd1357);
    v7.data() = _mm256_unpackhi_ps(cd0246, cd1357);
}/*}}}*/

static Vc_ALWAYS_INLINE void _avx_deinterleave_double(const double *VC_RESTRICT data, const uint_v &indexes, double_v &v0, double_v &v1)/*{{{*/
{
    const m256d ab02 = AVX::concat(_mm_loadu_pd(&data[indexes[0]]), _mm_loadu_pd(&data[indexes[2]]));
    const m256d ab13 = AVX::concat(_mm_loadu_pd(&data[indexes[1]]), _mm_loadu_pd(&data[indexes[3]]));

    v0.data() = _mm256_unpacklo_pd(ab02, ab13);
    v1.data() = _mm256_unpackhi_pd(ab02, ab13);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<double_v>::deinterleave(double_v &v0, double_v &v1) const/*{{{*/
{
    _avx_deinterleave_double(m_data    , m_indexes, v0, v1);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<double_v>::deinterleave(double_v &v0, double_v &v1, double_v &v2) const/*{{{*/
{
    _avx_deinterleave_double(m_data    , m_indexes, v0, v1);
    v2.gather(m_data + 2, m_indexes);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<double_v>::deinterleave(double_v &v0, double_v &v1, double_v &v2, double_v &v3) const/*{{{*/
{
    _avx_deinterleave_double(m_data    , m_indexes, v0, v1);
    _avx_deinterleave_double(m_data + 2, m_indexes, v2, v3);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<double_v>::deinterleave(double_v &v0, double_v &v1, double_v &v2, double_v &v3, double_v &v4) const/*{{{*/
{
    _avx_deinterleave_double(m_data    , m_indexes, v0, v1);
    _avx_deinterleave_double(m_data + 2, m_indexes, v2, v3);
    v4.gather(m_data + 4, m_indexes);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<double_v>::deinterleave(double_v &v0, double_v &v1, double_v &v2, double_v &v3, double_v &v4, double_v &v5) const/*{{{*/
{
    _avx_deinterleave_double(m_data    , m_indexes, v0, v1);
    _avx_deinterleave_double(m_data + 2, m_indexes, v2, v3);
    _avx_deinterleave_double(m_data + 4, m_indexes, v4, v5);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<double_v>::deinterleave(double_v &v0, double_v &v1, double_v &v2, double_v &v3, double_v &v4, double_v &v5, double_v &v6) const/*{{{*/
{
    _avx_deinterleave_double(m_data    , m_indexes, v0, v1);
    _avx_deinterleave_double(m_data + 2, m_indexes, v2, v3);
    _avx_deinterleave_double(m_data + 4, m_indexes, v4, v5);
    v6.gather(m_data + 6, m_indexes);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<double_v>::deinterleave(double_v &v0, double_v &v1, double_v &v2, double_v &v3, double_v &v4, double_v &v5, double_v &v6, double_v &v7) const/*{{{*/
{
    _avx_deinterleave_double(m_data    , m_indexes, v0, v1);
    _avx_deinterleave_double(m_data + 2, m_indexes, v2, v3);
    _avx_deinterleave_double(m_data + 4, m_indexes, v4, v5);
    _avx_deinterleave_double(m_data + 6, m_indexes, v6, v7);
}/*}}}*/

template<> inline void InterleavedMemoryAccessBase<short_v>::deinterleave(short_v &v0, short_v &v1) const {/*{{{*/
    const m128i a = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&m_data[m_indexes[0]]));
    const m128i b = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&m_data[m_indexes[1]]));
    const m128i c = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&m_data[m_indexes[2]]));
    const m128i d = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&m_data[m_indexes[3]]));
    const m128i e = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&m_data[m_indexes[4]]));
    const m128i f = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&m_data[m_indexes[5]]));
    const m128i g = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&m_data[m_indexes[6]]));
    const m128i h = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&m_data[m_indexes[7]]));

    const m128i tmp2  = _mm_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4
    const m128i tmp3  = _mm_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
    const m128i tmp4  = _mm_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
    const m128i tmp5  = _mm_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7

    const m128i tmp0  = _mm_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6
    const m128i tmp1  = _mm_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7

    v0.data() = _mm_unpacklo_epi16(tmp0, tmp1);
    v1.data() = _mm_unpackhi_epi16(tmp0, tmp1);
}
/*}}}*/
template<> inline void InterleavedMemoryAccessBase<short_v>::deinterleave(short_v &v0, short_v &v1,/*{{{*/
        short_v &v2) const {
    const m128i a = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[0]]));
    const m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[1]]));
    const m128i c = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[2]]));
    const m128i d = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[3]]));
    const m128i e = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[4]]));
    const m128i f = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[5]]));
    const m128i g = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[6]]));
    const m128i h = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[7]]));

    const m128i tmp2  = _mm_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4
    const m128i tmp4  = _mm_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
    const m128i tmp3  = _mm_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
    const m128i tmp5  = _mm_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7

    const m128i tmp0  = _mm_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6
    const m128i tmp1  = _mm_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7
    const m128i tmp6  = _mm_unpackhi_epi16(tmp2, tmp3); // c0 c2 c4 c6 d0 d2 d4 d6
    const m128i tmp7  = _mm_unpackhi_epi16(tmp4, tmp5); // c1 c3 c5 c7 d1 d3 d5 d7

    v0.data() = _mm_unpacklo_epi16(tmp0, tmp1);
    v1.data() = _mm_unpackhi_epi16(tmp0, tmp1);
    v2.data() = _mm_unpacklo_epi16(tmp6, tmp7);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<short_v>::deinterleave(short_v &v0, short_v &v1,/*{{{*/
        short_v &v2, short_v &v3) const {
    const m128i a = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[0]]));
    const m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[1]]));
    const m128i c = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[2]]));
    const m128i d = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[3]]));
    const m128i e = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[4]]));
    const m128i f = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[5]]));
    const m128i g = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[6]]));
    const m128i h = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&m_data[m_indexes[7]]));

    const m128i tmp2  = _mm_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4
    const m128i tmp4  = _mm_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
    const m128i tmp3  = _mm_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
    const m128i tmp5  = _mm_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7

    const m128i tmp0  = _mm_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6
    const m128i tmp1  = _mm_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7
    const m128i tmp6  = _mm_unpackhi_epi16(tmp2, tmp3); // c0 c2 c4 c6 d0 d2 d4 d6
    const m128i tmp7  = _mm_unpackhi_epi16(tmp4, tmp5); // c1 c3 c5 c7 d1 d3 d5 d7

    v0.data() = _mm_unpacklo_epi16(tmp0, tmp1);
    v1.data() = _mm_unpackhi_epi16(tmp0, tmp1);
    v2.data() = _mm_unpacklo_epi16(tmp6, tmp7);
    v3.data() = _mm_unpackhi_epi16(tmp6, tmp7);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<short_v>::deinterleave(short_v &v0, short_v &v1,/*{{{*/
        short_v &v2, short_v &v3, short_v &v4) const {
    const m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[0]]));
    const m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[1]]));
    const m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[2]]));
    const m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[3]]));
    const m128i e = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[4]]));
    const m128i f = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[5]]));
    const m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[6]]));
    const m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[7]]));

    const m128i tmp2  = _mm_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4
    const m128i tmp4  = _mm_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
    const m128i tmp3  = _mm_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
    const m128i tmp5  = _mm_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7
    const m128i tmp10 = _mm_unpackhi_epi16(a, e); // e0 e4 f0 f4 g0 g4 h0 h4
    const m128i tmp11 = _mm_unpackhi_epi16(c, g); // e1 e5 f1 f5 g1 g5 h1 h5
    const m128i tmp12 = _mm_unpackhi_epi16(b, f); // e2 e6 f2 f6 g2 g6 h2 h6
    const m128i tmp13 = _mm_unpackhi_epi16(d, h); // e3 e7 f3 f7 g3 g7 h3 h7

    const m128i tmp0  = _mm_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6
    const m128i tmp1  = _mm_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7
    const m128i tmp6  = _mm_unpackhi_epi16(tmp2, tmp3); // c0 c2 c4 c6 d0 d2 d4 d6
    const m128i tmp7  = _mm_unpackhi_epi16(tmp4, tmp5); // c1 c3 c5 c7 d1 d3 d5 d7
    const m128i tmp8  = _mm_unpacklo_epi16(tmp10, tmp11); // e0 e2 e4 e6 f0 f2 f4 f6
    const m128i tmp9  = _mm_unpacklo_epi16(tmp12, tmp13); // e1 e3 e5 e7 f1 f3 f5 f7

    v0.data() = _mm_unpacklo_epi16(tmp0, tmp1);
    v1.data() = _mm_unpackhi_epi16(tmp0, tmp1);
    v2.data() = _mm_unpacklo_epi16(tmp6, tmp7);
    v3.data() = _mm_unpackhi_epi16(tmp6, tmp7);
    v4.data() = _mm_unpacklo_epi16(tmp8, tmp9);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<short_v>::deinterleave(short_v &v0, short_v &v1,/*{{{*/
        short_v &v2, short_v &v3, short_v &v4, short_v &v5) const {
    const m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[0]]));
    const m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[1]]));
    const m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[2]]));
    const m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[3]]));
    const m128i e = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[4]]));
    const m128i f = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[5]]));
    const m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[6]]));
    const m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[7]]));

    const m128i tmp2  = _mm_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4
    const m128i tmp4  = _mm_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
    const m128i tmp3  = _mm_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
    const m128i tmp5  = _mm_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7
    const m128i tmp10 = _mm_unpackhi_epi16(a, e); // e0 e4 f0 f4 g0 g4 h0 h4
    const m128i tmp11 = _mm_unpackhi_epi16(c, g); // e1 e5 f1 f5 g1 g5 h1 h5
    const m128i tmp12 = _mm_unpackhi_epi16(b, f); // e2 e6 f2 f6 g2 g6 h2 h6
    const m128i tmp13 = _mm_unpackhi_epi16(d, h); // e3 e7 f3 f7 g3 g7 h3 h7

    const m128i tmp0  = _mm_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6
    const m128i tmp1  = _mm_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7
    const m128i tmp6  = _mm_unpackhi_epi16(tmp2, tmp3); // c0 c2 c4 c6 d0 d2 d4 d6
    const m128i tmp7  = _mm_unpackhi_epi16(tmp4, tmp5); // c1 c3 c5 c7 d1 d3 d5 d7
    const m128i tmp8  = _mm_unpacklo_epi16(tmp10, tmp11); // e0 e2 e4 e6 f0 f2 f4 f6
    const m128i tmp9  = _mm_unpacklo_epi16(tmp12, tmp13); // e1 e3 e5 e7 f1 f3 f5 f7

    v0.data() = _mm_unpacklo_epi16(tmp0, tmp1);
    v1.data() = _mm_unpackhi_epi16(tmp0, tmp1);
    v2.data() = _mm_unpacklo_epi16(tmp6, tmp7);
    v3.data() = _mm_unpackhi_epi16(tmp6, tmp7);
    v4.data() = _mm_unpacklo_epi16(tmp8, tmp9);
    v5.data() = _mm_unpackhi_epi16(tmp8, tmp9);
}/*}}}*/
template<> inline void InterleavedMemoryAccessBase<short_v>::deinterleave(short_v &v0, short_v &v1,/*{{{*/
        short_v &v2, short_v &v3, short_v &v4, short_v &v5, short_v &v6) const {
    const m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[0]]));
    const m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[1]]));
    const m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[2]]));
    const m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[3]]));
    const m128i e = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[4]]));
    const m128i f = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[5]]));
    const m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[6]]));
    const m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[7]]));

    const m128i tmp2  = _mm_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4
    const m128i tmp4  = _mm_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
    const m128i tmp3  = _mm_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
    const m128i tmp5  = _mm_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7
    const m128i tmp10 = _mm_unpackhi_epi16(a, e); // e0 e4 f0 f4 g0 g4 h0 h4
    const m128i tmp11 = _mm_unpackhi_epi16(c, g); // e1 e5 f1 f5 g1 g5 h1 h5
    const m128i tmp12 = _mm_unpackhi_epi16(b, f); // e2 e6 f2 f6 g2 g6 h2 h6
    const m128i tmp13 = _mm_unpackhi_epi16(d, h); // e3 e7 f3 f7 g3 g7 h3 h7

    const m128i tmp0  = _mm_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6
    const m128i tmp1  = _mm_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7
    const m128i tmp6  = _mm_unpackhi_epi16(tmp2, tmp3); // c0 c2 c4 c6 d0 d2 d4 d6
    const m128i tmp7  = _mm_unpackhi_epi16(tmp4, tmp5); // c1 c3 c5 c7 d1 d3 d5 d7
    const m128i tmp8  = _mm_unpacklo_epi16(tmp10, tmp11); // e0 e2 e4 e6 f0 f2 f4 f6
    const m128i tmp9  = _mm_unpacklo_epi16(tmp12, tmp13); // e1 e3 e5 e7 f1 f3 f5 f7
    const m128i tmp14 = _mm_unpackhi_epi16(tmp10, tmp11); // g0 g2 g4 g6 h0 h2 h4 h6
    const m128i tmp15 = _mm_unpackhi_epi16(tmp12, tmp13); // g1 g3 g5 g7 h1 h3 h5 h7

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
    const m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[0]]));
    const m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[1]]));
    const m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[2]]));
    const m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[3]]));
    const m128i e = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[4]]));
    const m128i f = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[5]]));
    const m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[6]]));
    const m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_data[m_indexes[7]]));

    const m128i tmp2  = _mm_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4
    const m128i tmp4  = _mm_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
    const m128i tmp3  = _mm_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
    const m128i tmp5  = _mm_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7
    const m128i tmp10 = _mm_unpackhi_epi16(a, e); // e0 e4 f0 f4 g0 g4 h0 h4
    const m128i tmp11 = _mm_unpackhi_epi16(c, g); // e1 e5 f1 f5 g1 g5 h1 h5
    const m128i tmp12 = _mm_unpackhi_epi16(b, f); // e2 e6 f2 f6 g2 g6 h2 h6
    const m128i tmp13 = _mm_unpackhi_epi16(d, h); // e3 e7 f3 f7 g3 g7 h3 h7

    const m128i tmp0  = _mm_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6
    const m128i tmp1  = _mm_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7
    const m128i tmp6  = _mm_unpackhi_epi16(tmp2, tmp3); // c0 c2 c4 c6 d0 d2 d4 d6
    const m128i tmp7  = _mm_unpackhi_epi16(tmp4, tmp5); // c1 c3 c5 c7 d1 d3 d5 d7
    const m128i tmp8  = _mm_unpacklo_epi16(tmp10, tmp11); // e0 e2 e4 e6 f0 f2 f4 f6
    const m128i tmp9  = _mm_unpacklo_epi16(tmp12, tmp13); // e1 e3 e5 e7 f1 f3 f5 f7
    const m128i tmp14 = _mm_unpackhi_epi16(tmp10, tmp11); // g0 g2 g4 g6 h0 h2 h4 h6
    const m128i tmp15 = _mm_unpackhi_epi16(tmp12, tmp13); // g1 g3 g5 g7 h1 h3 h5 h7

    v0.data() = _mm_unpacklo_epi16(tmp0, tmp1);
    v1.data() = _mm_unpackhi_epi16(tmp0, tmp1);
    v2.data() = _mm_unpacklo_epi16(tmp6, tmp7);
    v3.data() = _mm_unpackhi_epi16(tmp6, tmp7);
    v4.data() = _mm_unpacklo_epi16(tmp8, tmp9);
    v5.data() = _mm_unpackhi_epi16(tmp8, tmp9);
    v6.data() = _mm_unpacklo_epi16(tmp14, tmp15);
    v7.data() = _mm_unpackhi_epi16(tmp14, tmp15);
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

#endif // VC_AVX_INTERLEAVEDMEMORY_TCC

// vim: foldmethod=marker
