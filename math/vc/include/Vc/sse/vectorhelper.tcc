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

#include "casts.h"
#include <cstdlib>

namespace ROOT {
namespace Vc
{
namespace SSE
{

////////////////////////////////////////////////////////////////////////////////////////////////////
// float_v
template<> Vc_ALWAYS_INLINE Vc_PURE _M128 VectorHelper<_M128>::load(const float *x, AlignedFlag)
{
    return _mm_load_ps(x);
}

template<> Vc_ALWAYS_INLINE Vc_PURE _M128 VectorHelper<_M128>::load(const float *x, UnalignedFlag)
{
    return _mm_loadu_ps(x);
}

template<> Vc_ALWAYS_INLINE Vc_PURE _M128 VectorHelper<_M128>::load(const float *x, StreamingAndAlignedFlag)
{
    return _mm_stream_load(x);
}

template<> Vc_ALWAYS_INLINE Vc_PURE _M128 VectorHelper<_M128>::load(const float *x, StreamingAndUnalignedFlag)
{
    return load(x, Unaligned);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// stores
Vc_ALWAYS_INLINE void VectorHelper<_M128>::store(float *mem, const VectorType x, AlignedFlag)
{
    _mm_store_ps(mem, x);
}
Vc_ALWAYS_INLINE void VectorHelper<_M128>::store(float *mem, const VectorType x, UnalignedFlag)
{
    _mm_storeu_ps(mem, x);
}
Vc_ALWAYS_INLINE void VectorHelper<_M128>::store(float *mem, const VectorType x, StreamingAndAlignedFlag)
{
    _mm_stream_ps(mem, x);
}
Vc_ALWAYS_INLINE void VectorHelper<_M128>::store(float *mem, const VectorType x, StreamingAndUnalignedFlag)
{
    _mm_maskmoveu_si128(_mm_castps_si128(x), _mm_setallone_si128(), reinterpret_cast<char *>(mem));
}
Vc_ALWAYS_INLINE void VectorHelper<_M128>::store(float *mem, const VectorType x, const VectorType m, AlignedFlag)
{
    _mm_store_ps(mem, mm_blendv_ps(_mm_load_ps(mem), x, m));
}
Vc_ALWAYS_INLINE void VectorHelper<_M128>::store(float *mem, const VectorType x, const VectorType m, UnalignedFlag)
{
    _mm_storeu_ps(mem, mm_blendv_ps(_mm_loadu_ps(mem), x, m));
}
Vc_ALWAYS_INLINE void VectorHelper<_M128>::store(float *mem, const VectorType x, const VectorType m, StreamingAndAlignedFlag)
{
    _mm_maskmoveu_si128(_mm_castps_si128(x), _mm_castps_si128(m), reinterpret_cast<char *>(mem));
}
Vc_ALWAYS_INLINE void VectorHelper<_M128>::store(float *mem, const VectorType x, const VectorType m, StreamingAndUnalignedFlag)
{
    _mm_maskmoveu_si128(_mm_castps_si128(x), _mm_castps_si128(m), reinterpret_cast<char *>(mem));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// sfloat_v
template<> Vc_ALWAYS_INLINE Vc_PURE M256 VectorHelper<M256>::load(const float *x, AlignedFlag)
{
    return VectorType::create(_mm_load_ps(x), _mm_load_ps(x + 4));
}

template<> Vc_ALWAYS_INLINE Vc_PURE M256 VectorHelper<M256>::load(const float *x, UnalignedFlag)
{
    return VectorType::create(_mm_loadu_ps(x), _mm_loadu_ps(x + 4));
}

template<> Vc_ALWAYS_INLINE Vc_PURE M256 VectorHelper<M256>::load(const float *x, StreamingAndAlignedFlag)
{
    return VectorType::create(_mm_stream_load(&x[0]), _mm_stream_load(&x[4]));
}

template<> Vc_ALWAYS_INLINE Vc_PURE M256 VectorHelper<M256>::load(const float *x, StreamingAndUnalignedFlag)
{
    return load(x, Unaligned);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// stores
Vc_ALWAYS_INLINE void VectorHelper<M256>::store(float *mem, VectorTypeArg x, AlignedFlag)
{
    _mm_store_ps(mem, x[0]);
    _mm_store_ps(mem + 4, x[1]);
}
Vc_ALWAYS_INLINE void VectorHelper<M256>::store(float *mem, VectorTypeArg x, UnalignedFlag)
{
    _mm_storeu_ps(mem, x[0]);
    _mm_storeu_ps(mem + 4, x[1]);
}
Vc_ALWAYS_INLINE void VectorHelper<M256>::store(float *mem, VectorTypeArg x, StreamingAndAlignedFlag)
{
    _mm_stream_ps(mem, x[0]);
    _mm_stream_ps(mem + 4, x[1]);
}
Vc_ALWAYS_INLINE void VectorHelper<M256>::store(float *mem, VectorTypeArg x, StreamingAndUnalignedFlag)
{
    _mm_maskmoveu_si128(_mm_castps_si128(x[0]), _mm_setallone_si128(), reinterpret_cast<char *>(mem));
    _mm_maskmoveu_si128(_mm_castps_si128(x[1]), _mm_setallone_si128(), reinterpret_cast<char *>(mem + 4));
}
Vc_ALWAYS_INLINE void VectorHelper<M256>::store(float *mem, VectorTypeArg x, VectorTypeArg m, AlignedFlag)
{
    _mm_store_ps(mem, mm_blendv_ps(_mm_load_ps(mem), x[0], m[0]));
    _mm_store_ps(mem + 4, mm_blendv_ps(_mm_load_ps(mem + 4), x[1], m[1]));
}
Vc_ALWAYS_INLINE void VectorHelper<M256>::store(float *mem, VectorTypeArg x, VectorTypeArg m, UnalignedFlag)
{
    _mm_storeu_ps(mem, mm_blendv_ps(_mm_loadu_ps(mem), x[0], m[0]));
    _mm_storeu_ps(mem + 4, mm_blendv_ps(_mm_loadu_ps(mem + 4), x[1], m[1]));
}
Vc_ALWAYS_INLINE void VectorHelper<M256>::store(float *mem, VectorTypeArg x, VectorTypeArg m, StreamingAndAlignedFlag)
{
    _mm_maskmoveu_si128(_mm_castps_si128(x[0]), _mm_castps_si128(m[0]), reinterpret_cast<char *>(mem));
    _mm_maskmoveu_si128(_mm_castps_si128(x[1]), _mm_castps_si128(m[1]), reinterpret_cast<char *>(mem + 4));
}
Vc_ALWAYS_INLINE void VectorHelper<M256>::store(float *mem, VectorTypeArg x, VectorTypeArg m, StreamingAndUnalignedFlag)
{
    _mm_maskmoveu_si128(_mm_castps_si128(x[0]), _mm_castps_si128(m[0]), reinterpret_cast<char *>(mem));
    _mm_maskmoveu_si128(_mm_castps_si128(x[1]), _mm_castps_si128(m[1]), reinterpret_cast<char *>(mem + 4));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// double_v
template<> Vc_ALWAYS_INLINE Vc_PURE _M128D VectorHelper<_M128D>::load(const double *x, AlignedFlag)
{
    return _mm_load_pd(x);
}

template<> Vc_ALWAYS_INLINE Vc_PURE _M128D VectorHelper<_M128D>::load(const double *x, UnalignedFlag)
{
    return _mm_loadu_pd(x);
}

template<> Vc_ALWAYS_INLINE Vc_PURE _M128D VectorHelper<_M128D>::load(const double *x, StreamingAndAlignedFlag)
{
    return _mm_stream_load(x);
}

template<> Vc_ALWAYS_INLINE Vc_PURE _M128D VectorHelper<_M128D>::load(const double *x, StreamingAndUnalignedFlag)
{
    return load(x, Unaligned);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// stores
Vc_ALWAYS_INLINE void VectorHelper<_M128D>::store(double *mem, const VectorType x, AlignedFlag)
{
    _mm_store_pd(mem, x);
}
Vc_ALWAYS_INLINE void VectorHelper<_M128D>::store(double *mem, const VectorType x, UnalignedFlag)
{
    _mm_storeu_pd(mem, x);
}
Vc_ALWAYS_INLINE void VectorHelper<_M128D>::store(double *mem, const VectorType x, StreamingAndAlignedFlag)
{
    _mm_stream_pd(mem, x);
}
Vc_ALWAYS_INLINE void VectorHelper<_M128D>::store(double *mem, const VectorType x, StreamingAndUnalignedFlag)
{
    _mm_maskmoveu_si128(_mm_castpd_si128(x), _mm_setallone_si128(), reinterpret_cast<char *>(mem));
}
Vc_ALWAYS_INLINE void VectorHelper<_M128D>::store(double *mem, const VectorType x, const VectorType m, AlignedFlag)
{
    _mm_store_pd(mem, mm_blendv_pd(_mm_load_pd(mem), x, m));
}
Vc_ALWAYS_INLINE void VectorHelper<_M128D>::store(double *mem, const VectorType x, const VectorType m, UnalignedFlag)
{
    _mm_storeu_pd(mem, mm_blendv_pd(_mm_loadu_pd(mem), x, m));
}
Vc_ALWAYS_INLINE void VectorHelper<_M128D>::store(double *mem, const VectorType x, const VectorType m, StreamingAndAlignedFlag)
{
    _mm_maskmoveu_si128(_mm_castpd_si128(x), _mm_castpd_si128(m), reinterpret_cast<char *>(mem));
}
Vc_ALWAYS_INLINE void VectorHelper<_M128D>::store(double *mem, const VectorType x, const VectorType m, StreamingAndUnalignedFlag)
{
    _mm_maskmoveu_si128(_mm_castpd_si128(x), _mm_castpd_si128(m), reinterpret_cast<char *>(mem));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// int_v, uint_v, short_v, ushort_v
template<typename T> Vc_ALWAYS_INLINE Vc_PURE _M128I VectorHelper<_M128I>::load(const T *x, AlignedFlag)
{
    return _mm_load_si128(reinterpret_cast<const VectorType *>(x));
}

template<typename T> Vc_ALWAYS_INLINE Vc_PURE _M128I VectorHelper<_M128I>::load(const T *x, UnalignedFlag)
{
    return _mm_loadu_si128(reinterpret_cast<const VectorType *>(x));
}

template<typename T> Vc_ALWAYS_INLINE Vc_PURE _M128I VectorHelper<_M128I>::load(const T *x, StreamingAndAlignedFlag)
{
    return _mm_stream_load(x);
}

template<typename T> Vc_ALWAYS_INLINE Vc_PURE _M128I VectorHelper<_M128I>::load(const T *x, StreamingAndUnalignedFlag)
{
    return load(x, Unaligned);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// stores
template<typename T> Vc_ALWAYS_INLINE void VectorHelper<_M128I>::store(T *mem, const VectorType x, AlignedFlag)
{
    _mm_store_si128(reinterpret_cast<VectorType *>(mem), x);
}
template<typename T> Vc_ALWAYS_INLINE void VectorHelper<_M128I>::store(T *mem, const VectorType x, UnalignedFlag)
{
    _mm_storeu_si128(reinterpret_cast<VectorType *>(mem), x);
}
template<typename T> Vc_ALWAYS_INLINE void VectorHelper<_M128I>::store(T *mem, const VectorType x, StreamingAndAlignedFlag)
{
    _mm_stream_si128(reinterpret_cast<VectorType *>(mem), x);
}
template<typename T> Vc_ALWAYS_INLINE void VectorHelper<_M128I>::store(T *mem, const VectorType x, StreamingAndUnalignedFlag)
{
    _mm_maskmoveu_si128(x, _mm_setallone_si128(), reinterpret_cast<char *>(mem));
}
template<typename T> Vc_ALWAYS_INLINE void VectorHelper<_M128I>::store(T *mem, const VectorType x, const VectorType m, AlignedFlag align)
{
    store(mem, mm_blendv_epi8(load(mem, align), x, m), align);
}
template<typename T> Vc_ALWAYS_INLINE void VectorHelper<_M128I>::store(T *mem, const VectorType x, const VectorType m, UnalignedFlag align)
{
    store(mem, mm_blendv_epi8(load(mem, align), x, m), align);
}
template<typename T> Vc_ALWAYS_INLINE void VectorHelper<_M128I>::store(T *mem, const VectorType x, const VectorType m, StreamingAndAlignedFlag)
{
    _mm_maskmoveu_si128(x, m, reinterpret_cast<char *>(mem));
}
template<typename T> Vc_ALWAYS_INLINE void VectorHelper<_M128I>::store(T *mem, const VectorType x, const VectorType m, StreamingAndUnalignedFlag)
{
    _mm_maskmoveu_si128(x, m, reinterpret_cast<char *>(mem));
}

    template<> inline Vc_CONST _M128I SortHelper<_M128I, 8>::sort(_M128I x)
    {
        _M128I lo, hi, y;
        // sort pairs
        y = Mem::permute<X1, X0, X3, X2, X5, X4, X7, X6>(x);
        lo = _mm_min_epi16(x, y);
        hi = _mm_max_epi16(x, y);
        x = mm_blend_epi16(lo, hi, 0xaa);

        // merge left and right quads
        y = Mem::permute<X3, X2, X1, X0, X7, X6, X5, X4>(x);
        lo = _mm_min_epi16(x, y);
        hi = _mm_max_epi16(x, y);
        x = mm_blend_epi16(lo, hi, 0xcc);
        y = _mm_srli_si128(x, 2);
        lo = _mm_min_epi16(x, y);
        hi = _mm_max_epi16(x, y);
        x = mm_blend_epi16(lo, _mm_slli_si128(hi, 2), 0xaa);

        // merge quads into octs
        y = _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 3, 2));
        y = _mm_shufflelo_epi16(y, _MM_SHUFFLE(0, 1, 2, 3));
        lo = _mm_min_epi16(x, y);
        hi = _mm_max_epi16(x, y);

        x = _mm_unpacklo_epi16(lo, hi);
        y = _mm_srli_si128(x, 8);
        lo = _mm_min_epi16(x, y);
        hi = _mm_max_epi16(x, y);

        x = _mm_unpacklo_epi16(lo, hi);
        y = _mm_srli_si128(x, 8);
        lo = _mm_min_epi16(x, y);
        hi = _mm_max_epi16(x, y);

        return _mm_unpacklo_epi16(lo, hi);
    }
    template<> inline Vc_CONST _M128I SortHelper<_M128I, 4>::sort(_M128I x)
    {
        /*
        // in 16,67% of the cases the merge can be replaced by an append

        // x = [a b c d]
        // y = [c d a b]
        _M128I y = _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 3, 2));
        _M128I l = mm_min_epi32(x, y); // min[ac bd ac bd]
        _M128I h = mm_max_epi32(x, y); // max[ac bd ac bd]
        if (IS_UNLIKELY(_mm_cvtsi128_si32(h) <= l[1])) { // l[0] < h[0] < l[1] < h[1]
            return _mm_unpacklo_epi32(l, h);
        }
        // h[0] > l[1]
        */

        // sort pairs
        _M128I y = _mm_shuffle_epi32(x, _MM_SHUFFLE(2, 3, 0, 1));
        _M128I l = mm_min_epi32(x, y);
        _M128I h = mm_max_epi32(x, y);
        x = _mm_unpacklo_epi32(l, h);
        y = _mm_unpackhi_epi32(h, l);

        // sort quads
        l = mm_min_epi32(x, y);
        h = mm_max_epi32(x, y);
        x = _mm_unpacklo_epi32(l, h);
        y = _mm_unpackhi_epi64(x, x);

        l = mm_min_epi32(x, y);
        h = mm_max_epi32(x, y);
        return _mm_unpacklo_epi32(l, h);
    }
    template<> inline Vc_CONST _M128 SortHelper<_M128, 4>::sort(_M128 x)
    {
        _M128 y = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
        _M128 l = _mm_min_ps(x, y);
        _M128 h = _mm_max_ps(x, y);
        x = _mm_unpacklo_ps(l, h);
        y = _mm_unpackhi_ps(h, l);

        l = _mm_min_ps(x, y);
        h = _mm_max_ps(x, y);
        x = _mm_unpacklo_ps(l, h);
        y = _mm_movehl_ps(x, x);

        l = _mm_min_ps(x, y);
        h = _mm_max_ps(x, y);
        return _mm_unpacklo_ps(l, h);
//X         _M128 k = _mm_cmpgt_ps(x, y);
//X         k = _mm_shuffle_ps(k, k, _MM_SHUFFLE(2, 2, 0, 0));
//X         x = mm_blendv_ps(x, y, k);
//X         y = _mm_shuffle_ps(x, x, _MM_SHUFFLE(1, 0, 3, 2));
//X         k = _mm_cmpgt_ps(x, y);
//X         k = _mm_shuffle_ps(k, k, _MM_SHUFFLE(1, 0, 1, 0));
//X         x = mm_blendv_ps(x, y, k);
//X         y = _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 1, 2, 0));
//X         k = _mm_cmpgt_ps(x, y);
//X         k = _mm_shuffle_ps(k, k, _MM_SHUFFLE(0, 1, 1, 0));
//X         return mm_blendv_ps(x, y, k);
    }
    template<> inline Vc_PURE M256 SortHelper<M256, 8>::sort(const M256 &_x)
    {
        M256 x = _x;
        typedef SortHelper<_M128, 4> H;

        _M128 a, b, l, h;
        a = H::sort(x[0]);
        b = H::sort(x[1]);

        // merge
        b = _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 1, 2, 3));
        l = _mm_min_ps(a, b);
        h = _mm_max_ps(a, b);

        a = _mm_unpacklo_ps(l, h);
        b = _mm_unpackhi_ps(l, h);
        l = _mm_min_ps(a, b);
        h = _mm_max_ps(a, b);

        a = _mm_unpacklo_ps(l, h);
        b = _mm_unpackhi_ps(l, h);
        l = _mm_min_ps(a, b);
        h = _mm_max_ps(a, b);

        x[0] = _mm_unpacklo_ps(l, h);
        x[1] = _mm_unpackhi_ps(l, h);
        return x;
    }
    template<> inline Vc_CONST _M128D SortHelper<_M128D, 2>::sort(_M128D x)
    {
        const _M128D y = _mm_shuffle_pd(x, x, _MM_SHUFFLE2(0, 1));
        return _mm_unpacklo_pd(_mm_min_sd(x, y), _mm_max_sd(x, y));
    }

    // can be used to multiply with a constant. For some special constants it doesn't need an extra
    // vector but can use a shift instead, basically encoding the factor in the instruction.
    template<typename IndexType, unsigned int constant> Vc_ALWAYS_INLINE Vc_CONST IndexType mulConst(const IndexType x) {
        typedef VectorHelper<typename IndexType::EntryType> H;
        switch (constant) {
            case    0: return H::zero();
            case    1: return x;
            case    2: return H::slli(x.data(),  1);
            case    4: return H::slli(x.data(),  2);
            case    8: return H::slli(x.data(),  3);
            case   16: return H::slli(x.data(),  4);
            case   32: return H::slli(x.data(),  5);
            case   64: return H::slli(x.data(),  6);
            case  128: return H::slli(x.data(),  7);
            case  256: return H::slli(x.data(),  8);
            case  512: return H::slli(x.data(),  9);
            case 1024: return H::slli(x.data(), 10);
            case 2048: return H::slli(x.data(), 11);
        }
#ifndef VC_IMPL_SSE4_1
        // without SSE 4.1 int multiplication is not so nice
        if (sizeof(typename IndexType::EntryType) == 4) {
            switch (constant) {
                case    3: return H::add(        x.data()    , H::slli(x.data(),  1));
                case    5: return H::add(        x.data()    , H::slli(x.data(),  2));
                case    9: return H::add(        x.data()    , H::slli(x.data(),  3));
                case   17: return H::add(        x.data()    , H::slli(x.data(),  4));
                case   33: return H::add(        x.data()    , H::slli(x.data(),  5));
                case   65: return H::add(        x.data()    , H::slli(x.data(),  6));
                case  129: return H::add(        x.data()    , H::slli(x.data(),  7));
                case  257: return H::add(        x.data()    , H::slli(x.data(),  8));
                case  513: return H::add(        x.data()    , H::slli(x.data(),  9));
                case 1025: return H::add(        x.data()    , H::slli(x.data(), 10));
                case 2049: return H::add(        x.data()    , H::slli(x.data(), 11));
                case    6: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  2));
                case   10: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  3));
                case   18: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  4));
                case   34: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  5));
                case   66: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  6));
                case  130: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  7));
                case  258: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  8));
                case  514: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  9));
                case 1026: return H::add(H::slli(x.data(), 1), H::slli(x.data(), 10));
                case 2050: return H::add(H::slli(x.data(), 1), H::slli(x.data(), 11));
                case   12: return H::add(H::slli(x.data(), 2), H::slli(x.data(),  3));
                case   20: return H::add(H::slli(x.data(), 2), H::slli(x.data(),  4));
                case   36: return H::add(H::slli(x.data(), 2), H::slli(x.data(),  5));
                case   68: return H::add(H::slli(x.data(), 2), H::slli(x.data(),  6));
                case  132: return H::add(H::slli(x.data(), 2), H::slli(x.data(),  7));
                case  260: return H::add(H::slli(x.data(), 2), H::slli(x.data(),  8));
                case  516: return H::add(H::slli(x.data(), 2), H::slli(x.data(),  9));
                case 1028: return H::add(H::slli(x.data(), 2), H::slli(x.data(), 10));
                case 2052: return H::add(H::slli(x.data(), 2), H::slli(x.data(), 11));
                case   24: return H::add(H::slli(x.data(), 3), H::slli(x.data(),  4));
                case   40: return H::add(H::slli(x.data(), 3), H::slli(x.data(),  5));
                case   72: return H::add(H::slli(x.data(), 3), H::slli(x.data(),  6));
                case  136: return H::add(H::slli(x.data(), 3), H::slli(x.data(),  7));
                case  264: return H::add(H::slli(x.data(), 3), H::slli(x.data(),  8));
                case  520: return H::add(H::slli(x.data(), 3), H::slli(x.data(),  9));
                case 1032: return H::add(H::slli(x.data(), 3), H::slli(x.data(), 10));
                case 2056: return H::add(H::slli(x.data(), 3), H::slli(x.data(), 11));
                case   48: return H::add(H::slli(x.data(), 4), H::slli(x.data(),  5));
                case   80: return H::add(H::slli(x.data(), 4), H::slli(x.data(),  6));
                case  144: return H::add(H::slli(x.data(), 4), H::slli(x.data(),  7));
                case  272: return H::add(H::slli(x.data(), 4), H::slli(x.data(),  8));
                case  528: return H::add(H::slli(x.data(), 4), H::slli(x.data(),  9));
                case 1040: return H::add(H::slli(x.data(), 4), H::slli(x.data(), 10));
                case 2064: return H::add(H::slli(x.data(), 4), H::slli(x.data(), 11));
                case   96: return H::add(H::slli(x.data(), 5), H::slli(x.data(),  6));
                case  160: return H::add(H::slli(x.data(), 5), H::slli(x.data(),  7));
                case  288: return H::add(H::slli(x.data(), 5), H::slli(x.data(),  8));
                case  544: return H::add(H::slli(x.data(), 5), H::slli(x.data(),  9));
                case 1056: return H::add(H::slli(x.data(), 5), H::slli(x.data(), 10));
                case 2080: return H::add(H::slli(x.data(), 5), H::slli(x.data(), 11));
                case  192: return H::add(H::slli(x.data(), 6), H::slli(x.data(),  7));
                case  320: return H::add(H::slli(x.data(), 6), H::slli(x.data(),  8));
                case  576: return H::add(H::slli(x.data(), 6), H::slli(x.data(),  9));
                case 1088: return H::add(H::slli(x.data(), 6), H::slli(x.data(), 10));
                case 2112: return H::add(H::slli(x.data(), 6), H::slli(x.data(), 11));
                case  384: return H::add(H::slli(x.data(), 7), H::slli(x.data(),  8));
                case  640: return H::add(H::slli(x.data(), 7), H::slli(x.data(),  9));
                case 1152: return H::add(H::slli(x.data(), 7), H::slli(x.data(), 10));
                case 2176: return H::add(H::slli(x.data(), 7), H::slli(x.data(), 11));
                case  768: return H::add(H::slli(x.data(), 8), H::slli(x.data(),  9));
                case 1280: return H::add(H::slli(x.data(), 8), H::slli(x.data(), 10));
                case 2304: return H::add(H::slli(x.data(), 8), H::slli(x.data(), 11));
                case 1536: return H::add(H::slli(x.data(), 9), H::slli(x.data(), 10));
                case 2560: return H::add(H::slli(x.data(), 9), H::slli(x.data(), 11));
                case 3072: return H::add(H::slli(x.data(),10), H::slli(x.data(), 11));
            }
        }
#endif
        return H::mul(x.data(), H::set(constant));
    }
} // namespace SSE
} // namespace Vc
} // namespace ROOT
