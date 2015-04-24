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

#ifndef SSE_INTRINSICS_H
#define SSE_INTRINSICS_H

#include "../common/windows_fix_intrin.h"

// The GCC xxxintrin.h headers do not make sure that the intrinsics have C linkage. This not really
// a problem, unless there is another place where the exact same functions are declared. Then the
// linkage must be the same, otherwise it won't compile. Such a case occurs on Windows, where the
// intrin.h header (included indirectly via unistd.h) declares many SSE intrinsics again.
extern "C" {
// MMX
#include <mmintrin.h>
// SSE
#include <xmmintrin.h>
// SSE2
#include <emmintrin.h>
}

#include "../common/fix_clang_emmintrin.h"

#include "const_data.h"
#include <cstdlib>
#include "macros.h"

#ifdef __3dNOW__
extern "C" {
#include <mm3dnow.h>
}
#endif

namespace ROOT {
namespace Vc
{
namespace SSE
{
    enum VectorAlignmentEnum { VectorAlignment = 16 };

#if defined(VC_GCC) && VC_GCC < 0x40600 && !defined(VC_DONT_FIX_SSE_SHIFT)
    static Vc_INTRINSIC Vc_CONST __m128i _mm_sll_epi16(__m128i a, __m128i count) { __asm__("psllw %1,%0" : "+x"(a) : "x"(count)); return a; }
    static Vc_INTRINSIC Vc_CONST __m128i _mm_sll_epi32(__m128i a, __m128i count) { __asm__("pslld %1,%0" : "+x"(a) : "x"(count)); return a; }
    static Vc_INTRINSIC Vc_CONST __m128i _mm_sll_epi64(__m128i a, __m128i count) { __asm__("psllq %1,%0" : "+x"(a) : "x"(count)); return a; }
    static Vc_INTRINSIC Vc_CONST __m128i _mm_srl_epi16(__m128i a, __m128i count) { __asm__("psrlw %1,%0" : "+x"(a) : "x"(count)); return a; }
    static Vc_INTRINSIC Vc_CONST __m128i _mm_srl_epi32(__m128i a, __m128i count) { __asm__("psrld %1,%0" : "+x"(a) : "x"(count)); return a; }
    static Vc_INTRINSIC Vc_CONST __m128i _mm_srl_epi64(__m128i a, __m128i count) { __asm__("psrlq %1,%0" : "+x"(a) : "x"(count)); return a; }
#endif

#ifdef VC_GCC
    // Redefine the mul/add/sub intrinsics to use GCC-specific operators instead of builtin
    // functions. This way the fp-contraction optimization step kicks in and creates FMAs! :)
    static Vc_INTRINSIC Vc_CONST __m128d _mm_mul_pd(__m128d a, __m128d b) { return static_cast<__m128d>(static_cast<__v2df>(a) * static_cast<__v2df>(b)); }
    static Vc_INTRINSIC Vc_CONST __m128d _mm_add_pd(__m128d a, __m128d b) { return static_cast<__m128d>(static_cast<__v2df>(a) + static_cast<__v2df>(b)); }
    static Vc_INTRINSIC Vc_CONST __m128d _mm_sub_pd(__m128d a, __m128d b) { return static_cast<__m128d>(static_cast<__v2df>(a) - static_cast<__v2df>(b)); }
    static Vc_INTRINSIC Vc_CONST __m128  _mm_mul_ps(__m128  a, __m128  b) { return static_cast<__m128 >(static_cast<__v4sf>(a) * static_cast<__v4sf>(b)); }
    static Vc_INTRINSIC Vc_CONST __m128  _mm_add_ps(__m128  a, __m128  b) { return static_cast<__m128 >(static_cast<__v4sf>(a) + static_cast<__v4sf>(b)); }
    static Vc_INTRINSIC Vc_CONST __m128  _mm_sub_ps(__m128  a, __m128  b) { return static_cast<__m128 >(static_cast<__v4sf>(a) - static_cast<__v4sf>(b)); }
#endif

#if defined(VC_GNU_ASM) && !defined(NVALGRIND)
    static Vc_INTRINSIC __m128i Vc_CONST _mm_setallone() { __m128i r; __asm__("pcmpeqb %0,%0":"=x"(r)); return r; }
#else
    static Vc_INTRINSIC __m128i Vc_CONST _mm_setallone() { __m128i r = _mm_setzero_si128(); return _mm_cmpeq_epi8(r, r); }
#endif
    static Vc_INTRINSIC __m128i Vc_CONST _mm_setallone_si128() { return _mm_setallone(); }
    static Vc_INTRINSIC __m128d Vc_CONST _mm_setallone_pd() { return _mm_castsi128_pd(_mm_setallone()); }
    static Vc_INTRINSIC __m128  Vc_CONST _mm_setallone_ps() { return _mm_castsi128_ps(_mm_setallone()); }

    static Vc_INTRINSIC __m128i Vc_CONST _mm_setone_epi8 ()  { return _mm_set1_epi8(1); }
    static Vc_INTRINSIC __m128i Vc_CONST _mm_setone_epu8 ()  { return _mm_setone_epi8(); }
    static Vc_INTRINSIC __m128i Vc_CONST _mm_setone_epi16()  { return _mm_load_si128(reinterpret_cast<const __m128i *>(c_general::one16)); }
    static Vc_INTRINSIC __m128i Vc_CONST _mm_setone_epu16()  { return _mm_setone_epi16(); }
    static Vc_INTRINSIC __m128i Vc_CONST _mm_setone_epi32()  { return _mm_load_si128(reinterpret_cast<const __m128i *>(c_general::one32)); }
    static Vc_INTRINSIC __m128i Vc_CONST _mm_setone_epu32()  { return _mm_setone_epi32(); }

    static Vc_INTRINSIC __m128  Vc_CONST _mm_setone_ps()     { return _mm_load_ps(c_general::oneFloat); }
    static Vc_INTRINSIC __m128d Vc_CONST _mm_setone_pd()     { return _mm_load_pd(c_general::oneDouble); }

    static Vc_INTRINSIC __m128d Vc_CONST _mm_setabsmask_pd() { return _mm_load_pd(reinterpret_cast<const double *>(c_general::absMaskDouble)); }
    static Vc_INTRINSIC __m128  Vc_CONST _mm_setabsmask_ps() { return _mm_load_ps(reinterpret_cast<const float *>(c_general::absMaskFloat)); }
    static Vc_INTRINSIC __m128d Vc_CONST _mm_setsignmask_pd(){ return _mm_load_pd(reinterpret_cast<const double *>(c_general::signMaskDouble)); }
    static Vc_INTRINSIC __m128  Vc_CONST _mm_setsignmask_ps(){ return _mm_load_ps(reinterpret_cast<const float *>(c_general::signMaskFloat)); }

    //X         static Vc_INTRINSIC __m128i Vc_CONST _mm_setmin_epi8 () { return _mm_slli_epi8 (_mm_setallone_si128(),  7); }
    static Vc_INTRINSIC __m128i Vc_CONST _mm_setmin_epi16() { return _mm_load_si128(reinterpret_cast<const __m128i *>(c_general::minShort)); }
    static Vc_INTRINSIC __m128i Vc_CONST _mm_setmin_epi32() { return _mm_load_si128(reinterpret_cast<const __m128i *>(c_general::signMaskFloat)); }

    //X         static Vc_INTRINSIC __m128i Vc_CONST _mm_cmplt_epu8 (__m128i a, __m128i b) { return _mm_cmplt_epi8 (
    //X                 _mm_xor_si128(a, _mm_setmin_epi8 ()), _mm_xor_si128(b, _mm_setmin_epi8 ())); }
    //X         static Vc_INTRINSIC __m128i Vc_CONST _mm_cmpgt_epu8 (__m128i a, __m128i b) { return _mm_cmpgt_epi8 (
    //X                 _mm_xor_si128(a, _mm_setmin_epi8 ()), _mm_xor_si128(b, _mm_setmin_epi8 ())); }
    static Vc_INTRINSIC __m128i Vc_CONST _mm_cmplt_epu16(__m128i a, __m128i b) { return _mm_cmplt_epi16(
            _mm_xor_si128(a, _mm_setmin_epi16()), _mm_xor_si128(b, _mm_setmin_epi16())); }
    static Vc_INTRINSIC __m128i Vc_CONST _mm_cmpgt_epu16(__m128i a, __m128i b) { return _mm_cmpgt_epi16(
            _mm_xor_si128(a, _mm_setmin_epi16()), _mm_xor_si128(b, _mm_setmin_epi16())); }
    static Vc_INTRINSIC __m128i Vc_CONST _mm_cmplt_epu32(__m128i a, __m128i b) { return _mm_cmplt_epi32(
            _mm_xor_si128(a, _mm_setmin_epi32()), _mm_xor_si128(b, _mm_setmin_epi32())); }
    static Vc_INTRINSIC __m128i Vc_CONST _mm_cmpgt_epu32(__m128i a, __m128i b) { return _mm_cmpgt_epi32(
            _mm_xor_si128(a, _mm_setmin_epi32()), _mm_xor_si128(b, _mm_setmin_epi32())); }
} // namespace SSE
} // namespace Vc
} // namespace ROOT

// SSE3
#ifdef VC_IMPL_SSE3
extern "C" {
#include <pmmintrin.h>
}
#endif
// SSSE3
#ifdef VC_IMPL_SSSE3
extern "C" {
#include <tmmintrin.h>
}
#define mm_abs_epi8  _mm_abs_epi8
#define mm_abs_epi16 _mm_abs_epi16
#define mm_abs_epi32 _mm_abs_epi32
#define mm_alignr_epi8 _mm_alignr_epi8
namespace ROOT {
namespace Vc
{
namespace SSE
{

    // not overriding _mm_set1_epi8 because this one should only be used for non-constants
    static Vc_INTRINSIC __m128i Vc_CONST set1_epi8(int a) {
#if defined(VC_GCC) && VC_GCC < 0x40500
        return _mm_shuffle_epi8(_mm_cvtsi32_si128(a), _mm_setzero_si128());
#else
        // GCC 4.5 nows about the pshufb improvement
        return _mm_set1_epi8(a);
#endif
    }

} // namespace SSE
} // namespace Vc
} // namespace ROOT
#else
namespace ROOT {
namespace Vc
{
namespace SSE
{
    static Vc_INTRINSIC __m128i Vc_CONST mm_abs_epi8 (__m128i a) {
        __m128i negative = _mm_cmplt_epi8 (a, _mm_setzero_si128());
        return _mm_add_epi8 (_mm_xor_si128(a, negative), _mm_and_si128(negative,  _mm_setone_epi8()));
    }
    // positive value:
    //   negative == 0
    //   a unchanged after xor
    //   0 >> 31 -> 0
    //   a + 0 -> a
    // negative value:
    //   negative == -1
    //   a xor -1 -> -a - 1
    //   -1 >> 31 -> 1
    //   -a - 1 + 1 -> -a
    static Vc_INTRINSIC __m128i Vc_CONST mm_abs_epi16(__m128i a) {
        __m128i negative = _mm_cmplt_epi16(a, _mm_setzero_si128());
        return _mm_add_epi16(_mm_xor_si128(a, negative), _mm_srli_epi16(negative, 15));
    }
    static Vc_INTRINSIC __m128i Vc_CONST mm_abs_epi32(__m128i a) {
        __m128i negative = _mm_cmplt_epi32(a, _mm_setzero_si128());
        return _mm_add_epi32(_mm_xor_si128(a, negative), _mm_srli_epi32(negative, 31));
    }
    static Vc_INTRINSIC __m128i Vc_CONST set1_epi8(int a) {
        return _mm_set1_epi8(a);
    }
    static Vc_INTRINSIC __m128i Vc_CONST mm_alignr_epi8(__m128i a, __m128i b, const int s) {
        switch (s) {
            case  0: return b;
            case  1: return _mm_or_si128(_mm_slli_si128(a, 15), _mm_srli_si128(b,  1));
            case  2: return _mm_or_si128(_mm_slli_si128(a, 14), _mm_srli_si128(b,  2));
            case  3: return _mm_or_si128(_mm_slli_si128(a, 13), _mm_srli_si128(b,  3));
            case  4: return _mm_or_si128(_mm_slli_si128(a, 12), _mm_srli_si128(b,  4));
            case  5: return _mm_or_si128(_mm_slli_si128(a, 11), _mm_srli_si128(b,  5));
            case  6: return _mm_or_si128(_mm_slli_si128(a, 10), _mm_srli_si128(b,  6));
            case  7: return _mm_or_si128(_mm_slli_si128(a,  9), _mm_srli_si128(b,  7));
            case  8: return _mm_or_si128(_mm_slli_si128(a,  8), _mm_srli_si128(b,  8));
            case  9: return _mm_or_si128(_mm_slli_si128(a,  7), _mm_srli_si128(b,  9));
            case 10: return _mm_or_si128(_mm_slli_si128(a,  6), _mm_srli_si128(b, 10));
            case 11: return _mm_or_si128(_mm_slli_si128(a,  5), _mm_srli_si128(b, 11));
            case 12: return _mm_or_si128(_mm_slli_si128(a,  4), _mm_srli_si128(b, 12));
            case 13: return _mm_or_si128(_mm_slli_si128(a,  3), _mm_srli_si128(b, 13));
            case 14: return _mm_or_si128(_mm_slli_si128(a,  2), _mm_srli_si128(b, 14));
            case 15: return _mm_or_si128(_mm_slli_si128(a,  1), _mm_srli_si128(b, 15));
            case 16: return a;
            case 17: return _mm_srli_si128(a,  1);
            case 18: return _mm_srli_si128(a,  2);
            case 19: return _mm_srli_si128(a,  3);
            case 20: return _mm_srli_si128(a,  4);
            case 21: return _mm_srli_si128(a,  5);
            case 22: return _mm_srli_si128(a,  6);
            case 23: return _mm_srli_si128(a,  7);
            case 24: return _mm_srli_si128(a,  8);
            case 25: return _mm_srli_si128(a,  9);
            case 26: return _mm_srli_si128(a, 10);
            case 27: return _mm_srli_si128(a, 11);
            case 28: return _mm_srli_si128(a, 12);
            case 29: return _mm_srli_si128(a, 13);
            case 30: return _mm_srli_si128(a, 14);
            case 31: return _mm_srli_si128(a, 15);
        }
        return _mm_setzero_si128();
    }

} // namespace SSE
} // namespace Vc
} // namespace ROOT

#endif

// SSE4.1
#ifdef VC_IMPL_SSE4_1
extern "C" {
#include <smmintrin.h>
}
namespace ROOT {
namespace Vc
{
namespace SSE
{
#define mm_blendv_pd _mm_blendv_pd
#define mm_blendv_ps _mm_blendv_ps
#define mm_blendv_epi8 _mm_blendv_epi8
#define mm_blend_epi16 _mm_blend_epi16
#define mm_blend_ps _mm_blend_ps
#define mm_blend_pd _mm_blend_pd

#define mm_min_epi32 _mm_min_epi32
#define mm_max_epi32 _mm_max_epi32
#define mm_min_epu32 _mm_min_epu32
#define mm_max_epu32 _mm_max_epu32
//#define mm_min_epi16 _mm_min_epi16
//#define mm_max_epi16 _mm_max_epi16
#define mm_min_epu16 _mm_min_epu16
#define mm_max_epu16 _mm_max_epu16
#define mm_min_epi8  _mm_min_epi8
#define mm_max_epi8  _mm_max_epi8

#define mm_cvtepu16_epi32 _mm_cvtepu16_epi32
#define mm_cvtepu8_epi16 _mm_cvtepu8_epi16
#define mm_cvtepi8_epi16 _mm_cvtepi8_epi16
#define mm_cvtepu16_epi32 _mm_cvtepu16_epi32
#define mm_cvtepi16_epi32 _mm_cvtepi16_epi32
#define mm_cvtepu8_epi32 _mm_cvtepu8_epi32
#define mm_cvtepi8_epi32 _mm_cvtepi8_epi32
#define mm_stream_load_si128 _mm_stream_load_si128
// TODO
} // namespace SSE
} // namespace Vc
} // namespace ROOT
#else
namespace ROOT {
namespace Vc
{
namespace SSE
{
    static Vc_INTRINSIC __m128d mm_blendv_pd(__m128d a, __m128d b, __m128d c) {
        return _mm_or_pd(_mm_andnot_pd(c, a), _mm_and_pd(c, b));
    }
    static Vc_INTRINSIC __m128  mm_blendv_ps(__m128  a, __m128  b, __m128  c) {
        return _mm_or_ps(_mm_andnot_ps(c, a), _mm_and_ps(c, b));
    }
    static Vc_INTRINSIC __m128i mm_blendv_epi8(__m128i a, __m128i b, __m128i c) {
        return _mm_or_si128(_mm_andnot_si128(c, a), _mm_and_si128(c, b));
    }

    // only use the following blend functions with immediates as mask and, of course, compiling
    // with optimization
    static Vc_INTRINSIC __m128d mm_blend_pd(__m128d a, __m128d b, const int mask) {
        switch (mask) {
        case 0x0:
            return a;
        case 0x1:
            return _mm_shuffle_pd(b, a, 2);
        case 0x2:
            return _mm_shuffle_pd(a, b, 2);
        case 0x3:
            return b;
        default:
            abort();
            return a; // should never be reached, but MSVC needs it else it warns about 'not all control paths return a value'
        }
    }
    static Vc_INTRINSIC __m128  mm_blend_ps(__m128  a, __m128  b, const int mask) {
        __m128i c;
        switch (mask) {
        case 0x0:
            return a;
        case 0x1:
            c = _mm_srli_si128(_mm_setallone_si128(), 12);
            break;
        case 0x2:
            c = _mm_slli_si128(_mm_srli_si128(_mm_setallone_si128(), 12), 4);
            break;
        case 0x3:
            c = _mm_srli_si128(_mm_setallone_si128(), 8);
            break;
        case 0x4:
            c = _mm_slli_si128(_mm_srli_si128(_mm_setallone_si128(), 12), 8);
            break;
        case 0x5:
            c = _mm_set_epi32(0, -1, 0, -1);
            break;
        case 0x6:
            c = _mm_slli_si128(_mm_srli_si128(_mm_setallone_si128(), 8), 4);
            break;
        case 0x7:
            c = _mm_srli_si128(_mm_setallone_si128(), 4);
            break;
        case 0x8:
            c = _mm_slli_si128(_mm_setallone_si128(), 12);
            break;
        case 0x9:
            c = _mm_set_epi32(-1, 0, 0, -1);
            break;
        case 0xa:
            c = _mm_set_epi32(-1, 0, -1, 0);
            break;
        case 0xb:
            c = _mm_set_epi32(-1, 0, -1, -1);
            break;
        case 0xc:
            c = _mm_slli_si128(_mm_setallone_si128(), 8);
            break;
        case 0xd:
            c = _mm_set_epi32(-1, -1, 0, -1);
            break;
        case 0xe:
            c = _mm_slli_si128(_mm_setallone_si128(), 4);
            break;
        case 0xf:
            return b;
        default: // may not happen
            abort();
            c = _mm_setzero_si128();
            break;
        }
        __m128 _c = _mm_castsi128_ps(c);
        return _mm_or_ps(_mm_andnot_ps(_c, a), _mm_and_ps(_c, b));
    }
    static Vc_INTRINSIC __m128i mm_blend_epi16(__m128i a, __m128i b, const int mask) {
        __m128i c;
        switch (mask) {
        case 0x00:
            return a;
        case 0x01:
            c = _mm_srli_si128(_mm_setallone_si128(), 14);
            break;
        case 0x03:
            c = _mm_srli_si128(_mm_setallone_si128(), 12);
            break;
        case 0x07:
            c = _mm_srli_si128(_mm_setallone_si128(), 10);
            break;
        case 0x0f:
            return _mm_unpackhi_epi64(_mm_slli_si128(b, 8), a);
        case 0x1f:
            c = _mm_srli_si128(_mm_setallone_si128(), 6);
            break;
        case 0x3f:
            c = _mm_srli_si128(_mm_setallone_si128(), 4);
            break;
        case 0x7f:
            c = _mm_srli_si128(_mm_setallone_si128(), 2);
            break;
        case 0x80:
            c = _mm_slli_si128(_mm_setallone_si128(), 14);
            break;
        case 0xc0:
            c = _mm_slli_si128(_mm_setallone_si128(), 12);
            break;
        case 0xe0:
            c = _mm_slli_si128(_mm_setallone_si128(), 10);
            break;
        case 0xf0:
            c = _mm_slli_si128(_mm_setallone_si128(), 8);
            break;
        case 0xf8:
            c = _mm_slli_si128(_mm_setallone_si128(), 6);
            break;
        case 0xfc:
            c = _mm_slli_si128(_mm_setallone_si128(), 4);
            break;
        case 0xfe:
            c = _mm_slli_si128(_mm_setallone_si128(), 2);
            break;
        case 0xff:
            return b;
        case 0xcc:
            return _mm_unpacklo_epi32(_mm_shuffle_epi32(a, _MM_SHUFFLE(2, 0, 2, 0)), _mm_shuffle_epi32(b, _MM_SHUFFLE(3, 1, 3, 1)));
        case 0x33:
            return _mm_unpacklo_epi32(_mm_shuffle_epi32(b, _MM_SHUFFLE(2, 0, 2, 0)), _mm_shuffle_epi32(a, _MM_SHUFFLE(3, 1, 3, 1)));
        default:
            const __m128i shift = _mm_set_epi16(0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x2000, 0x4000, -0x7fff);
            c = _mm_srai_epi16(_mm_mullo_epi16(_mm_set1_epi16(mask), shift), 15);
            break;
        }
        return _mm_or_si128(_mm_andnot_si128(c, a), _mm_and_si128(c, b));
    }

    static Vc_INTRINSIC __m128i Vc_CONST mm_max_epi8 (__m128i a, __m128i b) {
        return mm_blendv_epi8(b, a, _mm_cmpgt_epi8 (a, b));
    }
    static Vc_INTRINSIC __m128i Vc_CONST mm_max_epi32(__m128i a, __m128i b) {
        return mm_blendv_epi8(b, a, _mm_cmpgt_epi32(a, b));
    }
//X         static Vc_INTRINSIC __m128i Vc_CONST mm_max_epu8 (__m128i a, __m128i b) {
//X             return mm_blendv_epi8(b, a, _mm_cmpgt_epu8 (a, b));
//X         }
    static Vc_INTRINSIC __m128i Vc_CONST mm_max_epu16(__m128i a, __m128i b) {
        return mm_blendv_epi8(b, a, _mm_cmpgt_epu16(a, b));
    }
    static Vc_INTRINSIC __m128i Vc_CONST mm_max_epu32(__m128i a, __m128i b) {
        return mm_blendv_epi8(b, a, _mm_cmpgt_epu32(a, b));
    }
//X         static Vc_INTRINSIC __m128i Vc_CONST mm_min_epu8 (__m128i a, __m128i b) {
//X             return mm_blendv_epi8(a, b, _mm_cmpgt_epu8 (a, b));
//X         }
    static Vc_INTRINSIC __m128i Vc_CONST mm_min_epu16(__m128i a, __m128i b) {
        return mm_blendv_epi8(a, b, _mm_cmpgt_epu16(a, b));
    }
    static Vc_INTRINSIC __m128i Vc_CONST mm_min_epu32(__m128i a, __m128i b) {
        return mm_blendv_epi8(a, b, _mm_cmpgt_epu32(a, b));
    }
    static Vc_INTRINSIC __m128i Vc_CONST mm_min_epi8 (__m128i a, __m128i b) {
        return mm_blendv_epi8(a, b, _mm_cmpgt_epi8 (a, b));
    }
    static Vc_INTRINSIC __m128i Vc_CONST mm_min_epi32(__m128i a, __m128i b) {
        return mm_blendv_epi8(a, b, _mm_cmpgt_epi32(a, b));
    }
    static Vc_INTRINSIC Vc_CONST __m128i mm_cvtepu8_epi16(__m128i epu8) {
        return _mm_unpacklo_epi8(epu8, _mm_setzero_si128());
    }
    static Vc_INTRINSIC Vc_CONST __m128i mm_cvtepi8_epi16(__m128i epi8) {
        return _mm_unpacklo_epi8(epi8, _mm_cmplt_epi8(epi8, _mm_setzero_si128()));
    }
    static Vc_INTRINSIC Vc_CONST __m128i mm_cvtepu16_epi32(__m128i epu16) {
        return _mm_unpacklo_epi16(epu16, _mm_setzero_si128());
    }
    static Vc_INTRINSIC Vc_CONST __m128i mm_cvtepi16_epi32(__m128i epu16) {
        return _mm_unpacklo_epi16(epu16, _mm_cmplt_epi16(epu16, _mm_setzero_si128()));
    }
    static Vc_INTRINSIC Vc_CONST __m128i mm_cvtepu8_epi32(__m128i epu8) {
        return mm_cvtepu16_epi32(mm_cvtepu8_epi16(epu8));
    }
    static Vc_INTRINSIC Vc_CONST __m128i mm_cvtepi8_epi32(__m128i epi8) {
        const __m128i neg = _mm_cmplt_epi8(epi8, _mm_setzero_si128());
        const __m128i epi16 = _mm_unpacklo_epi8(epi8, neg);
        return _mm_unpacklo_epi16(epi16, _mm_unpacklo_epi8(neg, neg));
    }
    static Vc_INTRINSIC Vc_PURE __m128i mm_stream_load_si128(__m128i *mem) {
        return _mm_load_si128(mem);
    }

} // namespace SSE
} // namespace Vc
} // namespace ROOT
#endif

#ifdef VC_IMPL_POPCNT
#include <popcntintrin.h>
#endif

// SSE4.2
#ifdef VC_IMPL_SSE4_2
extern "C" {
#include <nmmintrin.h>
}
#endif

namespace ROOT {
namespace Vc
{
namespace SSE
{
    static Vc_INTRINSIC Vc_CONST float extract_float_imm(const __m128 v, const size_t i) {
        float f = 0.;
        switch (i) {
        case 0:
            f = _mm_cvtss_f32(v);
            break;
#if defined VC_IMPL_SSE4_1 && !defined VC_MSVC
        default:
#ifdef VC_GCC
            f = __builtin_ia32_vec_ext_v4sf(static_cast<__v4sf>(v), (i));
#else
            // MSVC fails to compile this because it can't optimize i to an immediate
            _MM_EXTRACT_FLOAT(f, v, i);
#endif
            break;
#else
        case 1:
            f = _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v), 4)));
            break;
        case 2:
            f = _mm_cvtss_f32(_mm_movehl_ps(v, v));
            break;
        case 3:
            f = _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v), 12)));
            break;
#endif
        }
        return f;
    }
    static Vc_INTRINSIC Vc_CONST double extract_double_imm(const __m128d v, const size_t i) {
        if (i == 0) {
            return _mm_cvtsd_f64(v);
        }
        return _mm_cvtsd_f64(_mm_castps_pd(_mm_movehl_ps(_mm_castpd_ps(v), _mm_castpd_ps(v))));
    }
    static Vc_INTRINSIC Vc_CONST float extract_float(const __m128 v, const size_t i) {
#ifdef VC_GCC
        if (__builtin_constant_p(i)) {
            return extract_float_imm(v, i);
//X         if (index <= 1) {
//X             unsigned long long tmp = _mm_cvtsi128_si64(_mm_castps_si128(v));
//X             if (index == 0) tmp &= 0xFFFFFFFFull;
//X             if (index == 1) tmp >>= 32;
//X             return Common::AliasingEntryHelper<EntryType>(tmp);
//X         }
        } else {
            typedef float float4[4] Vc_MAY_ALIAS;
            const float4 &data = reinterpret_cast<const float4 &>(v);
            return data[i];
        }
#else
        union { __m128 v; float m[4]; } u;
        u.v = v;
        return u.m[i];
#endif
    }

    static Vc_INTRINSIC Vc_PURE __m128  _mm_stream_load(const float *mem) {
#ifdef VC_IMPL_SSE4_1
        return _mm_castsi128_ps(_mm_stream_load_si128(reinterpret_cast<__m128i *>(const_cast<float *>(mem))));
#else
        return _mm_load_ps(mem);
#endif
    }
    static Vc_INTRINSIC Vc_PURE __m128d _mm_stream_load(const double *mem) {
#ifdef VC_IMPL_SSE4_1
        return _mm_castsi128_pd(_mm_stream_load_si128(reinterpret_cast<__m128i *>(const_cast<double *>(mem))));
#else
        return _mm_load_pd(mem);
#endif
    }
    static Vc_INTRINSIC Vc_PURE __m128i _mm_stream_load(const int *mem) {
#ifdef VC_IMPL_SSE4_1
        return _mm_stream_load_si128(reinterpret_cast<__m128i *>(const_cast<int *>(mem)));
#else
        return _mm_load_si128(reinterpret_cast<const __m128i *>(mem));
#endif
    }
    static Vc_INTRINSIC Vc_PURE __m128i _mm_stream_load(const unsigned int *mem) {
        return _mm_stream_load(reinterpret_cast<const int *>(mem));
    }
    static Vc_INTRINSIC Vc_PURE __m128i _mm_stream_load(const short *mem) {
        return _mm_stream_load(reinterpret_cast<const int *>(mem));
    }
    static Vc_INTRINSIC Vc_PURE __m128i _mm_stream_load(const unsigned short *mem) {
        return _mm_stream_load(reinterpret_cast<const int *>(mem));
    }
    static Vc_INTRINSIC Vc_PURE __m128i _mm_stream_load(const signed char *mem) {
        return _mm_stream_load(reinterpret_cast<const int *>(mem));
    }
    static Vc_INTRINSIC Vc_PURE __m128i _mm_stream_load(const unsigned char *mem) {
        return _mm_stream_load(reinterpret_cast<const int *>(mem));
    }
} // namespace SSE
} // namespace Vc
} // namespace ROOT

// XOP / FMA4
#if defined(VC_IMPL_XOP) || defined(VC_IMPL_FMA4)
extern "C" {
#include <x86intrin.h>
}
#endif

#include "undomacros.h"
#include "shuffle.h"

#endif // SSE_INTRINSICS_H
