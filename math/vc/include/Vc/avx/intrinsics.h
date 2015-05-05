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

#ifndef VC_AVX_INTRINSICS_H
#define VC_AVX_INTRINSICS_H

#include "../common/windows_fix_intrin.h"

#include <Vc/global.h>

// see comment in sse/intrinsics.h
extern "C" {
// AVX
#include <immintrin.h>

#if (defined(VC_IMPL_XOP) || defined(VC_IMPL_FMA4)) && !defined(VC_MSVC)
#include <x86intrin.h>
#endif
}

#include "../common/fix_clang_emmintrin.h"

#if defined(VC_CLANG) && VC_CLANG < 0x30100
// _mm_permute_ps is broken: http://llvm.org/bugs/show_bug.cgi?id=12401
#undef _mm_permute_ps
#define _mm_permute_ps(A, C) __extension__ ({ \
  m128 __A = (A); \
  (m128)__builtin_shufflevector((__v4sf)__A, (__v4sf) _mm_setzero_ps(), \
                                   (C) & 0x3, ((C) & 0xc) >> 2, \
                                   ((C) & 0x30) >> 4, ((C) & 0xc0) >> 6); })
#endif

#include "const_data.h"
#include "macros.h"
#include <cstdlib>

#if defined(VC_CLANG) || defined(VC_MSVC) || (defined(VC_GCC) && !defined(__OPTIMIZE__))
#define VC_REQUIRES_MACRO_FOR_IMMEDIATE_ARGUMENT
#endif

#if defined(VC_CLANG) && VC_CLANG <= 0x30000
// _mm_alignr_epi8 doesn't specify its return type, thus breaking overload resolution
#undef _mm_alignr_epi8
#define _mm_alignr_epi8(a, b, n) ((m128i)__builtin_ia32_palignr128((a), (b), (n)))
#endif

namespace ROOT {
namespace Vc
{
namespace AVX
{
    /* super evil hacking around C++ features:
     * consider
     * void fun(int);
     * namespace X { void fun(int); }
     * namespace X { void bar() { fun(0); } } // this will be a call to X::fun(int)
     *
     * void fun(m256);
     * namespace X { void fun(m256); }
     * namespace X { void bar() { fun(0); } } // this will be ambiguous because m256 is a
     *                                           non-fundamental type in the global namespace, thus
     *                                           adding ::fun(m256) to the candidates
     *
     * To make my own overloads of the intrinsics distinct I have to use a type that is inside the
     * Vc::AVX namespace. To reduce porting effort and increase generality I want to use the same
     * function names as used in the global namespace. The type name may not be the same, though
     * because identifiers starting with two underscores are reserved by the standard. Thus using
     * those would mean to depend on undefined behavior.
     * Sadly a typedef is not enough.
     * Public inheritance also does not work, because at least ICC considers the __m??? types to be
     * some sort of fundamental types.
     * Thus composition is the only solution.
     */
#ifdef VC_UNCONDITIONAL_AVX2_INTRINSICS
    template<typename T> struct Alias
    {
        typedef T Base;
        T _d;
        Vc_ALWAYS_INLINE operator T &() { return _d; }
        Vc_ALWAYS_INLINE operator const T &() const { return _d; }
        Vc_ALWAYS_INLINE Alias() {}
        Vc_ALWAYS_INLINE Alias(T x) : _d(x) {}
        Vc_ALWAYS_INLINE Alias(const Alias &x) : _d(x._d) {}
        Vc_ALWAYS_INLINE Alias &operator=(T x) { _d = x; return *this; }
        Vc_ALWAYS_INLINE Alias &operator=(const Alias &x) { _d = x._d; return *this; }
    };
    typedef Alias<__m128 > m128 ;
    typedef Alias<__m128d> m128d;
    typedef Alias<__m128i> m128i;
    typedef Alias<__m256 > m256 ;
    typedef Alias<__m256d> m256d;
    typedef Alias<__m256i> m256i;
#else
    typedef __m128  m128 ;
    typedef __m128d m128d;
    typedef __m128i m128i;
    typedef __m256  m256 ;
    typedef __m256d m256d;
    typedef __m256i m256i;
#endif
#if defined(VC_UNCONDITIONAL_AVX2_INTRINSICS) && defined(VC_PASSING_VECTOR_BY_VALUE_IS_BROKEN)
    typedef const m128  & param128 ;
    typedef const m128d & param128d;
    typedef const m128i & param128i;
    typedef const m256  & param256 ;
    typedef const m256d & param256d;
    typedef const m256i & param256i;
#else
    typedef const m128  param128 ;
    typedef const m128d param128d;
    typedef const m128i param128i;
    typedef const m256  param256 ;
    typedef const m256d param256d;
    typedef const m256i param256i;
#endif

#ifdef VC_UNCONDITIONAL_AVX2_INTRINSICS
    // Make use of cast intrinsics easier. But if param256 == const __m256 then these would lead to
    // ambiguities.
    static Vc_INTRINSIC m256i Vc_CONST _mm256_castps_si256(param256  a) { return ::_mm256_castps_si256(a); }
    static Vc_INTRINSIC m256d Vc_CONST _mm256_castps_pd   (param256  a) { return ::_mm256_castps_pd   (a); }
    static Vc_INTRINSIC m256i Vc_CONST _mm256_castpd_si256(param256d a) { return ::_mm256_castpd_si256(a); }
    static Vc_INTRINSIC m256  Vc_CONST _mm256_castpd_ps   (param256d a) { return ::_mm256_castpd_ps   (a); }
    static Vc_INTRINSIC m256  Vc_CONST _mm256_castsi256_ps(param256i a) { return ::_mm256_castsi256_ps(a); }
    static Vc_INTRINSIC m256d Vc_CONST _mm256_castsi256_pd(param256i a) { return ::_mm256_castsi256_pd(a); }
#endif

#ifdef VC_GCC
    // Redefine the mul/add/sub intrinsics to use GCC-specific operators instead of builtin
    // functions. This way the fp-contraction optimization step kicks in and creates FMAs! :)
    static Vc_INTRINSIC Vc_CONST m256d _mm256_mul_pd(m256d a, m256d b) { return static_cast<m256d>(static_cast<__v4df>(a) * static_cast<__v4df>(b)); }
    static Vc_INTRINSIC Vc_CONST m256d _mm256_add_pd(m256d a, m256d b) { return static_cast<m256d>(static_cast<__v4df>(a) + static_cast<__v4df>(b)); }
    static Vc_INTRINSIC Vc_CONST m256d _mm256_sub_pd(m256d a, m256d b) { return static_cast<m256d>(static_cast<__v4df>(a) - static_cast<__v4df>(b)); }
    static Vc_INTRINSIC Vc_CONST m256 _mm256_mul_ps(m256 a, m256 b) { return static_cast<m256>(static_cast<__v8sf>(a) * static_cast<__v8sf>(b)); }
    static Vc_INTRINSIC Vc_CONST m256 _mm256_add_ps(m256 a, m256 b) { return static_cast<m256>(static_cast<__v8sf>(a) + static_cast<__v8sf>(b)); }
    static Vc_INTRINSIC Vc_CONST m256 _mm256_sub_ps(m256 a, m256 b) { return static_cast<m256>(static_cast<__v8sf>(a) - static_cast<__v8sf>(b)); }
#endif

    static Vc_INTRINSIC m256  Vc_CONST _mm256_set1_ps   (float  a) { return ::_mm256_set1_ps   (a); }
    static Vc_INTRINSIC m256d Vc_CONST _mm256_set1_pd   (double a) { return ::_mm256_set1_pd   (a); }
    static Vc_INTRINSIC m256i Vc_CONST _mm256_set1_epi32(int    a) { return ::_mm256_set1_epi32(a); }
    //static Vc_INTRINSIC m256i Vc_CONST _mm256_set1_epu32(unsigned int a) { return ::_mm256_set1_epu32(a); }

#if defined(VC_GNU_ASM) && !defined(NVALGRIND)
    static Vc_INTRINSIC m128i Vc_CONST _mm_setallone() { m128i r; __asm__("pcmpeqb %0,%0":"=x"(r)); return r; }
#else
    static Vc_INTRINSIC m128i Vc_CONST _mm_setallone() { m128i r = _mm_setzero_si128(); return _mm_cmpeq_epi8(r, r); }
#endif
    static Vc_INTRINSIC m128i Vc_CONST _mm_setallone_si128() { return _mm_setallone(); }
    static Vc_INTRINSIC m128d Vc_CONST _mm_setallone_pd() { return _mm_castsi128_pd(_mm_setallone()); }
    static Vc_INTRINSIC m128  Vc_CONST _mm_setallone_ps() { return _mm_castsi128_ps(_mm_setallone()); }

    static Vc_INTRINSIC m128i Vc_CONST _mm_setone_epi8 ()  { return _mm_set1_epi8(1); }
    static Vc_INTRINSIC m128i Vc_CONST _mm_setone_epu8 ()  { return _mm_setone_epi8(); }
    static Vc_INTRINSIC m128i Vc_CONST _mm_setone_epi16()  { return _mm_castps_si128(_mm_broadcast_ss(reinterpret_cast<const float *>(c_general::one16))); }
    static Vc_INTRINSIC m128i Vc_CONST _mm_setone_epu16()  { return _mm_setone_epi16(); }
    static Vc_INTRINSIC m128i Vc_CONST _mm_setone_epi32()  { return _mm_castps_si128(_mm_broadcast_ss(reinterpret_cast<const float *>(&_IndexesFromZero32[1]))); }

#if defined(VC_GNU_ASM) && !defined(NVALGRIND)
    static Vc_INTRINSIC m256 Vc_CONST _mm256_setallone() { __m256 r; __asm__("vcmpps $8,%0,%0,%0":"=x"(r)); return r; }
#elif defined(VC_MSVC)
    // MSVC puts temporaries of this value on the stack, but sometimes at misaligned addresses, try
    // some other generator instead...
    static Vc_INTRINSIC m256 Vc_CONST _mm256_setallone() { return _mm256_castsi256_ps(_mm256_set1_epi32(-1)); }
#else
    static Vc_INTRINSIC m256 Vc_CONST _mm256_setallone() { m256 r = _mm256_setzero_ps(); return _mm256_cmp_ps(r, r, _CMP_EQ_UQ); }
#endif
    static Vc_INTRINSIC m256i Vc_CONST _mm256_setallone_si256() { return _mm256_castps_si256(_mm256_setallone()); }
    static Vc_INTRINSIC m256d Vc_CONST _mm256_setallone_pd() { return _mm256_castps_pd(_mm256_setallone()); }
    static Vc_INTRINSIC m256  Vc_CONST _mm256_setallone_ps() { return _mm256_setallone(); }

    static Vc_INTRINSIC m256i Vc_CONST _mm256_setone_epi8 ()  { return _mm256_set1_epi8(1); }
    static Vc_INTRINSIC m256i Vc_CONST _mm256_setone_epu8 ()  { return _mm256_setone_epi8(); }
    static Vc_INTRINSIC m256i Vc_CONST _mm256_setone_epi16()  { return _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float *>(c_general::one16))); }
    static Vc_INTRINSIC m256i Vc_CONST _mm256_setone_epu16()  { return _mm256_setone_epi16(); }
    static Vc_INTRINSIC m256i Vc_CONST _mm256_setone_epi32()  { return _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float *>(&_IndexesFromZero32[1]))); }
    static Vc_INTRINSIC m256i Vc_CONST _mm256_setone_epu32()  { return _mm256_setone_epi32(); }

    static Vc_INTRINSIC m256  Vc_CONST _mm256_setone_ps()     { return _mm256_broadcast_ss(&c_general::oneFloat); }
    static Vc_INTRINSIC m256d Vc_CONST _mm256_setone_pd()     { return _mm256_broadcast_sd(&c_general::oneDouble); }

    static Vc_INTRINSIC m256d Vc_CONST _mm256_setabsmask_pd() { return _mm256_broadcast_sd(reinterpret_cast<const double *>(&c_general::absMaskFloat[0])); }
    static Vc_INTRINSIC m256  Vc_CONST _mm256_setabsmask_ps() { return _mm256_broadcast_ss(reinterpret_cast<const float *>(&c_general::absMaskFloat[1])); }
    static Vc_INTRINSIC m256d Vc_CONST _mm256_setsignmask_pd(){ return _mm256_broadcast_sd(reinterpret_cast<const double *>(&c_general::signMaskFloat[0])); }
    static Vc_INTRINSIC m256  Vc_CONST _mm256_setsignmask_ps(){ return _mm256_broadcast_ss(reinterpret_cast<const float *>(&c_general::signMaskFloat[1])); }

    static Vc_INTRINSIC m256  Vc_CONST _mm256_set2power31_ps()    { return _mm256_broadcast_ss(&c_general::_2power31); }
    static Vc_INTRINSIC m256i Vc_CONST _mm256_set2power31_epu32() { return _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float *>(&c_general::signMaskFloat[1]))); }

    //X         static Vc_INTRINSIC m256i Vc_CONST _mm256_setmin_epi8 () { return _mm256_slli_epi8 (_mm256_setallone_si256(),  7); }
    static Vc_INTRINSIC m128i Vc_CONST _mm_setmin_epi16() { return _mm_castps_si128(_mm_broadcast_ss(reinterpret_cast<const float *>(c_general::minShort))); }
    static Vc_INTRINSIC m128i Vc_CONST _mm_setmin_epi32() { return _mm_castps_si128(_mm_broadcast_ss(reinterpret_cast<const float *>(&c_general::signMaskFloat[1]))); }
    static Vc_INTRINSIC m256i Vc_CONST _mm256_setmin_epi16() { return _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float *>(c_general::minShort))); }
    static Vc_INTRINSIC m256i Vc_CONST _mm256_setmin_epi32() { return _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float *>(&c_general::signMaskFloat[1]))); }

#ifdef VC_REQUIRES_MACRO_FOR_IMMEDIATE_ARGUMENT
#define _mm_extract_epu8 (x, i) (static_cast<unsigned char> (_mm_extract_epi8 ((x), (i))))
#define _mm_extract_epu16(x, i) (static_cast<unsigned short>(_mm_extract_epi16((x), (i))))
#define _mm_extract_epu32(x, i) (static_cast<unsigned int>  (_mm_extract_epi32((x), (i))))
#else
    static Vc_INTRINSIC unsigned char Vc_CONST _mm_extract_epu8(param128i x, const int i) { return _mm_extract_epi8(x, i); }
    static Vc_INTRINSIC unsigned short Vc_CONST _mm_extract_epu16(param128i x, const int i) { return _mm_extract_epi16(x, i); }
    static Vc_INTRINSIC unsigned int Vc_CONST _mm_extract_epu32(param128i x, const int i) { return _mm_extract_epi32(x, i); }
#endif

    /////////////////////// COMPARE OPS ///////////////////////
    static Vc_INTRINSIC m256d Vc_CONST _mm256_cmpeq_pd   (param256d a, param256d b) { return _mm256_cmp_pd(a, b, _CMP_EQ_OQ); }
    static Vc_INTRINSIC m256d Vc_CONST _mm256_cmpneq_pd  (param256d a, param256d b) { return _mm256_cmp_pd(a, b, _CMP_NEQ_UQ); }
    static Vc_INTRINSIC m256d Vc_CONST _mm256_cmplt_pd   (param256d a, param256d b) { return _mm256_cmp_pd(a, b, _CMP_LT_OS); }
    static Vc_INTRINSIC m256d Vc_CONST _mm256_cmpnlt_pd  (param256d a, param256d b) { return _mm256_cmp_pd(a, b, _CMP_NLT_US); }
    static Vc_INTRINSIC m256d Vc_CONST _mm256_cmple_pd   (param256d a, param256d b) { return _mm256_cmp_pd(a, b, _CMP_LE_OS); }
    static Vc_INTRINSIC m256d Vc_CONST _mm256_cmpnle_pd  (param256d a, param256d b) { return _mm256_cmp_pd(a, b, _CMP_NLE_US); }
    static Vc_INTRINSIC m256d Vc_CONST _mm256_cmpord_pd  (param256d a, param256d b) { return _mm256_cmp_pd(a, b, _CMP_ORD_Q); }
    static Vc_INTRINSIC m256d Vc_CONST _mm256_cmpunord_pd(param256d a, param256d b) { return _mm256_cmp_pd(a, b, _CMP_UNORD_Q); }

    static Vc_INTRINSIC m256  Vc_CONST _mm256_cmpeq_ps   (param256  a, param256  b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OQ); }
    static Vc_INTRINSIC m256  Vc_CONST _mm256_cmpneq_ps  (param256  a, param256  b) { return _mm256_cmp_ps(a, b, _CMP_NEQ_UQ); }
    static Vc_INTRINSIC m256  Vc_CONST _mm256_cmplt_ps   (param256  a, param256  b) { return _mm256_cmp_ps(a, b, _CMP_LT_OS); }
    static Vc_INTRINSIC m256  Vc_CONST _mm256_cmpnlt_ps  (param256  a, param256  b) { return _mm256_cmp_ps(a, b, _CMP_NLT_US); }
    static Vc_INTRINSIC m256  Vc_CONST _mm256_cmpge_ps   (param256  a, param256  b) { return _mm256_cmp_ps(a, b, _CMP_NLT_US); }
    static Vc_INTRINSIC m256  Vc_CONST _mm256_cmple_ps   (param256  a, param256  b) { return _mm256_cmp_ps(a, b, _CMP_LE_OS); }
    static Vc_INTRINSIC m256  Vc_CONST _mm256_cmpnle_ps  (param256  a, param256  b) { return _mm256_cmp_ps(a, b, _CMP_NLE_US); }
    static Vc_INTRINSIC m256  Vc_CONST _mm256_cmpgt_ps   (param256  a, param256  b) { return _mm256_cmp_ps(a, b, _CMP_NLE_US); }
    static Vc_INTRINSIC m256  Vc_CONST _mm256_cmpord_ps  (param256  a, param256  b) { return _mm256_cmp_ps(a, b, _CMP_ORD_Q); }
    static Vc_INTRINSIC m256  Vc_CONST _mm256_cmpunord_ps(param256  a, param256  b) { return _mm256_cmp_ps(a, b, _CMP_UNORD_Q); }

    static Vc_INTRINSIC m128i _mm_cmplt_epu16(param128i a, param128i b) {
        return _mm_cmplt_epi16(_mm_xor_si128(a, _mm_setmin_epi16()), _mm_xor_si128(b, _mm_setmin_epi16()));
    }
    static Vc_INTRINSIC m128i _mm_cmpgt_epu16(param128i a, param128i b) {
        return _mm_cmpgt_epi16(_mm_xor_si128(a, _mm_setmin_epi16()), _mm_xor_si128(b, _mm_setmin_epi16()));
    }

    /////////////////////// INTEGER OPS ///////////////////////
#define AVX_TO_SSE_2(name) \
    static Vc_INTRINSIC m256i Vc_CONST _mm256_##name(param256i a0, param256i b0) { \
        m128i a1 = _mm256_extractf128_si256(a0, 1); \
        m128i b1 = _mm256_extractf128_si256(b0, 1); \
        m128i r0 = _mm_##name(_mm256_castsi256_si128(a0), _mm256_castsi256_si128(b0)); \
        m128i r1 = _mm_##name(a1, b1); \
        return _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1); \
    }
#define AVX_TO_SSE_2_si128_si256(name) \
    static Vc_INTRINSIC m256i Vc_CONST _mm256_##name##_si256(param256i a0, param256i b0) { \
        m128i a1 = _mm256_extractf128_si256(a0, 1); \
        m128i b1 = _mm256_extractf128_si256(b0, 1); \
        m128i r0 = _mm_##name##_si128(_mm256_castsi256_si128(a0), _mm256_castsi256_si128(b0)); \
        m128i r1 = _mm_##name##_si128(a1, b1); \
        return _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1); \
    }
#define AVX_TO_SSE_1(name) \
    static Vc_INTRINSIC m256i Vc_CONST _mm256_##name(param256i a0) { \
        m128i a1 = _mm256_extractf128_si256(a0, 1); \
        m128i r0 = _mm_##name(_mm256_castsi256_si128(a0)); \
        m128i r1 = _mm_##name(a1); \
        return _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1); \
    }
#define AVX_TO_SSE_1i(name) \
    static Vc_INTRINSIC m256i Vc_CONST _mm256_##name(param256i a0, const int i) { \
        m128i a1 = _mm256_extractf128_si256(a0, 1); \
        m128i r0 = _mm_##name(_mm256_castsi256_si128(a0), i); \
        m128i r1 = _mm_##name(a1, i); \
        return _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1); \
    }

    AVX_TO_SSE_2(cmplt_epi8)
    AVX_TO_SSE_2(cmplt_epi16)
    AVX_TO_SSE_2(cmplt_epi32)
    AVX_TO_SSE_2(cmpeq_epi8)
    AVX_TO_SSE_2(cmpeq_epi16)
    AVX_TO_SSE_2(cmpeq_epi32)
    AVX_TO_SSE_2(cmpgt_epi8)
    AVX_TO_SSE_2(cmpgt_epi16)
    AVX_TO_SSE_2(cmpgt_epi32)

    // This code is AVX only (without AVX2). We never asked for AVX2 intrinsics. So go away... :)
#if defined _mm256_srli_si256
#undef _mm256_srli_si256
#endif
#if defined _mm256_slli_si256
#undef _mm256_slli_si256
#endif
#if defined _mm256_blend_epi16
#undef _mm256_blend_epi16
#endif
    static Vc_INTRINSIC m256i Vc_CONST _mm256_srli_si256(param256i a0, const int i) {
        const m128i vLo = _mm256_castsi256_si128(a0);
        const m128i vHi = _mm256_extractf128_si256(a0, 1);
        switch (i) {
        case  0: return a0;
        case  1: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo,  1)), _mm_srli_si128(vHi,  1), 1);
        case  2: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo,  2)), _mm_srli_si128(vHi,  2), 1);
        case  3: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo,  3)), _mm_srli_si128(vHi,  3), 1);
        case  4: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo,  4)), _mm_srli_si128(vHi,  4), 1);
        case  5: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo,  5)), _mm_srli_si128(vHi,  5), 1);
        case  6: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo,  6)), _mm_srli_si128(vHi,  6), 1);
        case  7: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo,  7)), _mm_srli_si128(vHi,  7), 1);
        case  8: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo,  8)), _mm_srli_si128(vHi,  8), 1);
        case  9: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo,  9)), _mm_srli_si128(vHi,  9), 1);
        case 10: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo, 10)), _mm_srli_si128(vHi, 10), 1);
        case 11: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo, 11)), _mm_srli_si128(vHi, 11), 1);
        case 12: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo, 12)), _mm_srli_si128(vHi, 12), 1);
        case 13: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo, 13)), _mm_srli_si128(vHi, 13), 1);
        case 14: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo, 14)), _mm_srli_si128(vHi, 14), 1);
        case 15: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo, 15)), _mm_srli_si128(vHi, 15), 1);
        case 16: return _mm256_permute2f128_si256(a0, a0, 0x81);
        case 17: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi,  1)), _mm256_castsi128_si256(_mm_srli_si128(vHi,  1)), 0x80);
        case 18: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi,  2)), _mm256_castsi128_si256(_mm_srli_si128(vHi,  2)), 0x80);
        case 19: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi,  3)), _mm256_castsi128_si256(_mm_srli_si128(vHi,  3)), 0x80);
        case 20: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi,  4)), _mm256_castsi128_si256(_mm_srli_si128(vHi,  4)), 0x80);
        case 21: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi,  5)), _mm256_castsi128_si256(_mm_srli_si128(vHi,  5)), 0x80);
        case 22: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi,  6)), _mm256_castsi128_si256(_mm_srli_si128(vHi,  6)), 0x80);
        case 23: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi,  7)), _mm256_castsi128_si256(_mm_srli_si128(vHi,  7)), 0x80);
        case 24: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi,  8)), _mm256_castsi128_si256(_mm_srli_si128(vHi,  8)), 0x80);
        case 25: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi,  9)), _mm256_castsi128_si256(_mm_srli_si128(vHi,  9)), 0x80);
        case 26: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi, 10)), _mm256_castsi128_si256(_mm_srli_si128(vHi, 10)), 0x80);
        case 27: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi, 11)), _mm256_castsi128_si256(_mm_srli_si128(vHi, 11)), 0x80);
        case 28: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi, 12)), _mm256_castsi128_si256(_mm_srli_si128(vHi, 12)), 0x80);
        case 29: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi, 13)), _mm256_castsi128_si256(_mm_srli_si128(vHi, 13)), 0x80);
        case 30: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi, 14)), _mm256_castsi128_si256(_mm_srli_si128(vHi, 14)), 0x80);
        case 31: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi, 15)), _mm256_castsi128_si256(_mm_srli_si128(vHi, 15)), 0x80);
        }
        return _mm256_setzero_si256();
    }
    static Vc_INTRINSIC m256i Vc_CONST _mm256_slli_si256(param256i a0, const int i) {
        const m128i vLo = _mm256_castsi256_si128(a0);
        const m128i vHi = _mm256_extractf128_si256(a0, 1);
        switch (i) {
        case  0: return a0;
        case  1: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  1)), _mm_alignr_epi8(vHi, vLo, 15), 1);
        case  2: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  2)), _mm_alignr_epi8(vHi, vLo, 14), 1);
        case  3: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  3)), _mm_alignr_epi8(vHi, vLo, 13), 1);
        case  4: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  4)), _mm_alignr_epi8(vHi, vLo, 12), 1);
        case  5: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  5)), _mm_alignr_epi8(vHi, vLo, 11), 1);
        case  6: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  6)), _mm_alignr_epi8(vHi, vLo, 10), 1);
        case  7: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  7)), _mm_alignr_epi8(vHi, vLo,  9), 1);
        case  8: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  8)), _mm_alignr_epi8(vHi, vLo,  8), 1);
        case  9: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  9)), _mm_alignr_epi8(vHi, vLo,  7), 1);
        case 10: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 10)), _mm_alignr_epi8(vHi, vLo,  6), 1);
        case 11: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 11)), _mm_alignr_epi8(vHi, vLo,  5), 1);
        case 12: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 12)), _mm_alignr_epi8(vHi, vLo,  4), 1);
        case 13: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 13)), _mm_alignr_epi8(vHi, vLo,  3), 1);
        case 14: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 14)), _mm_alignr_epi8(vHi, vLo,  2), 1);
        case 15: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 15)), _mm_alignr_epi8(vHi, vLo,  1), 1);
        case 16: return _mm256_permute2f128_si256(a0, a0, 0x8);
        case 17: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  1)), _mm256_castsi128_si256(_mm_slli_si128(vLo,  1)), 0x8);
        case 18: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  2)), _mm256_castsi128_si256(_mm_slli_si128(vLo,  2)), 0x8);
        case 19: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  3)), _mm256_castsi128_si256(_mm_slli_si128(vLo,  3)), 0x8);
        case 20: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  4)), _mm256_castsi128_si256(_mm_slli_si128(vLo,  4)), 0x8);
        case 21: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  5)), _mm256_castsi128_si256(_mm_slli_si128(vLo,  5)), 0x8);
        case 22: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  6)), _mm256_castsi128_si256(_mm_slli_si128(vLo,  6)), 0x8);
        case 23: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  7)), _mm256_castsi128_si256(_mm_slli_si128(vLo,  7)), 0x8);
        case 24: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  8)), _mm256_castsi128_si256(_mm_slli_si128(vLo,  8)), 0x8);
        case 25: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  9)), _mm256_castsi128_si256(_mm_slli_si128(vLo,  9)), 0x8);
        case 26: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 10)), _mm256_castsi128_si256(_mm_slli_si128(vLo, 10)), 0x8);
        case 27: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 11)), _mm256_castsi128_si256(_mm_slli_si128(vLo, 11)), 0x8);
        case 28: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 12)), _mm256_castsi128_si256(_mm_slli_si128(vLo, 12)), 0x8);
        case 29: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 13)), _mm256_castsi128_si256(_mm_slli_si128(vLo, 13)), 0x8);
        case 30: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 14)), _mm256_castsi128_si256(_mm_slli_si128(vLo, 14)), 0x8);
        case 31: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 15)), _mm256_castsi128_si256(_mm_slli_si128(vLo, 15)), 0x8);
        }
        return _mm256_setzero_si256();
    }

    static Vc_INTRINSIC m256i Vc_CONST _mm256_and_si256(param256i x, param256i y) {
        return _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y)));
    }
    static Vc_INTRINSIC m256i Vc_CONST _mm256_andnot_si256(param256i x, param256i y) {
        return _mm256_castps_si256(_mm256_andnot_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y)));
    }
    static Vc_INTRINSIC m256i Vc_CONST _mm256_or_si256(param256i x, param256i y) {
        return _mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y)));
    }
    static Vc_INTRINSIC m256i Vc_CONST _mm256_xor_si256(param256i x, param256i y) {
        return _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y)));
    }

    AVX_TO_SSE_2(packs_epi16)
    AVX_TO_SSE_2(packs_epi32)
    AVX_TO_SSE_2(packus_epi16)
    AVX_TO_SSE_2(unpackhi_epi8)
    AVX_TO_SSE_2(unpackhi_epi16)
    AVX_TO_SSE_2(unpackhi_epi32)
    AVX_TO_SSE_2(unpackhi_epi64)
    AVX_TO_SSE_2(unpacklo_epi8)
    AVX_TO_SSE_2(unpacklo_epi16)
    AVX_TO_SSE_2(unpacklo_epi32)
    AVX_TO_SSE_2(unpacklo_epi64)
    AVX_TO_SSE_2(add_epi8)
    AVX_TO_SSE_2(add_epi16)
    AVX_TO_SSE_2(add_epi32)
    AVX_TO_SSE_2(add_epi64)
    AVX_TO_SSE_2(adds_epi8)
    AVX_TO_SSE_2(adds_epi16)
    AVX_TO_SSE_2(adds_epu8)
    AVX_TO_SSE_2(adds_epu16)
    AVX_TO_SSE_2(sub_epi8)
    AVX_TO_SSE_2(sub_epi16)
    AVX_TO_SSE_2(sub_epi32)
    AVX_TO_SSE_2(sub_epi64)
    AVX_TO_SSE_2(subs_epi8)
    AVX_TO_SSE_2(subs_epi16)
    AVX_TO_SSE_2(subs_epu8)
    AVX_TO_SSE_2(subs_epu16)
    AVX_TO_SSE_2(madd_epi16)
    AVX_TO_SSE_2(mulhi_epi16)
    AVX_TO_SSE_2(mullo_epi16)
    AVX_TO_SSE_2(mul_epu32)
    AVX_TO_SSE_1i(slli_epi16)
    AVX_TO_SSE_1i(slli_epi32)
    AVX_TO_SSE_1i(slli_epi64)
    AVX_TO_SSE_1i(srai_epi16)
    AVX_TO_SSE_1i(srai_epi32)
    AVX_TO_SSE_1i(srli_epi16)
    AVX_TO_SSE_1i(srli_epi32)
    AVX_TO_SSE_1i(srli_epi64)
    AVX_TO_SSE_2(sll_epi16)
    AVX_TO_SSE_2(sll_epi32)
    AVX_TO_SSE_2(sll_epi64)
    AVX_TO_SSE_2(sra_epi16)
    AVX_TO_SSE_2(sra_epi32)
    AVX_TO_SSE_2(srl_epi16)
    AVX_TO_SSE_2(srl_epi32)
    AVX_TO_SSE_2(srl_epi64)
    AVX_TO_SSE_2(max_epi16)
    AVX_TO_SSE_2(max_epu8)
    AVX_TO_SSE_2(min_epi16)
    AVX_TO_SSE_2(min_epu8)
    Vc_INTRINSIC int Vc_CONST _mm256_movemask_epi8(param256i a0)
    {
        m128i a1 = _mm256_extractf128_si256(a0, 1);
        return (_mm_movemask_epi8(a1) << 16) | _mm_movemask_epi8(_mm256_castsi256_si128(a0));
    }
    AVX_TO_SSE_2(mulhi_epu16)
    // shufflehi_epi16
    // shufflelo_epi16 (param128i __A, const int __mask)
    // shuffle_epi32 (param128i __A, const int __mask)
    // maskmoveu_si128 (param128i __A, param128i __B, char *__C)
    AVX_TO_SSE_2(avg_epu8)
    AVX_TO_SSE_2(avg_epu16)
    AVX_TO_SSE_2(sad_epu8)
    // stream_si32 (int *__A, int __B)
    // stream_si128 (param128i *__A, param128i __B)
    // cvtsi32_si128 (int __A)
    // cvtsi64_si128 (long long __A)
    // cvtsi64x_si128 (long long __A)
    AVX_TO_SSE_2(hadd_epi16)
    AVX_TO_SSE_2(hadd_epi32)
    AVX_TO_SSE_2(hadds_epi16)
    AVX_TO_SSE_2(hsub_epi16)
    AVX_TO_SSE_2(hsub_epi32)
    AVX_TO_SSE_2(hsubs_epi16)
    AVX_TO_SSE_2(maddubs_epi16)
    AVX_TO_SSE_2(mulhrs_epi16)
    AVX_TO_SSE_2(shuffle_epi8)
    AVX_TO_SSE_2(sign_epi8)
    AVX_TO_SSE_2(sign_epi16)
    AVX_TO_SSE_2(sign_epi32)
    // alignr_epi8(param128i __X, param128i __Y, const int __N)
    AVX_TO_SSE_1(abs_epi8)
    AVX_TO_SSE_1(abs_epi16)
    AVX_TO_SSE_1(abs_epi32)
#if !defined(VC_REQUIRES_MACRO_FOR_IMMEDIATE_ARGUMENT)
    m256i Vc_INTRINSIC Vc_CONST _mm256_blend_epi16(param256i a0, param256i b0, const int m) {
        m128i a1 = _mm256_extractf128_si256(a0, 1);
        m128i b1 = _mm256_extractf128_si256(b0, 1);
        m128i r0 = _mm_blend_epi16(_mm256_castsi256_si128(a0), _mm256_castsi256_si128(b0), m & 0xff);
        m128i r1 = _mm_blend_epi16(a1, b1, m >> 8);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
    }
#else
#   define _mm256_blend_epi16(a0, b0, m) \
    _mm256_insertf128_si256( \
            _mm256_castsi128_si256( \
                _mm_blend_epi16( \
                    _mm256_castsi256_si128(a0), _mm256_castsi256_si128(b0), m & 0xff)), \
            _mm_blend_epi16(_mm256_extractf128_si256(a0, 1), _mm256_extractf128_si256(b0, 1), m >> 8);, 1)
#endif
    Vc_INTRINSIC m256i Vc_CONST _mm256_blendv_epi8(param256i a0, param256i b0, param256i m0) {
        m128i a1 = _mm256_extractf128_si256(a0, 1);
        m128i b1 = _mm256_extractf128_si256(b0, 1);
        m128i m1 = _mm256_extractf128_si256(m0, 1);
        m128i r0 = _mm_blendv_epi8(_mm256_castsi256_si128(a0), _mm256_castsi256_si128(b0), _mm256_castsi256_si128(m0));
        m128i r1 = _mm_blendv_epi8(a1, b1, m1);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
    }
    AVX_TO_SSE_2(cmpeq_epi64)
    AVX_TO_SSE_2(min_epi8)
    AVX_TO_SSE_2(max_epi8)
    AVX_TO_SSE_2(min_epu16)
    AVX_TO_SSE_2(max_epu16)
    AVX_TO_SSE_2(min_epi32)
    AVX_TO_SSE_2(max_epi32)
    AVX_TO_SSE_2(min_epu32)
    AVX_TO_SSE_2(max_epu32)
    AVX_TO_SSE_2(mullo_epi32)
    AVX_TO_SSE_2(mul_epi32)
#if !defined(VC_CLANG) || VC_CLANG > 0x30100
    // clang is missing _mm_minpos_epu16 from smmintrin.h
    // http://llvm.org/bugs/show_bug.cgi?id=12399
    AVX_TO_SSE_1(minpos_epu16)
#endif
    AVX_TO_SSE_1(cvtepi8_epi32)
    AVX_TO_SSE_1(cvtepi16_epi32)
    AVX_TO_SSE_1(cvtepi8_epi64)
    AVX_TO_SSE_1(cvtepi32_epi64)
    AVX_TO_SSE_1(cvtepi16_epi64)
    AVX_TO_SSE_1(cvtepi8_epi16)
    AVX_TO_SSE_1(cvtepu8_epi32)
    AVX_TO_SSE_1(cvtepu16_epi32)
    AVX_TO_SSE_1(cvtepu8_epi64)
    AVX_TO_SSE_1(cvtepu32_epi64)
    AVX_TO_SSE_1(cvtepu16_epi64)
    AVX_TO_SSE_1(cvtepu8_epi16)
    AVX_TO_SSE_2(packus_epi32)
    // mpsadbw_epu8 (param128i __X, param128i __Y, const int __M)
    // stream_load_si128 (param128i *__X)
    AVX_TO_SSE_2(cmpgt_epi64)

//X     static Vc_INTRINSIC m256i _mm256_cmplt_epu8 (param256i a, param256i b) { return _mm256_cmplt_epi8 (
//X             _mm256_xor_si256(a, _mm256_setmin_epi8 ()), _mm256_xor_si256(b, _mm256_setmin_epi8 ())); }
//X     static Vc_INTRINSIC m256i _mm256_cmpgt_epu8 (param256i a, param256i b) { return _mm256_cmpgt_epi8 (
//X             _mm256_xor_si256(a, _mm256_setmin_epi8 ()), _mm256_xor_si256(b, _mm256_setmin_epi8 ())); }
    static Vc_INTRINSIC m256i Vc_CONST _mm256_cmplt_epu32(param256i _a, param256i _b) {
        m256i a = _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(_a), _mm256_castsi256_ps(_mm256_setmin_epi32())));
        m256i b = _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(_b), _mm256_castsi256_ps(_mm256_setmin_epi32())));
        return _mm256_insertf128_si256(_mm256_castsi128_si256(
                    _mm_cmplt_epi32(_mm256_castsi256_si128(a), _mm256_castsi256_si128(b))),
                _mm_cmplt_epi32(_mm256_extractf128_si256(a, 1), _mm256_extractf128_si256(b, 1)), 1);
    }
    static Vc_INTRINSIC m256i Vc_CONST _mm256_cmpgt_epu32(param256i _a, param256i _b) {
        m256i a = _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(_a), _mm256_castsi256_ps(_mm256_setmin_epi32())));
        m256i b = _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(_b), _mm256_castsi256_ps(_mm256_setmin_epi32())));
        return _mm256_insertf128_si256(_mm256_castsi128_si256(
                    _mm_cmpgt_epi32(_mm256_castsi256_si128(a), _mm256_castsi256_si128(b))),
                _mm_cmpgt_epi32(_mm256_extractf128_si256(a, 1), _mm256_extractf128_si256(b, 1)), 1);
    }

        static Vc_INTRINSIC void _mm256_maskstore(float *mem, const param256 mask, const param256 v) {
#ifndef VC_MM256_MASKSTORE_WRONG_MASK_TYPE
            _mm256_maskstore_ps(mem, _mm256_castps_si256(mask), v);
#else
            _mm256_maskstore_ps(mem, mask, v);
#endif
        }
        static Vc_INTRINSIC void _mm256_maskstore(double *mem, const param256d mask, const param256d v) {
#ifndef VC_MM256_MASKSTORE_WRONG_MASK_TYPE
            _mm256_maskstore_pd(mem, _mm256_castpd_si256(mask), v);
#else
            _mm256_maskstore_pd(mem, mask, v);
#endif
        }
        static Vc_INTRINSIC void _mm256_maskstore(int *mem, const param256i mask, const param256i v) {
#ifndef VC_MM256_MASKSTORE_WRONG_MASK_TYPE
            _mm256_maskstore_ps(reinterpret_cast<float *>(mem), mask, _mm256_castsi256_ps(v));
#else
            _mm256_maskstore_ps(reinterpret_cast<float *>(mem), _mm256_castsi256_ps(mask), _mm256_castsi256_ps(v));
#endif
        }
        static Vc_INTRINSIC void _mm256_maskstore(unsigned int *mem, const param256i mask, const param256i v) {
            _mm256_maskstore(reinterpret_cast<int *>(mem), mask, v);
        }

#if defined(VC_IMPL_FMA4) && defined(VC_CLANG) && VC_CLANG < 0x30300
        // clang miscompiles _mm256_macc_ps: http://llvm.org/bugs/show_bug.cgi?id=15040
        static Vc_INTRINSIC __m256 my256_macc_ps(__m256 a, __m256 b, __m256 c) {
            __m256 r;
            // avoid loading c from memory as that would trigger the bug
            asm("vfmaddps %[c], %[b], %[a], %[r]" : [r]"=x"(r) : [a]"x"(a), [b]"x"(b), [c]"x"(c));
            return r;
        }
#ifdef _mm256_macc_ps
#undef _mm256_macc_ps
#endif
#define _mm256_macc_ps(a, b, c) Vc::AVX::my256_macc_ps(a, b, c)

        static Vc_INTRINSIC __m256d my256_macc_pd(__m256d a, __m256d b, __m256d c) {
            __m256d r;
            // avoid loading c from memory as that would trigger the bug
            asm("vfmaddpd %[c], %[b], %[a], %[r]" : [r]"=x"(r) : [a]"x"(a), [b]"x"(b), [c]"x"(c));
            return r;
        }
#ifdef _mm256_macc_pd
#undef _mm256_macc_pd
#endif
#define _mm256_macc_pd(a, b, c) Vc::AVX::my256_macc_pd(a, b, c)
#endif
} // namespace AVX
} // namespace Vc
} // namespace ROOT
#include "undomacros.h"

#include "shuffle.h"

#endif // VC_AVX_INTRINSICS_H
