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

#ifndef AVX_VECTORHELPER_H
#define AVX_VECTORHELPER_H

#include <limits>
#include "types.h"
#include "intrinsics.h"
#include "casts.h"
#include "macros.h"

namespace ROOT {
namespace Vc
{
namespace AVX
{

namespace Internal
{
Vc_INTRINSIC Vc_CONST m256 exponent(param256 v)
{
    m128i tmp0 = _mm_srli_epi32(avx_cast<m128i>(v), 23);
    m128i tmp1 = _mm_srli_epi32(avx_cast<m128i>(hi128(v)), 23);
    tmp0 = _mm_sub_epi32(tmp0, _mm_set1_epi32(0x7f));
    tmp1 = _mm_sub_epi32(tmp1, _mm_set1_epi32(0x7f));
    return _mm256_cvtepi32_ps(concat(tmp0, tmp1));
}
Vc_INTRINSIC Vc_CONST m256d exponent(param256d v)
{
    m128i tmp0 = _mm_srli_epi64(avx_cast<m128i>(v), 52);
    m128i tmp1 = _mm_srli_epi64(avx_cast<m128i>(hi128(v)), 52);
    tmp0 = _mm_sub_epi32(tmp0, _mm_set1_epi32(0x3ff));
    tmp1 = _mm_sub_epi32(tmp1, _mm_set1_epi32(0x3ff));
    return _mm256_cvtepi32_pd(avx_cast<m128i>(Mem::shuffle<X0, X2, Y0, Y2>(avx_cast<m128>(tmp0), avx_cast<m128>(tmp1))));
}
} // namespace Internal

#define OP0(name, code) static Vc_ALWAYS_INLINE Vc_CONST VectorType name() { return code; }
#define OP1(name, code) static Vc_ALWAYS_INLINE Vc_CONST VectorType name(VTArg a) { return code; }
#define OP2(name, code) static Vc_ALWAYS_INLINE Vc_CONST VectorType name(VTArg a, VTArg b) { return code; }
#define OP3(name, code) static Vc_ALWAYS_INLINE Vc_CONST VectorType name(VTArg a, VTArg b, VTArg c) { return code; }

        template<> struct VectorHelper<m256>
        {
            typedef m256 VectorType;
#ifdef VC_PASSING_VECTOR_BY_VALUE_IS_BROKEN
            typedef const VectorType & VTArg;
#else
            typedef const VectorType VTArg;
#endif
            template<typename A> static Vc_ALWAYS_INLINE_L Vc_PURE_L VectorType load(const float *x, A) Vc_ALWAYS_INLINE_R Vc_PURE_R;
            static Vc_ALWAYS_INLINE_L void store(float *mem, VTArg x, AlignedFlag) Vc_ALWAYS_INLINE_R;
            static Vc_ALWAYS_INLINE_L void store(float *mem, VTArg x, UnalignedFlag) Vc_ALWAYS_INLINE_R;
            static Vc_ALWAYS_INLINE_L void store(float *mem, VTArg x, StreamingAndAlignedFlag) Vc_ALWAYS_INLINE_R;
            static Vc_ALWAYS_INLINE_L void store(float *mem, VTArg x, StreamingAndUnalignedFlag) Vc_ALWAYS_INLINE_R;
            static Vc_ALWAYS_INLINE_L void store(float *mem, VTArg x, VTArg m, AlignedFlag) Vc_ALWAYS_INLINE_R;
            static Vc_ALWAYS_INLINE_L void store(float *mem, VTArg x, VTArg m, UnalignedFlag) Vc_ALWAYS_INLINE_R;
            static Vc_ALWAYS_INLINE_L void store(float *mem, VTArg x, VTArg m, StreamingAndAlignedFlag) Vc_ALWAYS_INLINE_R;
            static Vc_ALWAYS_INLINE_L void store(float *mem, VTArg x, VTArg m, StreamingAndUnalignedFlag) Vc_ALWAYS_INLINE_R;

            static Vc_ALWAYS_INLINE Vc_CONST VectorType cdab(VTArg x) { return _mm256_permute_ps(x, _MM_SHUFFLE(2, 3, 0, 1)); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType badc(VTArg x) { return _mm256_permute_ps(x, _MM_SHUFFLE(1, 0, 3, 2)); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType aaaa(VTArg x) { return _mm256_permute_ps(x, _MM_SHUFFLE(0, 0, 0, 0)); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType bbbb(VTArg x) { return _mm256_permute_ps(x, _MM_SHUFFLE(1, 1, 1, 1)); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cccc(VTArg x) { return _mm256_permute_ps(x, _MM_SHUFFLE(2, 2, 2, 2)); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType dddd(VTArg x) { return _mm256_permute_ps(x, _MM_SHUFFLE(3, 3, 3, 3)); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType dacb(VTArg x) { return _mm256_permute_ps(x, _MM_SHUFFLE(3, 0, 2, 1)); }

            OP0(allone, _mm256_setallone_ps())
            OP0(zero, _mm256_setzero_ps())
            OP2(or_, _mm256_or_ps(a, b))
            OP2(xor_, _mm256_xor_ps(a, b))
            OP2(and_, _mm256_and_ps(a, b))
            OP2(andnot_, _mm256_andnot_ps(a, b))
            OP3(blend, _mm256_blendv_ps(a, b, c))
        };

        template<> struct VectorHelper<m256d>
        {
            typedef m256d VectorType;
#ifdef VC_PASSING_VECTOR_BY_VALUE_IS_BROKEN
            typedef const VectorType & VTArg;
#else
            typedef const VectorType VTArg;
#endif
            template<typename A> static Vc_ALWAYS_INLINE_L Vc_PURE_L VectorType load(const double *x, A) Vc_ALWAYS_INLINE_R Vc_PURE_R;
            static Vc_ALWAYS_INLINE_L void store(double *mem, VTArg x, AlignedFlag) Vc_ALWAYS_INLINE_R;
            static Vc_ALWAYS_INLINE_L void store(double *mem, VTArg x, UnalignedFlag) Vc_ALWAYS_INLINE_R;
            static Vc_ALWAYS_INLINE_L void store(double *mem, VTArg x, StreamingAndAlignedFlag) Vc_ALWAYS_INLINE_R;
            static Vc_ALWAYS_INLINE_L void store(double *mem, VTArg x, StreamingAndUnalignedFlag) Vc_ALWAYS_INLINE_R;
            static Vc_ALWAYS_INLINE_L void store(double *mem, VTArg x, VTArg m, AlignedFlag) Vc_ALWAYS_INLINE_R;
            static Vc_ALWAYS_INLINE_L void store(double *mem, VTArg x, VTArg m, UnalignedFlag) Vc_ALWAYS_INLINE_R;
            static Vc_ALWAYS_INLINE_L void store(double *mem, VTArg x, VTArg m, StreamingAndAlignedFlag) Vc_ALWAYS_INLINE_R;
            static Vc_ALWAYS_INLINE_L void store(double *mem, VTArg x, VTArg m, StreamingAndUnalignedFlag) Vc_ALWAYS_INLINE_R;

            static VectorType cdab(VTArg x) { return _mm256_permute_pd(x, 5); }
            static VectorType badc(VTArg x) { return _mm256_permute2f128_pd(x, x, 1); }
            // aaaa bbbb cccc dddd specialized in vector.tcc
            static VectorType dacb(VTArg x) {
                const m128d cb = avx_cast<m128d>(_mm_alignr_epi8(avx_cast<m128i>(lo128(x)),
                            avx_cast<m128i>(hi128(x)), sizeof(double))); // XXX: lo and hi swapped?
                const m128d da = _mm_blend_pd(lo128(x), hi128(x), 0 + 2); // XXX: lo and hi swapped?
                return concat(cb, da);
            }

            OP0(allone, _mm256_setallone_pd())
            OP0(zero, _mm256_setzero_pd())
            OP2(or_, _mm256_or_pd(a, b))
            OP2(xor_, _mm256_xor_pd(a, b))
            OP2(and_, _mm256_and_pd(a, b))
            OP2(andnot_, _mm256_andnot_pd(a, b))
            OP3(blend, _mm256_blendv_pd(a, b, c))
        };

        template<> struct VectorHelper<m256i>
        {
            typedef m256i VectorType;
#ifdef VC_PASSING_VECTOR_BY_VALUE_IS_BROKEN
            typedef const VectorType & VTArg;
#else
            typedef const VectorType VTArg;
#endif
            template<typename T> static VectorType load(const T *x, AlignedFlag) Vc_PURE;
            template<typename T> static VectorType load(const T *x, UnalignedFlag) Vc_PURE;
            template<typename T> static VectorType load(const T *x, StreamingAndAlignedFlag) Vc_PURE;
            template<typename T> static VectorType load(const T *x, StreamingAndUnalignedFlag) Vc_PURE;
            template<typename T> static void store(T *mem, VTArg x, AlignedFlag);
            template<typename T> static void store(T *mem, VTArg x, UnalignedFlag);
            template<typename T> static void store(T *mem, VTArg x, StreamingAndAlignedFlag);
            template<typename T> static void store(T *mem, VTArg x, StreamingAndUnalignedFlag);
            template<typename T> static void store(T *mem, VTArg x, VTArg m, AlignedFlag);
            template<typename T> static void store(T *mem, VTArg x, VTArg m, UnalignedFlag);
            template<typename T> static void store(T *mem, VTArg x, VTArg m, StreamingAndAlignedFlag);
            template<typename T> static void store(T *mem, VTArg x, VTArg m, StreamingAndUnalignedFlag);

            static VectorType cdab(VTArg x) { return avx_cast<VectorType>(_mm256_permute_ps(avx_cast<m256>(x), _MM_SHUFFLE(2, 3, 0, 1))); }
            static VectorType badc(VTArg x) { return avx_cast<VectorType>(_mm256_permute_ps(avx_cast<m256>(x), _MM_SHUFFLE(1, 0, 3, 2))); }
            static VectorType aaaa(VTArg x) { return avx_cast<VectorType>(_mm256_permute_ps(avx_cast<m256>(x), _MM_SHUFFLE(0, 0, 0, 0))); }
            static VectorType bbbb(VTArg x) { return avx_cast<VectorType>(_mm256_permute_ps(avx_cast<m256>(x), _MM_SHUFFLE(1, 1, 1, 1))); }
            static VectorType cccc(VTArg x) { return avx_cast<VectorType>(_mm256_permute_ps(avx_cast<m256>(x), _MM_SHUFFLE(2, 2, 2, 2))); }
            static VectorType dddd(VTArg x) { return avx_cast<VectorType>(_mm256_permute_ps(avx_cast<m256>(x), _MM_SHUFFLE(3, 3, 3, 3))); }
            static VectorType dacb(VTArg x) { return avx_cast<VectorType>(_mm256_permute_ps(avx_cast<m256>(x), _MM_SHUFFLE(3, 0, 2, 1))); }

            OP0(allone, _mm256_setallone_si256())
            OP0(zero, _mm256_setzero_si256())
            OP2(or_, _mm256_or_si256(a, b))
            OP2(xor_, _mm256_xor_si256(a, b))
            OP2(and_, _mm256_and_si256(a, b))
            OP2(andnot_, _mm256_andnot_si256(a, b))
            OP3(blend, _mm256_blendv_epi8(a, b, c))
        };

        template<> struct VectorHelper<m128i>
        {
            typedef m128i VectorType;
#ifdef VC_PASSING_VECTOR_BY_VALUE_IS_BROKEN
            typedef const VectorType & VTArg;
#else
            typedef const VectorType VTArg;
#endif
            template<typename T> static VectorType load(const T *x, AlignedFlag) Vc_PURE;
            template<typename T> static VectorType load(const T *x, UnalignedFlag) Vc_PURE;
            template<typename T> static VectorType load(const T *x, StreamingAndAlignedFlag) Vc_PURE;
            template<typename T> static VectorType load(const T *x, StreamingAndUnalignedFlag) Vc_PURE;
            template<typename T> static void store(T *mem, VTArg x, AlignedFlag);
            template<typename T> static void store(T *mem, VTArg x, UnalignedFlag);
            template<typename T> static void store(T *mem, VTArg x, StreamingAndAlignedFlag);
            template<typename T> static void store(T *mem, VTArg x, StreamingAndUnalignedFlag);
            template<typename T> static void store(T *mem, VTArg x, VTArg m, AlignedFlag);
            template<typename T> static void store(T *mem, VTArg x, VTArg m, UnalignedFlag);
            template<typename T> static void store(T *mem, VTArg x, VTArg m, StreamingAndAlignedFlag);
            template<typename T> static void store(T *mem, VTArg x, VTArg m, StreamingAndUnalignedFlag);

            static VectorType cdab(VTArg x) { const __m128i tmp = _mm_shufflelo_epi16(x, _MM_SHUFFLE(2, 3, 0, 1)); return _mm_shufflehi_epi16(tmp, _MM_SHUFFLE(2, 3, 0, 1)); }
            static VectorType badc(VTArg x) { const __m128i tmp = _mm_shufflelo_epi16(x, _MM_SHUFFLE(1, 0, 3, 2)); return _mm_shufflehi_epi16(tmp, _MM_SHUFFLE(1, 0, 3, 2)); }
            static VectorType aaaa(VTArg x) { const __m128i tmp = _mm_shufflelo_epi16(x, _MM_SHUFFLE(0, 0, 0, 0)); return _mm_shufflehi_epi16(tmp, _MM_SHUFFLE(0, 0, 0, 0)); }
            static VectorType bbbb(VTArg x) { const __m128i tmp = _mm_shufflelo_epi16(x, _MM_SHUFFLE(1, 1, 1, 1)); return _mm_shufflehi_epi16(tmp, _MM_SHUFFLE(1, 1, 1, 1)); }
            static VectorType cccc(VTArg x) { const __m128i tmp = _mm_shufflelo_epi16(x, _MM_SHUFFLE(2, 2, 2, 2)); return _mm_shufflehi_epi16(tmp, _MM_SHUFFLE(2, 2, 2, 2)); }
            static VectorType dddd(VTArg x) { const __m128i tmp = _mm_shufflelo_epi16(x, _MM_SHUFFLE(3, 3, 3, 3)); return _mm_shufflehi_epi16(tmp, _MM_SHUFFLE(3, 3, 3, 3)); }
            static VectorType dacb(VTArg x) { const __m128i tmp = _mm_shufflelo_epi16(x, _MM_SHUFFLE(3, 0, 2, 1)); return _mm_shufflehi_epi16(tmp, _MM_SHUFFLE(3, 0, 2, 1)); }

            OP0(allone, _mm_setallone_si128())
            OP0(zero, _mm_setzero_si128())
            OP2(or_, _mm_or_si128(a, b))
            OP2(xor_, _mm_xor_si128(a, b))
            OP2(and_, _mm_and_si128(a, b))
            OP2(andnot_, _mm_andnot_si128(a, b))
            OP3(blend, _mm_blendv_epi8(a, b, c))
        };
#undef OP1
#undef OP2
#undef OP3

#define OP1(op) \
        static Vc_INTRINSIC VectorType Vc_CONST op(VTArg a) { return CAT(_mm256_##op##_, SUFFIX)(a); }
#define OP(op) \
        static Vc_INTRINSIC VectorType Vc_CONST op(VTArg a, VTArg b) { return CAT(_mm256_##op##_ , SUFFIX)(a, b); }
#define OP_(op) \
        static Vc_INTRINSIC VectorType Vc_CONST op(VTArg a, VTArg b) { return CAT(_mm256_##op    , SUFFIX)(a, b); }
#define OPx(op, op2) \
        static Vc_INTRINSIC VectorType Vc_CONST op(VTArg a, VTArg b) { return CAT(_mm256_##op2##_, SUFFIX)(a, b); }
#define OPcmp(op) \
        static Vc_INTRINSIC VectorType Vc_CONST cmp##op(VTArg a, VTArg b) { return CAT(_mm256_cmp##op##_, SUFFIX)(a, b); }
#define OP_CAST_(op) \
        static Vc_INTRINSIC VectorType Vc_CONST op(VTArg a, VTArg b) { return CAT(_mm256_castps_, SUFFIX)( \
            _mm256_##op##ps(CAT(CAT(_mm256_cast, SUFFIX), _ps)(a), \
              CAT(CAT(_mm256_cast, SUFFIX), _ps)(b))); \
        }
#define MINMAX \
        static Vc_INTRINSIC VectorType Vc_CONST min(VTArg a, VTArg b) { return CAT(_mm256_min_, SUFFIX)(a, b); } \
        static Vc_INTRINSIC VectorType Vc_CONST max(VTArg a, VTArg b) { return CAT(_mm256_max_, SUFFIX)(a, b); }

        template<> struct VectorHelper<double> {
            typedef m256d VectorType;
#ifdef VC_PASSING_VECTOR_BY_VALUE_IS_BROKEN
            typedef const VectorType & VTArg;
#else
            typedef const VectorType VTArg;
#endif
            typedef double EntryType;
            typedef double ConcatType;
#define SUFFIX pd

            static Vc_ALWAYS_INLINE VectorType notMaskedToZero(VTArg a, param256 mask) { return CAT(_mm256_and_, SUFFIX)(_mm256_castps_pd(mask), a); }
            static Vc_ALWAYS_INLINE VectorType set(const double a) { return CAT(_mm256_set1_, SUFFIX)(a); }
            static Vc_ALWAYS_INLINE VectorType set(const double a, const double b, const double c, const double d) {
                return CAT(_mm256_set_, SUFFIX)(a, b, c, d);
            }
            static Vc_ALWAYS_INLINE VectorType zero() { return CAT(_mm256_setzero_, SUFFIX)(); }
            static Vc_ALWAYS_INLINE VectorType one()  { return CAT(_mm256_setone_, SUFFIX)(); }// set(1.); }

            static inline void fma(VectorType &v1, VTArg v2, VTArg v3) {
#ifdef VC_IMPL_FMA4
                v1 = _mm256_macc_pd(v1, v2, v3);
#else
                VectorType h1 = _mm256_and_pd(v1, _mm256_broadcast_sd(reinterpret_cast<const double *>(&c_general::highMaskDouble)));
                VectorType h2 = _mm256_and_pd(v2, _mm256_broadcast_sd(reinterpret_cast<const double *>(&c_general::highMaskDouble)));
#if defined(VC_GCC) && VC_GCC < 0x40703
                // GCC before 4.7.3 uses an incorrect optimization where it replaces the subtraction with an andnot
                // http://gcc.gnu.org/bugzilla/show_bug.cgi?id=54703
                asm("":"+x"(h1), "+x"(h2));
#endif
                const VectorType l1 = _mm256_sub_pd(v1, h1);
                const VectorType l2 = _mm256_sub_pd(v2, h2);
                const VectorType ll = mul(l1, l2);
                const VectorType lh = add(mul(l1, h2), mul(h1, l2));
                const VectorType hh = mul(h1, h2);
                // ll < lh < hh for all entries is certain
                const VectorType lh_lt_v3 = cmplt(abs(lh), abs(v3)); // |lh| < |v3|
                const VectorType b = _mm256_blendv_pd(v3, lh, lh_lt_v3);
                const VectorType c = _mm256_blendv_pd(lh, v3, lh_lt_v3);
                v1 = add(add(ll, b), add(c, hh));
#endif
            }

            OP(add) OP(sub) OP(mul)
            OPcmp(eq) OPcmp(neq)
            OPcmp(lt) OPcmp(nlt)
            OPcmp(le) OPcmp(nle)

            OP1(sqrt)
            static Vc_ALWAYS_INLINE Vc_CONST VectorType rsqrt(VTArg x) {
                return _mm256_div_pd(one(), sqrt(x));
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType reciprocal(VTArg x) {
                return _mm256_div_pd(one(), x);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType isNaN(VTArg x) {
                return _mm256_cmpunord_pd(x, x);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType isFinite(VTArg x) {
                return _mm256_cmpord_pd(x, _mm256_mul_pd(zero(), x));
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType abs(VTArg a) {
                return CAT(_mm256_and_, SUFFIX)(a, _mm256_setabsmask_pd());
            }

            MINMAX
            static Vc_ALWAYS_INLINE Vc_CONST EntryType min(VTArg a) {
                m128d b = _mm_min_pd(avx_cast<m128d>(a), _mm256_extractf128_pd(a, 1));
                b = _mm_min_sd(b, _mm_unpackhi_pd(b, b));
                return _mm_cvtsd_f64(b);
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType max(VTArg a) {
                m128d b = _mm_max_pd(avx_cast<m128d>(a), _mm256_extractf128_pd(a, 1));
                b = _mm_max_sd(b, _mm_unpackhi_pd(b, b));
                return _mm_cvtsd_f64(b);
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType mul(VTArg a) {
                m128d b = _mm_mul_pd(avx_cast<m128d>(a), _mm256_extractf128_pd(a, 1));
                b = _mm_mul_sd(b, _mm_shuffle_pd(b, b, _MM_SHUFFLE2(0, 1)));
                return _mm_cvtsd_f64(b);
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType add(VTArg a) {
                m128d b = _mm_add_pd(avx_cast<m128d>(a), _mm256_extractf128_pd(a, 1));
                b = _mm_hadd_pd(b, b); // or: b = _mm_add_sd(b, _mm256_shuffle_pd(b, b, _MM_SHUFFLE2(0, 1)));
                return _mm_cvtsd_f64(b);
            }
#undef SUFFIX
            static Vc_ALWAYS_INLINE Vc_CONST VectorType round(VTArg a) {
                return _mm256_round_pd(a, _MM_FROUND_NINT);
            }
        };

        template<> struct VectorHelper<float> {
            typedef float EntryType;
            typedef m256 VectorType;
#ifdef VC_PASSING_VECTOR_BY_VALUE_IS_BROKEN
            typedef const VectorType & VTArg;
#else
            typedef const VectorType VTArg;
#endif
            typedef double ConcatType;
#define SUFFIX ps

            static Vc_ALWAYS_INLINE Vc_CONST VectorType notMaskedToZero(VTArg a, param256 mask) { return CAT(_mm256_and_, SUFFIX)(mask, a); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const float a) { return CAT(_mm256_set1_, SUFFIX)(a); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const float a, const float b, const float c, const float d,
                    const float e, const float f, const float g, const float h) {
                return CAT(_mm256_set_, SUFFIX)(a, b, c, d, e, f, g, h); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType zero() { return CAT(_mm256_setzero_, SUFFIX)(); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType one()  { return CAT(_mm256_setone_, SUFFIX)(); }// set(1.f); }
            static Vc_ALWAYS_INLINE Vc_CONST m256 concat(param256d a, param256d b) { return _mm256_insertf128_ps(avx_cast<m256>(_mm256_cvtpd_ps(a)), _mm256_cvtpd_ps(b), 1); }

            static inline void fma(VectorType &v1, VTArg v2, VTArg v3) {
#ifdef VC_IMPL_FMA4
                v1 = _mm256_macc_ps(v1, v2, v3);
#else
                m256d v1_0 = _mm256_cvtps_pd(lo128(v1));
                m256d v1_1 = _mm256_cvtps_pd(hi128(v1));
                m256d v2_0 = _mm256_cvtps_pd(lo128(v2));
                m256d v2_1 = _mm256_cvtps_pd(hi128(v2));
                m256d v3_0 = _mm256_cvtps_pd(lo128(v3));
                m256d v3_1 = _mm256_cvtps_pd(hi128(v3));
                v1 = AVX::concat(
                        _mm256_cvtpd_ps(_mm256_add_pd(_mm256_mul_pd(v1_0, v2_0), v3_0)),
                        _mm256_cvtpd_ps(_mm256_add_pd(_mm256_mul_pd(v1_1, v2_1), v3_1)));
#endif
            }

            OP(add) OP(sub) OP(mul)
            OPcmp(eq) OPcmp(neq)
            OPcmp(lt) OPcmp(nlt)
            OPcmp(le) OPcmp(nle)

            OP1(sqrt) OP1(rsqrt)
            static Vc_ALWAYS_INLINE Vc_CONST VectorType isNaN(VTArg x) {
                return _mm256_cmpunord_ps(x, x);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType isFinite(VTArg x) {
                return _mm256_cmpord_ps(x, _mm256_mul_ps(zero(), x));
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType reciprocal(VTArg x) {
                return _mm256_rcp_ps(x);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType abs(VTArg a) {
                return CAT(_mm256_and_, SUFFIX)(a, _mm256_setabsmask_ps());
            }

            MINMAX
            static Vc_ALWAYS_INLINE Vc_CONST EntryType min(VTArg a) {
                m128 b = _mm_min_ps(avx_cast<m128>(a), _mm256_extractf128_ps(a, 1));
                b = _mm_min_ps(b, _mm_movehl_ps(b, b));   // b = min(a0, a2), min(a1, a3), min(a2, a2), min(a3, a3)
                b = _mm_min_ss(b, _mm_shuffle_ps(b, b, _MM_SHUFFLE(1, 1, 1, 1))); // b = min(a0, a1), a1, a2, a3
                return _mm_cvtss_f32(b);
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType max(VTArg a) {
                m128 b = _mm_max_ps(avx_cast<m128>(a), _mm256_extractf128_ps(a, 1));
                b = _mm_max_ps(b, _mm_movehl_ps(b, b));   // b = max(a0, a2), max(a1, a3), max(a2, a2), max(a3, a3)
                b = _mm_max_ss(b, _mm_shuffle_ps(b, b, _MM_SHUFFLE(1, 1, 1, 1))); // b = max(a0, a1), a1, a2, a3
                return _mm_cvtss_f32(b);
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType mul(VTArg a) {
                m128 b = _mm_mul_ps(avx_cast<m128>(a), _mm256_extractf128_ps(a, 1));
                b = _mm_mul_ps(b, _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 1, 2, 3)));
                b = _mm_mul_ss(b, _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 2, 0, 1)));
                return _mm_cvtss_f32(b);
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType add(VTArg a) {
                m128 b = _mm_add_ps(avx_cast<m128>(a), _mm256_extractf128_ps(a, 1));
                b = _mm_add_ps(b, _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 1, 2, 3)));
                b = _mm_add_ss(b, _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 2, 0, 1)));
                return _mm_cvtss_f32(b);
            }
#undef SUFFIX
            static Vc_ALWAYS_INLINE Vc_CONST VectorType round(VTArg a) {
                return _mm256_round_ps(a, _MM_FROUND_NINT);
            }
        };

        template<> struct VectorHelper<sfloat> : public VectorHelper<float> {};

        template<> struct VectorHelper<int> {
            typedef int EntryType;
            typedef m256i VectorType;
#ifdef VC_PASSING_VECTOR_BY_VALUE_IS_BROKEN
            typedef const VectorType & VTArg;
#else
            typedef const VectorType VTArg;
#endif
            typedef long long ConcatType;
#define SUFFIX si256

            OP_(or_) OP_(and_) OP_(xor_)
            static Vc_INTRINSIC VectorType Vc_CONST zero() { return CAT(_mm256_setzero_, SUFFIX)(); }
            static Vc_INTRINSIC VectorType Vc_CONST notMaskedToZero(VTArg a, param256 mask) { return CAT(_mm256_and_, SUFFIX)(_mm256_castps_si256(mask), a); }
#undef SUFFIX
#define SUFFIX epi32
            static Vc_INTRINSIC VectorType Vc_CONST one() { return CAT(_mm256_setone_, SUFFIX)(); }

            static Vc_INTRINSIC VectorType Vc_CONST set(const int a) { return CAT(_mm256_set1_, SUFFIX)(a); }
            static Vc_INTRINSIC VectorType Vc_CONST set(const int a, const int b, const int c, const int d,
                    const int e, const int f, const int g, const int h) {
                return CAT(_mm256_set_, SUFFIX)(a, b, c, d, e, f, g, h); }

            static Vc_INTRINSIC void fma(VectorType &v1, VTArg v2, VTArg v3) { v1 = add(mul(v1, v2), v3); }

            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftLeft(VTArg a, int shift) {
                return CAT(_mm256_slli_, SUFFIX)(a, shift);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftRight(VTArg a, int shift) {
                return CAT(_mm256_srai_, SUFFIX)(a, shift);
            }
            OP1(abs)

            MINMAX
            static Vc_INTRINSIC EntryType Vc_CONST min(VTArg a) {
                m128i b = _mm_min_epi32(avx_cast<m128i>(a), _mm256_extractf128_si256(a, 1));
                b = _mm_min_epi32(b, _mm_shuffle_epi32(b, _MM_SHUFFLE(1, 0, 3, 2)));
                b = _mm_min_epi32(b, _mm_shufflelo_epi16(b, _MM_SHUFFLE(1, 0, 3, 2))); // using lo_epi16 for speed here
                return _mm_cvtsi128_si32(b);
            }
            static Vc_INTRINSIC EntryType Vc_CONST max(VTArg a) {
                m128i b = _mm_max_epi32(avx_cast<m128i>(a), _mm256_extractf128_si256(a, 1));
                b = _mm_max_epi32(b, _mm_shuffle_epi32(b, _MM_SHUFFLE(1, 0, 3, 2)));
                b = _mm_max_epi32(b, _mm_shufflelo_epi16(b, _MM_SHUFFLE(1, 0, 3, 2))); // using lo_epi16 for speed here
                return _mm_cvtsi128_si32(b);
            }
            static Vc_INTRINSIC EntryType Vc_CONST add(VTArg a) {
                m128i b = _mm_add_epi32(avx_cast<m128i>(a), _mm256_extractf128_si256(a, 1));
                b = _mm_add_epi32(b, _mm_shuffle_epi32(b, _MM_SHUFFLE(1, 0, 3, 2)));
                b = _mm_add_epi32(b, _mm_shufflelo_epi16(b, _MM_SHUFFLE(1, 0, 3, 2)));
                return _mm_cvtsi128_si32(b);
            }
            static Vc_INTRINSIC EntryType Vc_CONST mul(VTArg a) {
                m128i b = _mm_mullo_epi32(avx_cast<m128i>(a), _mm256_extractf128_si256(a, 1));
                b = _mm_mullo_epi32(b, _mm_shuffle_epi32(b, _MM_SHUFFLE(1, 0, 3, 2)));
                b = _mm_mullo_epi32(b, _mm_shufflelo_epi16(b, _MM_SHUFFLE(1, 0, 3, 2)));
                return _mm_cvtsi128_si32(b);
            }

            static Vc_INTRINSIC VectorType Vc_CONST mul(VTArg a, VTArg b) { return _mm256_mullo_epi32(a, b); }

            OP(add) OP(sub)
            OPcmp(eq)
            OPcmp(lt)
            OPcmp(gt)
            static Vc_INTRINSIC VectorType Vc_CONST cmpneq(VTArg a, VTArg b) { m256i x = cmpeq(a, b); return _mm256_andnot_si256(x, _mm256_setallone_si256()); }
            static Vc_INTRINSIC VectorType Vc_CONST cmpnlt(VTArg a, VTArg b) { m256i x = cmplt(a, b); return _mm256_andnot_si256(x, _mm256_setallone_si256()); }
            static Vc_INTRINSIC VectorType Vc_CONST cmple (VTArg a, VTArg b) { m256i x = cmpgt(a, b); return _mm256_andnot_si256(x, _mm256_setallone_si256()); }
            static Vc_INTRINSIC VectorType Vc_CONST cmpnle(VTArg a, VTArg b) { return cmpgt(a, b); }
#undef SUFFIX
            static Vc_INTRINSIC VectorType Vc_CONST round(VTArg a) { return a; }
        };

        template<> struct VectorHelper<unsigned int> {
            typedef unsigned int EntryType;
            typedef m256i VectorType;
#ifdef VC_PASSING_VECTOR_BY_VALUE_IS_BROKEN
            typedef const VectorType & VTArg;
#else
            typedef const VectorType VTArg;
#endif
            typedef unsigned long long ConcatType;
#define SUFFIX si256
            OP_CAST_(or_) OP_CAST_(and_) OP_CAST_(xor_)
            static Vc_INTRINSIC VectorType Vc_CONST zero() { return CAT(_mm256_setzero_, SUFFIX)(); }
            static Vc_INTRINSIC VectorType Vc_CONST notMaskedToZero(VTArg a, param256 mask) { return CAT(_mm256_and_, SUFFIX)(_mm256_castps_si256(mask), a); }

#undef SUFFIX
#define SUFFIX epu32
            static Vc_INTRINSIC VectorType Vc_CONST one() { return CAT(_mm256_setone_, SUFFIX)(); }

            MINMAX
            static Vc_INTRINSIC EntryType Vc_CONST min(VTArg a) {
                m128i b = _mm_min_epu32(avx_cast<m128i>(a), _mm256_extractf128_si256(a, 1));
                b = _mm_min_epu32(b, _mm_shuffle_epi32(b, _MM_SHUFFLE(1, 0, 3, 2)));
                b = _mm_min_epu32(b, _mm_shufflelo_epi16(b, _MM_SHUFFLE(1, 0, 3, 2))); // using lo_epi16 for speed here
                return _mm_cvtsi128_si32(b);
            }
            static Vc_INTRINSIC EntryType Vc_CONST max(VTArg a) {
                m128i b = _mm_max_epu32(avx_cast<m128i>(a), _mm256_extractf128_si256(a, 1));
                b = _mm_max_epu32(b, _mm_shuffle_epi32(b, _MM_SHUFFLE(1, 0, 3, 2)));
                b = _mm_max_epu32(b, _mm_shufflelo_epi16(b, _MM_SHUFFLE(1, 0, 3, 2))); // using lo_epi16 for speed here
                return _mm_cvtsi128_si32(b);
            }
            static Vc_INTRINSIC EntryType Vc_CONST add(VTArg a) {
                m128i b = _mm_add_epi32(avx_cast<m128i>(a), _mm256_extractf128_si256(a, 1));
                b = _mm_add_epi32(b, _mm_shuffle_epi32(b, _MM_SHUFFLE(1, 0, 3, 2)));
                b = _mm_add_epi32(b, _mm_shufflelo_epi16(b, _MM_SHUFFLE(1, 0, 3, 2)));
                return _mm_cvtsi128_si32(b);
            }
            static Vc_INTRINSIC EntryType Vc_CONST mul(VTArg a) {
                m128i b = _mm_mullo_epi32(avx_cast<m128i>(a), _mm256_extractf128_si256(a, 1));
                b = _mm_mullo_epi32(b, _mm_shuffle_epi32(b, _MM_SHUFFLE(1, 0, 3, 2)));
                b = _mm_mullo_epi32(b, _mm_shufflelo_epi16(b, _MM_SHUFFLE(1, 0, 3, 2)));
                return _mm_cvtsi128_si32(b);
            }

            static Vc_INTRINSIC VectorType Vc_CONST mul(VTArg a, VTArg b) { return _mm256_mullo_epi32(a, b); }
            static Vc_INTRINSIC void fma(VectorType &v1, VTArg v2, VTArg v3) { v1 = add(mul(v1, v2), v3); }

#undef SUFFIX
#define SUFFIX epi32
            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftLeft(VTArg a, int shift) {
                return CAT(_mm256_slli_, SUFFIX)(a, shift);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftRight(VTArg a, int shift) {
                return CAT(_mm256_srli_, SUFFIX)(a, shift);
            }
            static Vc_INTRINSIC VectorType Vc_CONST set(const unsigned int a) { return CAT(_mm256_set1_, SUFFIX)(a); }
            static Vc_INTRINSIC VectorType Vc_CONST set(const unsigned int a, const unsigned int b, const unsigned int c, const unsigned int d,
                    const unsigned int e, const unsigned int f, const unsigned int g, const unsigned int h) {
                return CAT(_mm256_set_, SUFFIX)(a, b, c, d, e, f, g, h); }

            OP(add) OP(sub)
            OPcmp(eq)
            static Vc_INTRINSIC VectorType Vc_CONST cmpneq(VTArg a, VTArg b) { return _mm256_andnot_si256(cmpeq(a, b), _mm256_setallone_si256()); }

#ifndef USE_INCORRECT_UNSIGNED_COMPARE
            static Vc_INTRINSIC VectorType Vc_CONST cmplt(VTArg a, VTArg b) {
                return _mm256_cmplt_epu32(a, b);
            }
            static Vc_INTRINSIC VectorType Vc_CONST cmpgt(VTArg a, VTArg b) {
                return _mm256_cmpgt_epu32(a, b);
            }
#else
            OPcmp(lt)
            OPcmp(gt)
#endif
            static Vc_INTRINSIC VectorType Vc_CONST cmpnlt(VTArg a, VTArg b) { return _mm256_andnot_si256(cmplt(a, b), _mm256_setallone_si256()); }
            static Vc_INTRINSIC VectorType Vc_CONST cmple (VTArg a, VTArg b) { return _mm256_andnot_si256(cmpgt(a, b), _mm256_setallone_si256()); }
            static Vc_INTRINSIC VectorType Vc_CONST cmpnle(VTArg a, VTArg b) { return cmpgt(a, b); }

#undef SUFFIX
            static Vc_INTRINSIC VectorType Vc_CONST round(VTArg a) { return a; }
        };

        template<> struct VectorHelper<signed short> {
            typedef VectorTypeHelper<signed short>::Type VectorType;
#ifdef VC_PASSING_VECTOR_BY_VALUE_IS_BROKEN
            typedef const VectorType & VTArg;
#else
            typedef const VectorType VTArg;
#endif
            typedef signed short EntryType;
            typedef int ConcatType;

            static Vc_INTRINSIC VectorType Vc_CONST or_(VTArg a, VTArg b) { return _mm_or_si128(a, b); }
            static Vc_INTRINSIC VectorType Vc_CONST and_(VTArg a, VTArg b) { return _mm_and_si128(a, b); }
            static Vc_INTRINSIC VectorType Vc_CONST xor_(VTArg a, VTArg b) { return _mm_xor_si128(a, b); }
            static Vc_INTRINSIC VectorType Vc_CONST zero() { return _mm_setzero_si128(); }
            static Vc_INTRINSIC VectorType Vc_CONST notMaskedToZero(VTArg a, param128 mask) { return _mm_and_si128(_mm_castps_si128(mask), a); }

#define SUFFIX epi16
            static Vc_INTRINSIC VectorType Vc_CONST one() { return CAT(_mm_setone_, SUFFIX)(); }

            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftLeft(VTArg a, int shift) {
                return CAT(_mm_slli_, SUFFIX)(a, shift);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftRight(VTArg a, int shift) {
                return CAT(_mm_srai_, SUFFIX)(a, shift);
            }
            static Vc_INTRINSIC VectorType Vc_CONST set(const EntryType a) { return CAT(_mm_set1_, SUFFIX)(a); }
            static Vc_INTRINSIC VectorType Vc_CONST set(const EntryType a, const EntryType b, const EntryType c, const EntryType d,
                    const EntryType e, const EntryType f, const EntryType g, const EntryType h) {
                return CAT(_mm_set_, SUFFIX)(a, b, c, d, e, f, g, h);
            }

            static Vc_INTRINSIC void fma(VectorType &v1, VTArg v2, VTArg v3) {
                v1 = add(mul(v1, v2), v3);
            }

            static Vc_INTRINSIC VectorType Vc_CONST abs(VTArg a) { return _mm_abs_epi16(a); }
            static Vc_INTRINSIC VectorType Vc_CONST mul(VTArg a, VTArg b) { return _mm_mullo_epi16(a, b); }
            static Vc_INTRINSIC VectorType Vc_CONST min(VTArg a, VTArg b) { return _mm_min_epi16(a, b); }
            static Vc_INTRINSIC VectorType Vc_CONST max(VTArg a, VTArg b) { return _mm_max_epi16(a, b); }

            static Vc_INTRINSIC EntryType Vc_CONST min(VTArg _a) {
                // reminder: _MM_SHUFFLE(3, 2, 1, 0) means "no change"
                VectorType a = min(_a, _mm_shuffle_epi32(_a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }
            static Vc_INTRINSIC EntryType Vc_CONST max(VTArg _a) {
                // reminder: _MM_SHUFFLE(3, 2, 1, 0) means "no change"
                VectorType a = max(_a, _mm_shuffle_epi32(_a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }
            static Vc_INTRINSIC EntryType Vc_CONST mul(VTArg _a) {
                VectorType a = mul(_a, _mm_shuffle_epi32(_a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = mul(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = mul(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }
            static Vc_INTRINSIC EntryType Vc_CONST add(VTArg _a) {
                VectorType a = add(_a, _mm_shuffle_epi32(_a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = add(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = add(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }

            static Vc_INTRINSIC VectorType Vc_CONST add(VTArg a, VTArg b) { return _mm_add_epi16(a, b); }
            static Vc_INTRINSIC VectorType Vc_CONST sub(VTArg a, VTArg b) { return _mm_sub_epi16(a, b); }
            static Vc_INTRINSIC VectorType Vc_CONST cmpeq(VTArg a, VTArg b) { return _mm_cmpeq_epi16(a, b); }
            static Vc_INTRINSIC VectorType Vc_CONST cmplt(VTArg a, VTArg b) { return _mm_cmplt_epi16(a, b); }
            static Vc_INTRINSIC VectorType Vc_CONST cmpgt(VTArg a, VTArg b) { return _mm_cmpgt_epi16(a, b); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmpneq(VTArg a, VTArg b) { m128i x = cmpeq(a, b); return _mm_andnot_si128(x, _mm_setallone_si128()); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmpnlt(VTArg a, VTArg b) { m128i x = cmplt(a, b); return _mm_andnot_si128(x, _mm_setallone_si128()); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmple (VTArg a, VTArg b) { m128i x = cmpgt(a, b); return _mm_andnot_si128(x, _mm_setallone_si128()); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmpnle(VTArg a, VTArg b) { return cmpgt(a, b); }
#undef SUFFIX
            static Vc_ALWAYS_INLINE Vc_CONST VectorType round(VTArg a) { return a; }
        };

        template<> struct VectorHelper<unsigned short> {
            typedef VectorTypeHelper<unsigned short>::Type VectorType;
#ifdef VC_PASSING_VECTOR_BY_VALUE_IS_BROKEN
            typedef const VectorType & VTArg;
#else
            typedef const VectorType VTArg;
#endif
            typedef unsigned short EntryType;
            typedef unsigned int ConcatType;

            static Vc_INTRINSIC VectorType Vc_CONST or_(VTArg a, VTArg b) { return _mm_or_si128(a, b); }
            static Vc_INTRINSIC VectorType Vc_CONST and_(VTArg a, VTArg b) { return _mm_and_si128(a, b); }
            static Vc_INTRINSIC VectorType Vc_CONST xor_(VTArg a, VTArg b) { return _mm_xor_si128(a, b); }
            static Vc_INTRINSIC VectorType Vc_CONST zero() { return _mm_setzero_si128(); }
            static Vc_INTRINSIC VectorType Vc_CONST notMaskedToZero(VTArg a, param128 mask) { return _mm_and_si128(_mm_castps_si128(mask), a); }
            static Vc_INTRINSIC VectorType Vc_CONST one() { return _mm_setone_epu16(); }

            static Vc_INTRINSIC VectorType Vc_CONST mul(VTArg a, VTArg b) { return _mm_mullo_epi16(a, b); }
            static Vc_INTRINSIC VectorType Vc_CONST min(VTArg a, VTArg b) { return _mm_min_epu16(a, b); }
            static Vc_INTRINSIC VectorType Vc_CONST max(VTArg a, VTArg b) { return _mm_max_epu16(a, b); }

#define SUFFIX epi16
            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftLeft(VTArg a, int shift) {
                return CAT(_mm_slli_, SUFFIX)(a, shift);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftRight(VTArg a, int shift) {
                return CAT(_mm_srli_, SUFFIX)(a, shift);
            }
            static Vc_INTRINSIC EntryType Vc_CONST min(VTArg _a) {
                // reminder: _MM_SHUFFLE(3, 2, 1, 0) means "no change"
                VectorType a = min(_a, _mm_shuffle_epi32(_a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }
            static Vc_INTRINSIC EntryType Vc_CONST max(VTArg _a) {
                // reminder: _MM_SHUFFLE(3, 2, 1, 0) means "no change"
                VectorType a = max(_a, _mm_shuffle_epi32(_a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }
            static Vc_INTRINSIC EntryType Vc_CONST mul(VTArg _a) {
                // reminder: _MM_SHUFFLE(3, 2, 1, 0) means "no change"
                VectorType a = mul(_a, _mm_shuffle_epi32(_a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = mul(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = mul(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }
            static Vc_INTRINSIC EntryType Vc_CONST add(VTArg _a) {
                // reminder: _MM_SHUFFLE(3, 2, 1, 0) means "no change"
                VectorType a = add(_a, _mm_shuffle_epi32(_a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = add(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = add(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }
            static Vc_INTRINSIC VectorType Vc_CONST set(const EntryType a) { return CAT(_mm_set1_, SUFFIX)(a); }
            static Vc_INTRINSIC VectorType Vc_CONST set(const EntryType a, const EntryType b, const EntryType c,
                    const EntryType d, const EntryType e, const EntryType f,
                    const EntryType g, const EntryType h) {
                return CAT(_mm_set_, SUFFIX)(a, b, c, d, e, f, g, h);
            }
            static Vc_INTRINSIC void fma(VectorType &v1, VTArg v2, VTArg v3) { v1 = add(mul(v1, v2), v3); }

            static Vc_INTRINSIC VectorType Vc_CONST add(VTArg a, VTArg b) { return _mm_add_epi16(a, b); }
            static Vc_INTRINSIC VectorType Vc_CONST sub(VTArg a, VTArg b) { return _mm_sub_epi16(a, b); }
            static Vc_INTRINSIC VectorType Vc_CONST cmpeq(VTArg a, VTArg b) { return _mm_cmpeq_epi16(a, b); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmpneq(VTArg a, VTArg b) { return _mm_andnot_si128(cmpeq(a, b), _mm_setallone_si128()); }

#ifndef USE_INCORRECT_UNSIGNED_COMPARE
            static Vc_INTRINSIC VectorType Vc_CONST cmplt(VTArg a, VTArg b) { return _mm_cmplt_epu16(a, b); }
            static Vc_INTRINSIC VectorType Vc_CONST cmpgt(VTArg a, VTArg b) { return _mm_cmpgt_epu16(a, b); }
#else
            static Vc_INTRINSIC VectorType Vc_CONST cmplt(VTArg a, VTArg b) { return _mm_cmplt_epi16(a, b); }
            static Vc_INTRINSIC VectorType Vc_CONST cmpgt(VTArg a, VTArg b) { return _mm_cmpgt_epi16(a, b); }
#endif
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmpnlt(VTArg a, VTArg b) { return _mm_andnot_si128(cmplt(a, b), _mm_setallone_si128()); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmple (VTArg a, VTArg b) { return _mm_andnot_si128(cmpgt(a, b), _mm_setallone_si128()); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmpnle(VTArg a, VTArg b) { return cmpgt(a, b); }
#undef SUFFIX
            static Vc_ALWAYS_INLINE Vc_CONST VectorType round(VTArg a) { return a; }
        };
#undef OP1
#undef OP
#undef OP_
#undef OPx
#undef OPcmp

template<> struct VectorHelper<char>
{
    typedef VectorTypeHelper<char>::Type VectorType;
#ifdef VC_PASSING_VECTOR_BY_VALUE_IS_BROKEN
    typedef const VectorType & VTArg;
#else
    typedef const VectorType VTArg;
#endif
    typedef char EntryType;
    typedef short ConcatType;
};

template<> struct VectorHelper<unsigned char>
{
    typedef VectorTypeHelper<unsigned char>::Type VectorType;
#ifdef VC_PASSING_VECTOR_BY_VALUE_IS_BROKEN
    typedef const VectorType & VTArg;
#else
    typedef const VectorType VTArg;
#endif
    typedef unsigned char EntryType;
    typedef unsigned short ConcatType;
};

} // namespace AVX
} // namespace Vc
} // namespace ROOT

#include "vectorhelper.tcc"
#include "undomacros.h"

#endif // AVX_VECTORHELPER_H
