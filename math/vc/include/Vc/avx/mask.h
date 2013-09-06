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

#ifndef VC_AVX_MASK_H
#define VC_AVX_MASK_H

#include "intrinsics.h"
#include "../common/bitscanintrinsics.h"
#include "macros.h"

namespace ROOT {
namespace Vc
{
namespace AVX
{

template<unsigned int VectorSize> class Mask<VectorSize, 32u>
{
    friend class Mask<4u, 32u>; // double_v
    friend class Mask<8u, 32u>; // float_v, (u)int_v
    friend class Mask<8u, 16u>; // (u)short_v
    friend class Mask<16u, 16u>; // (u)char_v
    public:
        FREE_STORE_OPERATORS_ALIGNED(32)

        // abstracts the way Masks are passed to functions, it can easily be changed to const ref here
#if defined VC_MSVC && defined _WIN32
        typedef const Mask<VectorSize, 32u> &AsArg;
#else
        typedef Mask<VectorSize, 32u> AsArg;
#endif

        Vc_ALWAYS_INLINE Mask() {}
        Vc_ALWAYS_INLINE Mask(param256  x) : k(x) {}
        Vc_ALWAYS_INLINE Mask(param256d x) : k(_mm256_castpd_ps(x)) {}
        Vc_ALWAYS_INLINE Mask(param256i x) : k(_mm256_castsi256_ps(x)) {}
#ifdef VC_UNCONDITIONAL_AVX2_INTRINSICS
        Vc_ALWAYS_INLINE Mask(__m256  x) : k(x) {}
        Vc_ALWAYS_INLINE Mask(__m256d x) : k(_mm256_castpd_ps(x)) {}
        Vc_ALWAYS_INLINE Mask(__m256i x) : k(_mm256_castsi256_ps(x)) {}
#endif
        Vc_ALWAYS_INLINE explicit Mask(VectorSpecialInitializerZero::ZEnum) : k(_mm256_setzero_ps()) {}
        Vc_ALWAYS_INLINE explicit Mask(VectorSpecialInitializerOne::OEnum) : k(_mm256_setallone_ps()) {}
        Vc_ALWAYS_INLINE explicit Mask(bool b) : k(b ? _mm256_setallone_ps() : m256(_mm256_setzero_ps())) {}
        Vc_ALWAYS_INLINE Mask(const Mask &rhs) : k(rhs.k) {}
        Vc_ALWAYS_INLINE Mask(const Mask<VectorSize, 16u> &rhs) : k(avx_cast<m256>(concat(
                        _mm_unpacklo_epi16(rhs.dataI(), rhs.dataI()),
                        _mm_unpackhi_epi16(rhs.dataI(), rhs.dataI())))) {}
        Vc_ALWAYS_INLINE_L Mask(const Mask<VectorSize * 2, 32u> &m) Vc_ALWAYS_INLINE_R;
        Vc_ALWAYS_INLINE_L Mask(const Mask<VectorSize / 2, 32u> &m) Vc_ALWAYS_INLINE_R;

        Vc_ALWAYS_INLINE bool operator==(const Mask &rhs) const { return 0 != _mm256_testc_ps(k, rhs.k); }
        Vc_ALWAYS_INLINE bool operator!=(const Mask &rhs) const { return 0 == _mm256_testc_ps(k, rhs.k); }

        Vc_ALWAYS_INLINE Mask operator!() const { return _mm256_andnot_ps(data(), _mm256_setallone_ps()); }

        Vc_ALWAYS_INLINE Mask &operator&=(const Mask &rhs) { k = _mm256_and_ps(k, rhs.k); return *this; }
        Vc_ALWAYS_INLINE Mask &operator|=(const Mask &rhs) { k = _mm256_or_ps (k, rhs.k); return *this; }
        Vc_ALWAYS_INLINE Mask &operator^=(const Mask &rhs) { k = _mm256_xor_ps(k, rhs.k); return *this; }

        // no need for expression template optimizations because cmp(n)eq for floats are not bitwise
        // compares
        Vc_ALWAYS_INLINE bool isFull () const { return 0 != _mm256_testc_ps(k, _mm256_setallone_ps()); }
        Vc_ALWAYS_INLINE bool isEmpty() const { return 0 != _mm256_testz_ps(k, k); }
        Vc_ALWAYS_INLINE bool isMix  () const { return 0 != _mm256_testnzc_ps(k, _mm256_setallone_ps()); }

#ifndef VC_NO_AUTOMATIC_BOOL_FROM_MASK
        Vc_ALWAYS_INLINE operator bool() const { return isFull(); }
#endif

        Vc_ALWAYS_INLINE_L Vc_PURE_L int shiftMask() const Vc_ALWAYS_INLINE_R Vc_PURE_R;
        Vc_ALWAYS_INLINE_L Vc_PURE_L int toInt() const Vc_ALWAYS_INLINE_R Vc_PURE_R;

        Vc_ALWAYS_INLINE m256  data () const { return k; }
        Vc_ALWAYS_INLINE m256i dataI() const { return _mm256_castps_si256(k); }
        Vc_ALWAYS_INLINE m256d dataD() const { return _mm256_castps_pd(k); }

        Vc_ALWAYS_INLINE_L Vc_PURE_L bool operator[](int index) const Vc_ALWAYS_INLINE_R Vc_PURE_R;

        Vc_ALWAYS_INLINE_L Vc_PURE_L int count() const Vc_ALWAYS_INLINE_R Vc_PURE_R;
        Vc_ALWAYS_INLINE_L Vc_PURE_L int firstOne() const Vc_ALWAYS_INLINE_R Vc_PURE_R;

    private:
#ifdef VC_COMPILE_BENCHMARKS
    public:
#endif
        m256 k;
};

template<unsigned int VectorSize> class Mask<VectorSize, 16u>
{
    friend class Mask<4u, 32u>; // double_v
    friend class Mask<8u, 32u>; // float_v, (u)int_v
    friend class Mask<8u, 16u>; // (u)short_v
    friend class Mask<16u, 16u>; // (u)char_v
    public:
        FREE_STORE_OPERATORS_ALIGNED(16)

        // abstracts the way Masks are passed to functions, it can easily be changed to const ref here
#if defined VC_MSVC && defined _WIN32
        typedef const Mask<VectorSize, 16u> &AsArg;
#else
        typedef Mask<VectorSize, 16u> AsArg;
#endif

        Vc_ALWAYS_INLINE Mask() {}
        Vc_ALWAYS_INLINE Mask(param128  x) : k(x) {}
        Vc_ALWAYS_INLINE Mask(param128d x) : k(_mm_castpd_ps(x)) {}
        Vc_ALWAYS_INLINE Mask(param128i x) : k(_mm_castsi128_ps(x)) {}
#ifdef VC_UNCONDITIONAL_AVX2_INTRINSICS
        Vc_ALWAYS_INLINE Mask(__m128  x) : k(x) {}
        Vc_ALWAYS_INLINE Mask(__m128d x) : k(_mm_castpd_ps(x)) {}
        Vc_ALWAYS_INLINE Mask(__m128i x) : k(_mm_castsi128_ps(x)) {}
#endif
        Vc_ALWAYS_INLINE explicit Mask(VectorSpecialInitializerZero::ZEnum) : k(_mm_setzero_ps()) {}
        Vc_ALWAYS_INLINE explicit Mask(VectorSpecialInitializerOne::OEnum) : k(_mm_setallone_ps()) {}
        Vc_ALWAYS_INLINE explicit Mask(bool b) : k(b ? _mm_setallone_ps() : m128(_mm_setzero_ps())) {}
        Vc_ALWAYS_INLINE Mask(const Mask &rhs) : k(rhs.k) {}
        Vc_ALWAYS_INLINE Mask(const Mask<VectorSize, 32u> &rhs) : k(avx_cast<m128>(
                _mm_packs_epi32(avx_cast<m128i>(rhs.data()), _mm256_extractf128_si256(rhs.dataI(), 1)))) {}
        Vc_ALWAYS_INLINE Mask(const Mask<VectorSize / 2, 16u> *a) : k(avx_cast<m128>(
                _mm_packs_epi16(a[0].dataI(), a[1].dataI()))) {}

        Vc_ALWAYS_INLINE bool operator==(const Mask &rhs) const { return 0 != _mm_testc_si128(dataI(), rhs.dataI()); }
        Vc_ALWAYS_INLINE bool operator!=(const Mask &rhs) const { return 0 == _mm_testc_si128(dataI(), rhs.dataI()); }

        Vc_ALWAYS_INLINE Mask operator!() const { return _mm_andnot_ps(data(), _mm_setallone_ps()); }

        Vc_ALWAYS_INLINE Mask &operator&=(const Mask &rhs) { k = _mm_and_ps(k, rhs.k); return *this; }
        Vc_ALWAYS_INLINE Mask &operator|=(const Mask &rhs) { k = _mm_or_ps (k, rhs.k); return *this; }
        Vc_ALWAYS_INLINE Mask &operator^=(const Mask &rhs) { k = _mm_xor_ps(k, rhs.k); return *this; }

        // TODO: use expression templates to optimize (v1 == v2).isFull() and friends
        Vc_ALWAYS_INLINE bool isFull () const { return 0 != _mm_testc_si128(dataI(), _mm_setallone_si128()); }
        Vc_ALWAYS_INLINE bool isEmpty() const { return 0 != _mm_testz_si128(dataI(), dataI()); }
        Vc_ALWAYS_INLINE bool isMix  () const { return 0 != _mm_testnzc_si128(dataI(), _mm_setallone_si128()); }

#ifndef VC_NO_AUTOMATIC_BOOL_FROM_MASK
        Vc_ALWAYS_INLINE operator bool() const { return isFull(); }
#endif

        Vc_ALWAYS_INLINE_L Vc_PURE_L int shiftMask() const Vc_ALWAYS_INLINE_R Vc_PURE_R;
        Vc_ALWAYS_INLINE_L Vc_PURE_L int toInt() const Vc_ALWAYS_INLINE_R Vc_PURE_R;

        Vc_ALWAYS_INLINE m128  data () const { return k; }
        Vc_ALWAYS_INLINE m128i dataI() const { return avx_cast<m128i>(k); }
        Vc_ALWAYS_INLINE m128d dataD() const { return avx_cast<m128d>(k); }

        Vc_ALWAYS_INLINE_L Vc_PURE_L bool operator[](int index) const Vc_ALWAYS_INLINE_R Vc_PURE_R;

        Vc_ALWAYS_INLINE_L Vc_PURE_L int count() const Vc_ALWAYS_INLINE_R Vc_PURE_R;
        Vc_ALWAYS_INLINE_L Vc_PURE_L int firstOne() const Vc_ALWAYS_INLINE_R Vc_PURE_R;

    private:
#ifdef VC_COMPILE_BENCHMARKS
    public:
#endif
        m128 k;
};

struct ForeachHelper
{
    size_t mask;
    bool brk;
    bool outerBreak;
    Vc_ALWAYS_INLINE ForeachHelper(size_t _mask) : mask(_mask), brk(false), outerBreak(false) {}
    Vc_ALWAYS_INLINE bool outer() const { return mask != 0 && !outerBreak; }
    Vc_ALWAYS_INLINE bool inner() { return (brk = !brk); }
    Vc_ALWAYS_INLINE void noBreak() { outerBreak = false; }
    Vc_ALWAYS_INLINE size_t next() {
        outerBreak = true;
#ifdef VC_GNU_ASM
        const size_t bit = __builtin_ctzl(mask);
        __asm__("btr %1,%0" : "+r"(mask) : "r"(bit));
#else
#ifdef VC_MSVC
#pragma warning(suppress : 4267) // conversion from 'size_t' to 'unsigned long', possible loss of data
#endif
        const size_t bit = _bit_scan_forward(mask);
        mask &= ~(1 << bit);
#endif
        return bit;
    }
};

#define Vc_foreach_bit(_it_, _mask_) \
    for (Vc::AVX::ForeachHelper Vc__make_unique(foreach_bit_obj)((_mask_).toInt()); Vc__make_unique(foreach_bit_obj).outer(); ) \
        for (_it_ = Vc__make_unique(foreach_bit_obj).next(); Vc__make_unique(foreach_bit_obj).inner(); Vc__make_unique(foreach_bit_obj).noBreak())

// Operators
namespace Intrinsics
{
    static Vc_ALWAYS_INLINE Vc_PURE m256 and_(param256 a, param256 b) { return _mm256_and_ps(a, b); }
    static Vc_ALWAYS_INLINE Vc_PURE m256  or_(param256 a, param256 b) { return _mm256_or_ps(a, b); }
    static Vc_ALWAYS_INLINE Vc_PURE m256 xor_(param256 a, param256 b) { return _mm256_xor_ps(a, b); }

    static Vc_ALWAYS_INLINE Vc_PURE m128 and_(param128 a, param128 b) { return _mm_and_ps(a, b); }
    static Vc_ALWAYS_INLINE Vc_PURE m128  or_(param128 a, param128 b) { return _mm_or_ps(a, b); }
    static Vc_ALWAYS_INLINE Vc_PURE m128 xor_(param128 a, param128 b) { return _mm_xor_ps(a, b); }
} // namespace Intrinsics

// binary and/or/xor cannot work with one operand larger than the other
template<unsigned int LSize, unsigned int RSize, size_t LWidth, size_t RWidth> void operator&(const Mask<LSize, LWidth> &l, const Mask<RSize, RWidth> &r);
template<unsigned int LSize, unsigned int RSize, size_t LWidth, size_t RWidth> void operator|(const Mask<LSize, LWidth> &l, const Mask<RSize, RWidth> &r);
template<unsigned int LSize, unsigned int RSize, size_t LWidth, size_t RWidth> void operator^(const Mask<LSize, LWidth> &l, const Mask<RSize, RWidth> &r);

// let binary and/or/xor work for any combination of masks (as long as they have the same sizeof)
template<unsigned int LSize, unsigned int RSize, size_t Width> Vc_ALWAYS_INLINE Vc_PURE Mask<LSize, Width> operator&(const Mask<LSize, Width> &l, const Mask<RSize, Width> &r) { return Intrinsics::and_(l.data(), r.data()); }
template<unsigned int LSize, unsigned int RSize, size_t Width> Vc_ALWAYS_INLINE Vc_PURE Mask<LSize, Width> operator|(const Mask<LSize, Width> &l, const Mask<RSize, Width> &r) { return Intrinsics:: or_(l.data(), r.data()); }
template<unsigned int LSize, unsigned int RSize, size_t Width> Vc_ALWAYS_INLINE Vc_PURE Mask<LSize, Width> operator^(const Mask<LSize, Width> &l, const Mask<RSize, Width> &r) { return Intrinsics::xor_(l.data(), r.data()); }

// disable logical and/or for incompatible masks
template<unsigned int LSize, unsigned int RSize, size_t LWidth, size_t RWidth> void operator&&(const Mask<LSize, LWidth> &lhs, const Mask<RSize, RWidth> &rhs);
template<unsigned int LSize, unsigned int RSize, size_t LWidth, size_t RWidth> void operator||(const Mask<LSize, LWidth> &lhs, const Mask<RSize, RWidth> &rhs);

// logical and/or for compatible masks
template<unsigned int Size, size_t LWidth, size_t RWidth> Vc_ALWAYS_INLINE Vc_PURE Mask<Size, LWidth> operator&&(const Mask<Size, LWidth> &lhs, const Mask<Size, RWidth> &rhs) { return lhs && static_cast<Mask<Size, LWidth> >(rhs); }
template<unsigned int Size, size_t LWidth, size_t RWidth> Vc_ALWAYS_INLINE Vc_PURE Mask<Size, LWidth> operator||(const Mask<Size, LWidth> &lhs, const Mask<Size, RWidth> &rhs) { return lhs || static_cast<Mask<Size, LWidth> >(rhs); }

template<unsigned int Size, size_t Width> Vc_ALWAYS_INLINE Vc_PURE Mask<Size, Width> operator&&(const Mask<Size, Width> &lhs, const Mask<Size, Width> &rhs) { return Intrinsics::and_(lhs.data(), rhs.data()); }
template<unsigned int Size, size_t Width> Vc_ALWAYS_INLINE Vc_PURE Mask<Size, Width> operator||(const Mask<Size, Width> &lhs, const Mask<Size, Width> &rhs) { return Intrinsics::or_ (lhs.data(), rhs.data()); }

} // namespace AVX
} // namespace Vc
} // namespace ROOT

#include "mask.tcc"
#include "undomacros.h"

#endif // VC_AVX_MASK_H
