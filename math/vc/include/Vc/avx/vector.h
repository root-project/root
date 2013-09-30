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

#ifndef AVX_VECTOR_H
#define AVX_VECTOR_H

#include "intrinsics.h"
#include "vectorhelper.h"
#include "mask.h"
#include "writemaskedvector.h"
#include "sorthelper.h"
#include <algorithm>
#include <cmath>
#include "../common/aliasingentryhelper.h"
#include "../common/memoryfwd.h"
#include "macros.h"

#ifdef isfinite
#undef isfinite
#endif
#ifdef isnan
#undef isnan
#endif

namespace ROOT {
namespace Vc
{
namespace AVX
{
enum VectorAlignmentEnum { VectorAlignment = 32 };

template<typename T> class Vector
{
    public:
        FREE_STORE_OPERATORS_ALIGNED(32)

        typedef typename VectorTypeHelper<T>::Type VectorType;
        typedef typename DetermineEntryType<T>::Type EntryType;
        enum Constants {
            Size = sizeof(VectorType) / sizeof(EntryType),
            HasVectorDivision = HasVectorDivisionHelper<T>::Value
        };
        typedef Vector<typename IndexTypeHelper<T>::Type> IndexType;
        typedef typename Vc::AVX::Mask<Size, sizeof(VectorType)> Mask;
        typedef typename Mask::AsArg MaskArg;
        typedef Vc::Memory<Vector<T>, Size> Memory;
#ifdef VC_PASSING_VECTOR_BY_VALUE_IS_BROKEN
        typedef const Vector<T> &AsArg;
        typedef const VectorType &VectorTypeArg;
#else
        typedef Vector<T> AsArg;
        typedef VectorType VectorTypeArg;
#endif

    protected:
        // helper that specializes on VectorType
        typedef VectorHelper<VectorType> HV;

        // helper that specializes on T
        typedef VectorHelper<T> HT;

        // cast any m256/m128 to VectorType
        static Vc_INTRINSIC VectorType _cast(param128  v) { return avx_cast<VectorType>(v); }
        static Vc_INTRINSIC VectorType _cast(param128i v) { return avx_cast<VectorType>(v); }
        static Vc_INTRINSIC VectorType _cast(param128d v) { return avx_cast<VectorType>(v); }
        static Vc_INTRINSIC VectorType _cast(param256  v) { return avx_cast<VectorType>(v); }
        static Vc_INTRINSIC VectorType _cast(param256i v) { return avx_cast<VectorType>(v); }
        static Vc_INTRINSIC VectorType _cast(param256d v) { return avx_cast<VectorType>(v); }

#ifdef VC_UNCONDITIONAL_AVX2_INTRINSICS
        typedef Common::VectorMemoryUnion<VectorType, EntryType, typename VectorType::Base> StorageType;
#else
        typedef Common::VectorMemoryUnion<VectorType, EntryType> StorageType;
#endif
        StorageType d;

    public:
        ///////////////////////////////////////////////////////////////////////////////////////////
        // uninitialized
        Vc_ALWAYS_INLINE Vector() {}

        ///////////////////////////////////////////////////////////////////////////////////////////
        // constants
        explicit Vc_ALWAYS_INLINE_L Vector(VectorSpecialInitializerZero::ZEnum) Vc_ALWAYS_INLINE_R;
        explicit Vc_ALWAYS_INLINE_L Vector(VectorSpecialInitializerOne::OEnum) Vc_ALWAYS_INLINE_R;
        explicit Vc_ALWAYS_INLINE_L Vector(VectorSpecialInitializerIndexesFromZero::IEnum) Vc_ALWAYS_INLINE_R;
        static Vc_INTRINSIC_L Vc_CONST_L Vector Zero() Vc_INTRINSIC_R Vc_CONST_R;
        static Vc_INTRINSIC_L Vc_CONST_L Vector One() Vc_INTRINSIC_R Vc_CONST_R;
        static Vc_INTRINSIC_L Vc_CONST_L Vector IndexesFromZero() Vc_INTRINSIC_R Vc_CONST_R;
        static Vc_ALWAYS_INLINE_L Vector Random() Vc_ALWAYS_INLINE_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // internal: required to enable returning objects of VectorType
        Vc_ALWAYS_INLINE Vector(VectorTypeArg x) : d(x) {}
#ifdef VC_UNCONDITIONAL_AVX2_INTRINSICS
        Vc_ALWAYS_INLINE Vector(typename VectorType::Base x) : d(x) {}
#endif

        ///////////////////////////////////////////////////////////////////////////////////////////
        // static_cast / copy ctor
        template<typename T2> explicit Vector(VC_ALIGNED_PARAMETER(Vector<T2>) x);

        // implicit cast
        template<typename OtherT> Vc_INTRINSIC_L Vector &operator=(const Vector<OtherT> &x) Vc_INTRINSIC_R;

        // copy assignment
        Vc_ALWAYS_INLINE Vector &operator=(AsArg v) { d.v() = v.d.v(); return *this; }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // broadcast
        explicit Vc_ALWAYS_INLINE_L Vector(EntryType a) Vc_ALWAYS_INLINE_R;
        template<typename TT> Vc_INTRINSIC Vector(TT x, VC_EXACT_TYPE(TT, EntryType, void *) = 0) : d(HT::set(x)) {}
        Vc_ALWAYS_INLINE Vector &operator=(EntryType a) { d.v() = HT::set(a); return *this; }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // load ctors
        explicit Vc_INTRINSIC_L
            Vector(const EntryType *x) Vc_INTRINSIC_R;
        template<typename Alignment> Vc_INTRINSIC_L
            Vector(const EntryType *x, Alignment align) Vc_INTRINSIC_R;
        template<typename OtherT> explicit Vc_INTRINSIC_L
            Vector(const OtherT    *x) Vc_INTRINSIC_R;
        template<typename OtherT, typename Alignment> Vc_INTRINSIC_L
            Vector(const OtherT    *x, Alignment align) Vc_INTRINSIC_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // load member functions
        Vc_INTRINSIC_L
            void load(const EntryType *mem) Vc_INTRINSIC_R;
        template<typename Alignment> Vc_INTRINSIC_L
            void load(const EntryType *mem, Alignment align) Vc_INTRINSIC_R;
        template<typename OtherT> Vc_INTRINSIC_L
            void load(const OtherT    *mem) Vc_INTRINSIC_R;
        template<typename OtherT, typename Alignment> Vc_INTRINSIC_L
            void load(const OtherT    *mem, Alignment align) Vc_INTRINSIC_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // expand/merge 1 float_v <=> 2 double_v          XXX rationale? remove it for release? XXX
        explicit inline Vector(const Vector<typename HT::ConcatType> *a);
        inline void expand(Vector<typename HT::ConcatType> *x) const;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // zeroing
        Vc_INTRINSIC_L void setZero() Vc_INTRINSIC_R;
        Vc_INTRINSIC_L void setZero(const Mask &k) Vc_INTRINSIC_R;

        Vc_INTRINSIC_L void setQnan() Vc_INTRINSIC_R;
        Vc_INTRINSIC_L void setQnan(MaskArg k) Vc_INTRINSIC_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // stores
        Vc_INTRINSIC_L void store(EntryType *mem) const Vc_INTRINSIC_R;
        Vc_INTRINSIC_L void store(EntryType *mem, const Mask &mask) const Vc_INTRINSIC_R;
        template<typename A> Vc_INTRINSIC_L void store(EntryType *mem, A align) const Vc_INTRINSIC_R;
        template<typename A> Vc_INTRINSIC_L void store(EntryType *mem, const Mask &mask, A align) const Vc_INTRINSIC_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // swizzles
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T> &abcd() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  cdab() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  badc() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  aaaa() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  bbbb() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  cccc() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  dddd() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  bcad() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  bcda() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  dabc() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  acbd() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  dbca() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  dcba() const Vc_INTRINSIC_R Vc_PURE_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // gathers
        template<typename IndexT> Vector(const EntryType *mem, const IndexT *indexes);
        template<typename IndexT> Vector(const EntryType *mem, VC_ALIGNED_PARAMETER(Vector<IndexT>) indexes);
        template<typename IndexT> Vector(const EntryType *mem, const IndexT *indexes, MaskArg mask);
        template<typename IndexT> Vector(const EntryType *mem, VC_ALIGNED_PARAMETER(Vector<IndexT>) indexes, MaskArg mask);
        template<typename S1, typename IT> Vector(const S1 *array, const EntryType S1::* member1, VC_ALIGNED_PARAMETER(IT) indexes);
        template<typename S1, typename IT> Vector(const S1 *array, const EntryType S1::* member1, VC_ALIGNED_PARAMETER(IT) indexes, MaskArg mask);
        template<typename S1, typename S2, typename IT> Vector(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, VC_ALIGNED_PARAMETER(IT) indexes);
        template<typename S1, typename S2, typename IT> Vector(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, VC_ALIGNED_PARAMETER(IT) indexes, MaskArg mask);
        template<typename S1, typename IT1, typename IT2> Vector(const S1 *array, const EntryType *const S1::* ptrMember1, VC_ALIGNED_PARAMETER(IT1) outerIndexes, VC_ALIGNED_PARAMETER(IT2) innerIndexes);
        template<typename S1, typename IT1, typename IT2> Vector(const S1 *array, const EntryType *const S1::* ptrMember1, VC_ALIGNED_PARAMETER(IT1) outerIndexes, VC_ALIGNED_PARAMETER(IT2) innerIndexes, MaskArg mask);
        template<typename Index> void gather(const EntryType *mem, VC_ALIGNED_PARAMETER(Index) indexes);
        template<typename Index> void gather(const EntryType *mem, VC_ALIGNED_PARAMETER(Index) indexes, MaskArg mask);
#ifdef VC_USE_SET_GATHERS
        template<typename IT> void gather(const EntryType *mem, VC_ALIGNED_PARAMETER(Vector<IT>) indexes, MaskArg mask);
#endif
        template<typename S1, typename IT> void gather(const S1 *array, const EntryType S1::* member1, VC_ALIGNED_PARAMETER(IT) indexes);
        template<typename S1, typename IT> void gather(const S1 *array, const EntryType S1::* member1, VC_ALIGNED_PARAMETER(IT) indexes, MaskArg mask);
        template<typename S1, typename S2, typename IT> void gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, VC_ALIGNED_PARAMETER(IT) indexes);
        template<typename S1, typename S2, typename IT> void gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, VC_ALIGNED_PARAMETER(IT) indexes, MaskArg mask);
        template<typename S1, typename IT1, typename IT2> void gather(const S1 *array, const EntryType *const S1::* ptrMember1, VC_ALIGNED_PARAMETER(IT1) outerIndexes, VC_ALIGNED_PARAMETER(IT2) innerIndexes);
        template<typename S1, typename IT1, typename IT2> void gather(const S1 *array, const EntryType *const S1::* ptrMember1, VC_ALIGNED_PARAMETER(IT1) outerIndexes, VC_ALIGNED_PARAMETER(IT2) innerIndexes, MaskArg mask);

        ///////////////////////////////////////////////////////////////////////////////////////////
        // scatters
        template<typename Index> void scatter(EntryType *mem, VC_ALIGNED_PARAMETER(Index) indexes) const;
        template<typename Index> void scatter(EntryType *mem, VC_ALIGNED_PARAMETER(Index) indexes, MaskArg mask) const;
        template<typename S1, typename IT> void scatter(S1 *array, EntryType S1::* member1, VC_ALIGNED_PARAMETER(IT) indexes) const;
        template<typename S1, typename IT> void scatter(S1 *array, EntryType S1::* member1, VC_ALIGNED_PARAMETER(IT) indexes, MaskArg mask) const;
        template<typename S1, typename S2, typename IT> void scatter(S1 *array, S2 S1::* member1, EntryType S2::* member2, VC_ALIGNED_PARAMETER(IT) indexes) const;
        template<typename S1, typename S2, typename IT> void scatter(S1 *array, S2 S1::* member1, EntryType S2::* member2, VC_ALIGNED_PARAMETER(IT) indexes, MaskArg mask) const;
        template<typename S1, typename IT1, typename IT2> void scatter(S1 *array, EntryType *S1::* ptrMember1, VC_ALIGNED_PARAMETER(IT1) outerIndexes, VC_ALIGNED_PARAMETER(IT2) innerIndexes) const;
        template<typename S1, typename IT1, typename IT2> void scatter(S1 *array, EntryType *S1::* ptrMember1, VC_ALIGNED_PARAMETER(IT1) outerIndexes, VC_ALIGNED_PARAMETER(IT2) innerIndexes, MaskArg mask) const;

        ///////////////////////////////////////////////////////////////////////////////////////////
        //prefix
        Vc_ALWAYS_INLINE Vector &operator++() { data() = VectorHelper<T>::add(data(), VectorHelper<T>::one()); return *this; }
        Vc_ALWAYS_INLINE Vector &operator--() { data() = VectorHelper<T>::sub(data(), VectorHelper<T>::one()); return *this; }
        //postfix
        Vc_ALWAYS_INLINE Vector operator++(int) { const Vector<T> r = *this; data() = VectorHelper<T>::add(data(), VectorHelper<T>::one()); return r; }
        Vc_ALWAYS_INLINE Vector operator--(int) { const Vector<T> r = *this; data() = VectorHelper<T>::sub(data(), VectorHelper<T>::one()); return r; }

        Vc_INTRINSIC Common::AliasingEntryHelper<StorageType> operator[](size_t index) {
#if defined(VC_GCC) && VC_GCC >= 0x40300 && VC_GCC < 0x40400
            ::ROOT::Vc::Warnings::_operator_bracket_warning();
#endif
            return d.m(index);
        }
        Vc_ALWAYS_INLINE EntryType operator[](size_t index) const {
            return d.m(index);
        }

        Vc_ALWAYS_INLINE Vector operator~() const { return VectorHelper<VectorType>::andnot_(data(), VectorHelper<VectorType>::allone()); }
        Vc_ALWAYS_INLINE_L Vc_PURE_L Vector<typename NegateTypeHelper<T>::Type> operator-() const Vc_ALWAYS_INLINE_R Vc_PURE_R;
        Vc_INTRINSIC Vc_PURE Vector operator+() const { return *this; }

#define OP1(fun) \
        Vc_ALWAYS_INLINE Vector fun() const { return Vector<T>(VectorHelper<T>::fun(data())); } \
        Vc_ALWAYS_INLINE Vector &fun##_eq() { data() = VectorHelper<T>::fun(data()); return *this; }
        OP1(sqrt)
        OP1(abs)
#undef OP1

#define OP(symbol, fun) \
        Vc_ALWAYS_INLINE Vector &operator symbol##=(const Vector<T> &x) { data() = VectorHelper<T>::fun(data(), x.data()); return *this; } \
        Vc_ALWAYS_INLINE Vector &operator symbol##=(EntryType x) { return operator symbol##=(Vector(x)); } \
        Vc_ALWAYS_INLINE Vector operator symbol(const Vector<T> &x) const { return Vector<T>(VectorHelper<T>::fun(data(), x.data())); } \
        template<typename TT> Vc_ALWAYS_INLINE VC_EXACT_TYPE(TT, EntryType, Vector) operator symbol(TT x) const { return operator symbol(Vector(x)); }

        OP(+, add)
        OP(-, sub)
        OP(*, mul)
#undef OP
        inline Vector &operator/=(EntryType x);
        template<typename TT> inline Vc_PURE_L VC_EXACT_TYPE(TT, EntryType, Vector) operator/(TT x) const Vc_PURE_R;
        inline Vector &operator/=(const Vector<T> &x);
        inline Vc_PURE_L Vector  operator/ (const Vector<T> &x) const Vc_PURE_R;

        // bitwise ops
#define OP_VEC(op) \
        Vc_ALWAYS_INLINE_L Vector<T> &operator op##=(AsArg x) Vc_ALWAYS_INLINE_R; \
        Vc_ALWAYS_INLINE_L Vc_PURE_L Vector<T>  operator op   (AsArg x) const Vc_ALWAYS_INLINE_R Vc_PURE_R;
#define OP_ENTRY(op) \
        Vc_ALWAYS_INLINE Vector<T> &operator op##=(EntryType x) { return operator op##=(Vector(x)); } \
        template<typename TT> Vc_ALWAYS_INLINE Vc_PURE VC_EXACT_TYPE(TT, EntryType, Vector) operator op(TT x) const { return operator op(Vector(x)); }
        VC_ALL_BINARY(OP_VEC)
        VC_ALL_BINARY(OP_ENTRY)
        VC_ALL_SHIFTS(OP_VEC)
#undef OP_VEC
#undef OP_ENTRY

        Vc_ALWAYS_INLINE_L Vector<T> &operator>>=(int x) Vc_ALWAYS_INLINE_R;
        Vc_ALWAYS_INLINE_L Vector<T> &operator<<=(int x) Vc_ALWAYS_INLINE_R;
        Vc_ALWAYS_INLINE_L Vector<T> operator>>(int x) const Vc_ALWAYS_INLINE_R;
        Vc_ALWAYS_INLINE_L Vector<T> operator<<(int x) const Vc_ALWAYS_INLINE_R;

#define OPcmp(symbol, fun) \
        Vc_ALWAYS_INLINE Mask operator symbol(AsArg x) const { return VectorHelper<T>::fun(data(), x.data()); } \
        template<typename TT> Vc_ALWAYS_INLINE VC_EXACT_TYPE(TT, EntryType, Mask) operator symbol(TT x) const { return operator symbol(Vector(x)); }

        OPcmp(==, cmpeq)
        OPcmp(!=, cmpneq)
        OPcmp(>=, cmpnlt)
        OPcmp(>, cmpnle)
        OPcmp(<, cmplt)
        OPcmp(<=, cmple)
#undef OPcmp
        Vc_INTRINSIC_L Vc_PURE_L Mask isNegative() const Vc_PURE_R Vc_INTRINSIC_R;

        Vc_ALWAYS_INLINE void fusedMultiplyAdd(const Vector<T> &factor, const Vector<T> &summand) {
            VectorHelper<T>::fma(data(), factor.data(), summand.data());
        }

        Vc_ALWAYS_INLINE void assign( const Vector<T> &v, const Mask &mask ) {
            const VectorType k = avx_cast<VectorType>(mask.data());
            data() = VectorHelper<VectorType>::blend(data(), v.data(), k);
        }

        template<typename V2> Vc_ALWAYS_INLINE V2 staticCast() const { return V2(*this); }
        template<typename V2> Vc_ALWAYS_INLINE V2 reinterpretCast() const { return avx_cast<typename V2::VectorType>(data()); }

        Vc_ALWAYS_INLINE WriteMaskedVector<T> operator()(const Mask &k) { return WriteMaskedVector<T>(this, k); }

        /**
         * \return \p true  This vector was completely filled. m2 might be 0 or != 0. You still have
         *                  to test this.
         *         \p false This vector was not completely filled. m2 is all 0.
         */
        //inline bool pack(Mask &m1, Vector<T> &v2, Mask &m2) {
            //return VectorHelper<T>::pack(data(), m1.data, v2.data(), m2.data);
        //}

        Vc_ALWAYS_INLINE VectorType &data() { return d.v(); }
        Vc_ALWAYS_INLINE const VectorType data() const { return d.v(); }

        Vc_ALWAYS_INLINE EntryType min() const { return VectorHelper<T>::min(data()); }
        Vc_ALWAYS_INLINE EntryType max() const { return VectorHelper<T>::max(data()); }
        Vc_ALWAYS_INLINE EntryType product() const { return VectorHelper<T>::mul(data()); }
        Vc_ALWAYS_INLINE EntryType sum() const { return VectorHelper<T>::add(data()); }
        Vc_ALWAYS_INLINE_L EntryType min(MaskArg m) const Vc_ALWAYS_INLINE_R;
        Vc_ALWAYS_INLINE_L EntryType max(MaskArg m) const Vc_ALWAYS_INLINE_R;
        Vc_ALWAYS_INLINE_L EntryType product(MaskArg m) const Vc_ALWAYS_INLINE_R;
        Vc_ALWAYS_INLINE_L EntryType sum(MaskArg m) const Vc_ALWAYS_INLINE_R;

        Vc_INTRINSIC_L Vector shifted(int amount) const Vc_INTRINSIC_R;
        Vc_INTRINSIC_L Vector rotated(int amount) const Vc_INTRINSIC_R;
        Vc_ALWAYS_INLINE Vector sorted() const { return SortHelper<T>::sort(data()); }

        template<typename F> void callWithValuesSorted(F &f) {
            EntryType value = d.m(0);
            f(value);
            for (int i = 1; i < Size; ++i) {
                if (d.m(i) != value) {
                    value = d.m(i);
                    f(value);
                }
            }
        }

        template<typename F> Vc_INTRINSIC void call(const F &f) const {
            for_all_vector_entries(i,
                    f(EntryType(d.m(i)));
                    );
        }
        template<typename F> Vc_INTRINSIC void call(F &f) const {
            for_all_vector_entries(i,
                    f(EntryType(d.m(i)));
                    );
        }

        template<typename F> Vc_INTRINSIC void call(const F &f, const Mask &mask) const {
            Vc_foreach_bit(size_t i, mask) {
                f(EntryType(d.m(i)));
            }
        }
        template<typename F> Vc_INTRINSIC void call(F &f, const Mask &mask) const {
            Vc_foreach_bit(size_t i, mask) {
                f(EntryType(d.m(i)));
            }
        }

        template<typename F> Vc_INTRINSIC Vector<T> apply(const F &f) const {
            Vector<T> r;
            for_all_vector_entries(i,
                    r.d.m(i) = f(EntryType(d.m(i)));
                    );
            return r;
        }
        template<typename F> Vc_INTRINSIC Vector<T> apply(F &f) const {
            Vector<T> r;
            for_all_vector_entries(i,
                    r.d.m(i) = f(EntryType(d.m(i)));
                    );
            return r;
        }

        template<typename F> Vc_INTRINSIC Vector<T> apply(const F &f, const Mask &mask) const {
            Vector<T> r(*this);
            Vc_foreach_bit (size_t i, mask) {
                r.d.m(i) = f(EntryType(r.d.m(i)));
            }
            return r;
        }
        template<typename F> Vc_INTRINSIC Vector<T> apply(F &f, const Mask &mask) const {
            Vector<T> r(*this);
            Vc_foreach_bit (size_t i, mask) {
                r.d.m(i) = f(EntryType(r.d.m(i)));
            }
            return r;
        }

        template<typename IndexT> Vc_INTRINSIC void fill(EntryType (&f)(IndexT)) {
            for_all_vector_entries(i,
                    d.m(i) = f(i);
                    );
        }
        Vc_INTRINSIC void fill(EntryType (&f)()) {
            for_all_vector_entries(i,
                    d.m(i) = f();
                    );
        }

        Vc_INTRINSIC_L Vector copySign(AsArg reference) const Vc_INTRINSIC_R;
        Vc_INTRINSIC_L Vector exponent() const Vc_INTRINSIC_R;
};

typedef Vector<double>         double_v;
typedef Vector<float>          float_v;
typedef Vector<sfloat>         sfloat_v;
typedef Vector<int>            int_v;
typedef Vector<unsigned int>   uint_v;
typedef Vector<short>          short_v;
typedef Vector<unsigned short> ushort_v;
typedef double_v::Mask double_m;
typedef  float_v::Mask float_m;
typedef sfloat_v::Mask sfloat_m;
typedef    int_v::Mask int_m;
typedef   uint_v::Mask uint_m;
typedef  short_v::Mask short_m;
typedef ushort_v::Mask ushort_m;

template<typename T> class SwizzledVector : public Vector<T> {};

static Vc_ALWAYS_INLINE int_v    min(const int_v    &x, const int_v    &y) { return _mm256_min_epi32(x.data(), y.data()); }
static Vc_ALWAYS_INLINE uint_v   min(const uint_v   &x, const uint_v   &y) { return _mm256_min_epu32(x.data(), y.data()); }
static Vc_ALWAYS_INLINE short_v  min(const short_v  &x, const short_v  &y) { return _mm_min_epi16(x.data(), y.data()); }
static Vc_ALWAYS_INLINE ushort_v min(const ushort_v &x, const ushort_v &y) { return _mm_min_epu16(x.data(), y.data()); }
static Vc_ALWAYS_INLINE float_v  min(const float_v  &x, const float_v  &y) { return _mm256_min_ps(x.data(), y.data()); }
static Vc_ALWAYS_INLINE sfloat_v min(const sfloat_v &x, const sfloat_v &y) { return _mm256_min_ps(x.data(), y.data()); }
static Vc_ALWAYS_INLINE double_v min(const double_v &x, const double_v &y) { return _mm256_min_pd(x.data(), y.data()); }
static Vc_ALWAYS_INLINE int_v    max(const int_v    &x, const int_v    &y) { return _mm256_max_epi32(x.data(), y.data()); }
static Vc_ALWAYS_INLINE uint_v   max(const uint_v   &x, const uint_v   &y) { return _mm256_max_epu32(x.data(), y.data()); }
static Vc_ALWAYS_INLINE short_v  max(const short_v  &x, const short_v  &y) { return _mm_max_epi16(x.data(), y.data()); }
static Vc_ALWAYS_INLINE ushort_v max(const ushort_v &x, const ushort_v &y) { return _mm_max_epu16(x.data(), y.data()); }
static Vc_ALWAYS_INLINE float_v  max(const float_v  &x, const float_v  &y) { return _mm256_max_ps(x.data(), y.data()); }
static Vc_ALWAYS_INLINE sfloat_v max(const sfloat_v &x, const sfloat_v &y) { return _mm256_max_ps(x.data(), y.data()); }
static Vc_ALWAYS_INLINE double_v max(const double_v &x, const double_v &y) { return _mm256_max_pd(x.data(), y.data()); }

  template<typename T> static Vc_ALWAYS_INLINE Vector<T> sqrt (const Vector<T> &x) { return VectorHelper<T>::sqrt(x.data()); }
  template<typename T> static Vc_ALWAYS_INLINE Vector<T> rsqrt(const Vector<T> &x) { return VectorHelper<T>::rsqrt(x.data()); }
  template<typename T> static Vc_ALWAYS_INLINE Vector<T> abs  (const Vector<T> &x) { return VectorHelper<T>::abs(x.data()); }
  template<typename T> static Vc_ALWAYS_INLINE Vector<T> reciprocal(const Vector<T> &x) { return VectorHelper<T>::reciprocal(x.data()); }
  template<typename T> static Vc_ALWAYS_INLINE Vector<T> round(const Vector<T> &x) { return VectorHelper<T>::round(x.data()); }

  template<typename T> static Vc_ALWAYS_INLINE typename Vector<T>::Mask isfinite(const Vector<T> &x) { return VectorHelper<T>::isFinite(x.data()); }
  template<typename T> static Vc_ALWAYS_INLINE typename Vector<T>::Mask isnan(const Vector<T> &x) { return VectorHelper<T>::isNaN(x.data()); }

#include "forceToRegisters.tcc"
} // namespace AVX
} // namespace Vc
} // namespace ROOT

#include "vector.tcc"
#include "math.h"
#include "undomacros.h"

#endif // AVX_VECTOR_H
