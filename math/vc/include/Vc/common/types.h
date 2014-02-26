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

#ifndef VC_COMMON_TYPES_H
#define VC_COMMON_TYPES_H

#ifdef VC_CHECK_ALIGNMENT
#include <cstdlib>
#include <cstdio>
#endif

namespace ROOT {
namespace Vc
{

// helper type to implement sfloat_v (Vector<Vc::sfloat>)
struct sfloat {};

template<typename T> struct DetermineEntryType { typedef T Type; };
template<> struct DetermineEntryType<sfloat> { typedef float Type; };

template<typename T> struct NegateTypeHelper { typedef T Type; };
template<> struct NegateTypeHelper<unsigned char > { typedef char  Type; };
template<> struct NegateTypeHelper<unsigned short> { typedef short Type; };
template<> struct NegateTypeHelper<unsigned int  > { typedef int   Type; };

namespace VectorSpecialInitializerZero { enum ZEnum { Zero = 0 }; }
namespace VectorSpecialInitializerOne { enum OEnum { One = 1 }; }
namespace VectorSpecialInitializerIndexesFromZero { enum IEnum { IndexesFromZero }; }

template<typename V, size_t Size1, size_t Size2> class Memory;
#ifdef VC_MSVC
#  if defined(VC_IMPL_Scalar)
namespace Scalar {
    template<typename T> class Vector;
    template<unsigned int VectorSize> class Mask;
}
#define _Vector Vc::Scalar::Vector
#  elif defined(VC_IMPL_SSE)
namespace SSE {
    template<typename T> class Vector;
    template<unsigned int VectorSize> class Mask;
    class Float8Mask;
}
#define _Vector Vc::SSE::Vector
#  elif defined(VC_IMPL_AVX)
namespace AVX {
    template<typename T> class Vector;
    template<unsigned int VectorSize, size_t RegisterWidth> class Mask;
}
#define _Vector Vc::AVX::Vector
#  else
#    error "Sorry, MSVC is a nasty compiler and needs extra care. Please help."
#  endif
#endif
namespace
{
    template<bool Test, typename T = void> struct EnableIf { typedef T Value; };
    template<typename T> struct EnableIf<false, T> {};

    template<typename T> struct IsSignedInteger    { enum { Value = 0 }; };
    template<> struct IsSignedInteger<signed char> { enum { Value = 1 }; };
    template<> struct IsSignedInteger<short>       { enum { Value = 1 }; };
    template<> struct IsSignedInteger<int>         { enum { Value = 1 }; };
    template<> struct IsSignedInteger<long>        { enum { Value = 1 }; };
    template<> struct IsSignedInteger<long long>   { enum { Value = 1 }; };

    template<typename T> struct IsUnsignedInteger           { enum { Value = 0 }; };
    template<> struct IsUnsignedInteger<unsigned char>      { enum { Value = 1 }; };
    template<> struct IsUnsignedInteger<unsigned short>     { enum { Value = 1 }; };
    template<> struct IsUnsignedInteger<unsigned int>       { enum { Value = 1 }; };
    template<> struct IsUnsignedInteger<unsigned long>      { enum { Value = 1 }; };
    template<> struct IsUnsignedInteger<unsigned long long> { enum { Value = 1 }; };

    template<typename T> struct IsInteger { enum { Value = IsSignedInteger<T>::Value | IsUnsignedInteger<T>::Value }; };

    template<typename T> struct IsReal { enum { Value = 0 }; };
    template<> struct IsReal<float>    { enum { Value = 1 }; };
    template<> struct IsReal<double>   { enum { Value = 1 }; };

    template<typename T, typename U> struct IsEqualType { enum { Value = 0 }; };
    template<typename T> struct IsEqualType<T, T> { enum { Value = 1 }; };

    template<typename T, typename List0, typename List1 = void, typename List2 = void, typename List3 = void, typename List4 = void, typename List5 = void, typename List6 = void>
        struct IsInTypelist { enum { Value = false }; };
    template<typename T, typename List1, typename List2, typename List3, typename List4, typename List5, typename List6> struct IsInTypelist<T, T, List1, List2, List3, List4, List5, List6> { enum { Value = true }; };
    template<typename T, typename List0, typename List2, typename List3, typename List4, typename List5, typename List6> struct IsInTypelist<T, List0, T, List2, List3, List4, List5, List6> { enum { Value = true }; };
    template<typename T, typename List0, typename List1, typename List3, typename List4, typename List5, typename List6> struct IsInTypelist<T, List0, List1, T, List3, List4, List5, List6> { enum { Value = true }; };
    template<typename T, typename List0, typename List1, typename List2, typename List4, typename List5, typename List6> struct IsInTypelist<T, List0, List1, List2, T, List4, List5, List6> { enum { Value = true }; };
    template<typename T, typename List0, typename List1, typename List2, typename List3, typename List5, typename List6> struct IsInTypelist<T, List0, List1, List2, List3, T, List5, List6> { enum { Value = true }; };
    template<typename T, typename List0, typename List1, typename List2, typename List3, typename List4, typename List6> struct IsInTypelist<T, List0, List1, List2, List3, List4, T, List6> { enum { Value = true }; };
    template<typename T, typename List0, typename List1, typename List2, typename List3, typename List4, typename List5> struct IsInTypelist<T, List0, List1, List2, List3, List4, List5, T> { enum { Value = true }; };

    template<typename Arg0, typename Arg1, typename T0, typename T1> struct IsCombinationOf { enum { Value = false }; };
    template<typename Arg0, typename Arg1> struct IsCombinationOf<Arg0, Arg1, Arg0, Arg1> { enum { Value = true }; };
    template<typename Arg0, typename Arg1> struct IsCombinationOf<Arg0, Arg1, Arg1, Arg0> { enum { Value = true }; };

    namespace
    {
        struct yes { char x; };
        struct  no { yes x, y; };
    } // anonymous namespace

    template<typename From, typename To> struct HasImplicitCast
    {
#ifdef VC_MSVC
        // MSVC can't compile this code if we pass a type that has large alignment restrictions by
        // value
        // clang OTOH warns about this code if we pass a null-reference, thus we ifdef the const-ref
        // for MSVC only
        static yes test(const To &) { return yes(); }
#else
        static yes test( To) { return yes(); }
#endif
        static  no test(...) { return  no(); }
        enum {
#ifdef VC_MSVC
            // I want to test whether implicit cast works. If it works MSVC thinks it should give a warning. Wrong. Shut up.
#pragma warning(suppress : 4257 4267)
#endif
            Value = !!(sizeof(test(*static_cast<From *>(0))) == sizeof(yes))
        };
    };
#if defined(VC_GCC) && VC_GCC < 0x40300
    // GCC 4.1 is very noisy because of the float->int and double->int type trait tests. We get
    // around this noise with a little specialization.
    template<> struct HasImplicitCast<float , int> { enum { Value = true }; };
    template<> struct HasImplicitCast<double, int> { enum { Value = true }; };
#endif

#ifdef VC_MSVC
    // MSVC is such a broken compiler :'(
    // HasImplicitCast breaks if From has an __declspec(align(#)) modifier and has no implicit cast
    // to To.  That's because it'll call test(...) as test(From) and not test(const From &).
    // This results in C2718. And MSVC is too stupid to see that it should just shut up and
    // everybody would be happy.
    //
    // Because the HasImplicitCast specializations can only be implemented after the Vector class
    // was declared we have to write some nasty hacks.
    template<typename T1, typename T2> struct HasImplicitCast<_Vector<T1>, T2> { enum { Value = false }; };
#if defined(VC_IMPL_Scalar)
    template<unsigned int VS, typename T2> struct HasImplicitCast<Vc::Scalar::Mask<VS>, T2> { enum { Value = false }; };
    template<unsigned int VS> struct HasImplicitCast<Vc::Scalar::Mask<VS>, Vc::Scalar::Mask<VS> > { enum { Value = true }; };
#elif defined(VC_IMPL_SSE)
    template<unsigned int VS, typename T2> struct HasImplicitCast<Vc::SSE::Mask<VS>, T2> { enum { Value = false }; };
    template<unsigned int VS> struct HasImplicitCast<Vc::SSE::Mask<VS>, Vc::SSE::Mask<VS> > { enum { Value = true }; };
    template<typename T2> struct HasImplicitCast<Vc::SSE::Float8Mask, T2> { enum { Value = false }; };
    template<> struct HasImplicitCast<Vc::SSE::Float8Mask, Vc::SSE::Float8Mask> { enum { Value = true }; };
#elif defined(VC_IMPL_AVX)
    template<unsigned int VectorSize, size_t RegisterWidth, typename T2> struct HasImplicitCast<Vc::AVX::Mask<VectorSize, RegisterWidth>, T2> { enum { Value = false }; };
    template<unsigned int VectorSize, size_t RegisterWidth> struct HasImplicitCast<Vc::AVX::Mask<VectorSize, RegisterWidth>, Vc::AVX::Mask<VectorSize, RegisterWidth> > { enum { Value = true }; };
#endif
    template<typename T> struct HasImplicitCast<_Vector<T>, _Vector<T> > { enum { Value = true }; };
    //template<> struct HasImplicitCast<_Vector<           int>, _Vector<  unsigned int>> { enum { Value = true }; };
    //template<> struct HasImplicitCast<_Vector<  unsigned int>, _Vector<           int>> { enum { Value = true }; };
    //template<> struct HasImplicitCast<_Vector<         short>, _Vector<unsigned short>> { enum { Value = true }; };
    //template<> struct HasImplicitCast<_Vector<unsigned short>, _Vector<         short>> { enum { Value = true }; };
    template<typename V, size_t Size1, size_t Size2, typename T2> struct HasImplicitCast<Vc::Memory<V, Size1, Size2>, T2> { enum { Value = false }; };
    template<typename V, size_t Size1, size_t Size2> struct HasImplicitCast<Vc::Memory<V, Size1, Size2>, Vc::Memory<V, Size1, Size2> > { enum { Value = true }; };
#undef _Vector
#endif

    template<typename T> struct CanConvertToInt : public HasImplicitCast<T, int> {};
    template<> struct CanConvertToInt<bool>     { enum { Value = 0 }; };
    //template<> struct CanConvertToInt<float>    { enum { Value = 0 }; };
    //template<> struct CanConvertToInt<double>   { enum { Value = 0 }; };

    enum TestEnum {};
    VC_STATIC_ASSERT(CanConvertToInt<int>::Value == 1, CanConvertToInt_is_broken);
    VC_STATIC_ASSERT(CanConvertToInt<unsigned char>::Value == 1, CanConvertToInt_is_broken);
    VC_STATIC_ASSERT(CanConvertToInt<bool>::Value == 0, CanConvertToInt_is_broken);
    VC_STATIC_ASSERT(CanConvertToInt<float>::Value == 1, CanConvertToInt_is_broken);
    VC_STATIC_ASSERT(CanConvertToInt<double>::Value == 1, CanConvertToInt_is_broken);
    VC_STATIC_ASSERT(CanConvertToInt<float*>::Value == 0, CanConvertToInt_is_broken);
    VC_STATIC_ASSERT(CanConvertToInt<TestEnum>::Value == 1, CanConvertToInt_is_broken);

    typedef HasImplicitCast<TestEnum, short> HasImplicitCastTest0;
    typedef HasImplicitCast<int *, void *> HasImplicitCastTest1;
    typedef HasImplicitCast<int *, const void *> HasImplicitCastTest2;
    typedef HasImplicitCast<const int *, const void *> HasImplicitCastTest3;
    typedef HasImplicitCast<const int *, int *> HasImplicitCastTest4;

    VC_STATIC_ASSERT(HasImplicitCastTest0::Value ==  true, HasImplicitCast0_is_broken);
    VC_STATIC_ASSERT(HasImplicitCastTest1::Value ==  true, HasImplicitCast1_is_broken);
    VC_STATIC_ASSERT(HasImplicitCastTest2::Value ==  true, HasImplicitCast2_is_broken);
    VC_STATIC_ASSERT(HasImplicitCastTest3::Value ==  true, HasImplicitCast3_is_broken);
    VC_STATIC_ASSERT(HasImplicitCastTest4::Value == false, HasImplicitCast4_is_broken);

    template<typename T> struct IsLikeInteger { enum { Value = !IsReal<T>::Value && CanConvertToInt<T>::Value }; };
    template<typename T> struct IsLikeSignedInteger { enum { Value = IsLikeInteger<T>::Value && !IsUnsignedInteger<T>::Value }; };
} // anonymous namespace

#ifndef VC_CHECK_ALIGNMENT
template<typename _T> static Vc_ALWAYS_INLINE void assertCorrectAlignment(const _T *){}
#else
template<typename _T> static Vc_ALWAYS_INLINE void assertCorrectAlignment(const _T *ptr)
{
    const size_t s = Vc_ALIGNOF(_T);
    if((reinterpret_cast<size_t>(ptr) & ((s ^ (s & (s - 1))) - 1)) != 0) {
        fprintf(stderr, "A vector with incorrect alignment has just been created. Look at the stacktrace to find the guilty object.\n");
        abort();
    }
}
#endif

} // namespace Vc
} // namespace ROOT

#endif // VC_COMMON_TYPES_H
