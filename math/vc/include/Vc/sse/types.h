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

#ifndef SSE_TYPES_H
#define SSE_TYPES_H

#include "intrinsics.h"
#include "../common/storage.h"

#define VC_DOUBLE_V_SIZE 2
#define VC_FLOAT_V_SIZE 4
#define VC_SFLOAT_V_SIZE 8
#define VC_INT_V_SIZE 4
#define VC_UINT_V_SIZE 4
#define VC_SHORT_V_SIZE 8
#define VC_USHORT_V_SIZE 8

#include "../common/types.h"
#include "macros.h"

namespace ROOT {
namespace Vc
{
namespace SSE
{
    template<typename T> class Vector;
    template<typename T> class WriteMaskedVector;

    // define our own long because on Windows64 long == int while on Linux long == max. register width
    // since we want to have a type that depends on 32 vs. 64 bit we need to do some special casing on Windows
#ifdef _WIN64
    typedef __int64 _long;
    typedef unsigned __int64 _ulong;
#else
    typedef long _long;
    typedef unsigned long _ulong;
#endif


    class Float8Mask;
    class Float8GatherMask;
    template<unsigned int VectorSize> class Mask;

    /*
     * Hack to create a vector object with 8 floats
     */
    typedef Vc::sfloat float8;

    class M256 {
        public:
            //Vc_INTRINSIC M256() {}
            //Vc_INTRINSIC M256(_M128 a, _M128 b) { d[0] = a; d[1] = b; }
            static Vc_INTRINSIC Vc_CONST M256 dup(_M128 a) { M256 r; r.d[0] = a; r.d[1] = a; return r; }
            static Vc_INTRINSIC Vc_CONST M256 create(_M128 a, _M128 b) { M256 r; r.d[0] = a; r.d[1] = b; return r; }
            Vc_INTRINSIC _M128 &operator[](int i) { return d[i]; }
            Vc_INTRINSIC const _M128 &operator[](int i) const { return d[i]; }
        private:
#ifdef VC_COMPILE_BENCHMARKS
        public:
#endif
            _M128 d[2];
    };
#ifdef VC_CHECK_ALIGNMENT
static Vc_ALWAYS_INLINE void assertCorrectAlignment(const M256 *ptr)
{
    const size_t s = sizeof(__m128);
    if((reinterpret_cast<size_t>(ptr) & ((s ^ (s & (s - 1))) - 1)) != 0) {
        fprintf(stderr, "A vector with incorrect alignment has just been created. Look at the stacktrace to find the guilty object.\n");
        abort();
    }
}
#endif

    template<typename T> struct ParameterHelper {
        typedef T ByValue;
        typedef T & Reference;
        typedef const T & ConstRef;
    };
#if defined VC_MSVC && !defined _WIN64
    // The calling convention on WIN32 can't guarantee alignment.
    // An exception are the first three arguments, which may be passed in a register.
    template<> struct ParameterHelper<M256> {
        typedef const M256 & ByValue;
        typedef M256 & Reference;
        typedef const M256 & ConstRef;
    };
#endif

    template<typename T> struct VectorHelper {};

    template<unsigned int Size> struct IndexTypeHelper;
    template<> struct IndexTypeHelper<2u> { typedef unsigned int   Type; };
    template<> struct IndexTypeHelper<4u> { typedef unsigned int   Type; };
    template<> struct IndexTypeHelper<8u> { typedef unsigned short Type; };
    template<> struct IndexTypeHelper<16u>{ typedef unsigned char  Type; };

    template<typename T> struct CtorTypeHelper { typedef T Type; };
    template<> struct CtorTypeHelper<short> { typedef int Type; };
    template<> struct CtorTypeHelper<unsigned short> { typedef unsigned int Type; };
    template<> struct CtorTypeHelper<float> { typedef double Type; };

    template<typename T> struct ExpandTypeHelper { typedef T Type; };
    template<> struct ExpandTypeHelper<short> { typedef int Type; };
    template<> struct ExpandTypeHelper<unsigned short> { typedef unsigned int Type; };
    template<> struct ExpandTypeHelper<float> { typedef double Type; };

    template<typename T> struct VectorTypeHelper { typedef __m128i Type; };
    template<> struct VectorTypeHelper<double>   { typedef __m128d Type; };
    template<> struct VectorTypeHelper< float>   { typedef __m128  Type; };
    template<> struct VectorTypeHelper<sfloat>   { typedef   M256  Type; };

    template<typename T, unsigned int Size> struct DetermineMask { typedef Mask<Size> Type; };
    template<> struct DetermineMask<sfloat, 8> { typedef Float8Mask Type; };

    template<typename T> struct DetermineGatherMask { typedef T Type; };
    template<> struct DetermineGatherMask<Float8Mask> { typedef Float8GatherMask Type; };

    template<typename T> struct VectorTraits
    {
        typedef typename VectorTypeHelper<T>::Type VectorType;
        typedef typename DetermineEntryType<T>::Type EntryType;
        enum Constants {
            Size = sizeof(VectorType) / sizeof(EntryType),
            HasVectorDivision = !IsInteger<T>::Value
        };
        typedef typename DetermineMask<T, Size>::Type MaskType;
        typedef typename DetermineGatherMask<MaskType>::Type GatherMaskType;
        typedef Vector<typename IndexTypeHelper<Size>::Type> IndexType;
        typedef Common::VectorMemoryUnion<VectorType, EntryType> StorageType;
    };

    template<typename T> struct VectorHelperSize;

    template<typename V = Vector<float> >
    class STRUCT_ALIGN1(16) VectorAlignedBaseT
    {
        public:
            FREE_STORE_OPERATORS_ALIGNED(16)
    } STRUCT_ALIGN2(16);

} // namespace SSE
} // namespace Vc
} // namespace ROOT

#include "undomacros.h"

#endif // SSE_TYPES_H
