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

#ifndef AVX_TYPES_H
#define AVX_TYPES_H

#include "intrinsics.h"
#include "../common/storage.h"
#include "macros.h"

#define VC_DOUBLE_V_SIZE 4
#define VC_FLOAT_V_SIZE 8
#define VC_SFLOAT_V_SIZE 8
#define VC_INT_V_SIZE 8
#define VC_UINT_V_SIZE 8
#define VC_SHORT_V_SIZE 8
#define VC_USHORT_V_SIZE 8

#include "../common/types.h"

namespace ROOT {
namespace Vc
{
namespace AVX
{
    template<typename T> class Vector;

    template<unsigned int VectorSize, size_t RegisterWidth> class Mask;

    template<typename T> struct VectorHelper {};
    template<typename T> struct GatherHelper;
    template<typename T> struct ScatterHelper;

    template<typename T> struct IndexTypeHelper;
    template<> struct IndexTypeHelper<         char > { typedef unsigned char  Type; };
    template<> struct IndexTypeHelper<unsigned char > { typedef unsigned char  Type; };
    template<> struct IndexTypeHelper<         short> { typedef unsigned short Type; };
    template<> struct IndexTypeHelper<unsigned short> { typedef unsigned short Type; };
    template<> struct IndexTypeHelper<         int  > { typedef unsigned int   Type; };
    template<> struct IndexTypeHelper<unsigned int  > { typedef unsigned int   Type; };
    template<> struct IndexTypeHelper<         float> { typedef unsigned int   Type; };
    template<> struct IndexTypeHelper<        sfloat> { typedef unsigned short Type; };
    template<> struct IndexTypeHelper<        double> { typedef unsigned int   Type; }; // _M128I based int32 would be nice

    template<typename T> struct VectorTypeHelper;
    template<> struct VectorTypeHelper<         char > { typedef m128i Type; };
    template<> struct VectorTypeHelper<unsigned char > { typedef m128i Type; };
    template<> struct VectorTypeHelper<         short> { typedef m128i Type; };
    template<> struct VectorTypeHelper<unsigned short> { typedef m128i Type; };
    template<> struct VectorTypeHelper<         int  > { typedef m256i Type; };
    template<> struct VectorTypeHelper<unsigned int  > { typedef m256i Type; };
    template<> struct VectorTypeHelper<         float> { typedef m256  Type; };
    template<> struct VectorTypeHelper<        sfloat> { typedef m256  Type; };
    template<> struct VectorTypeHelper<        double> { typedef m256d Type; };

    template<typename T> struct SseVectorType;
    template<> struct SseVectorType<m256 > { typedef m128  Type; };
    template<> struct SseVectorType<m256i> { typedef m128i Type; };
    template<> struct SseVectorType<m256d> { typedef m128d Type; };
    template<> struct SseVectorType<m128 > { typedef m128  Type; };
    template<> struct SseVectorType<m128i> { typedef m128i Type; };
    template<> struct SseVectorType<m128d> { typedef m128d Type; };

    template<typename T> struct HasVectorDivisionHelper { enum { Value = 1 }; };
    //template<> struct HasVectorDivisionHelper<unsigned int> { enum { Value = 0 }; };

    template<typename T> struct VectorHelperSize;

#ifdef VC_MSVC
    // MSVC's __declspec(align(#)) only works with numbers, no enums or sizeof allowed ;(
    template<size_t size> class _VectorAlignedBaseHack;
    template<> class STRUCT_ALIGN1( 8) _VectorAlignedBaseHack< 8> {} STRUCT_ALIGN2( 8);
    template<> class STRUCT_ALIGN1(16) _VectorAlignedBaseHack<16> {} STRUCT_ALIGN2(16);
    template<> class STRUCT_ALIGN1(32) _VectorAlignedBaseHack<32> {} STRUCT_ALIGN2(32);
    template<> class STRUCT_ALIGN1(64) _VectorAlignedBaseHack<64> {} STRUCT_ALIGN2(64);
    template<typename V = Vector<float> >
    class VectorAlignedBaseT : public _VectorAlignedBaseHack<sizeof(V)>
    {
        public:
            FREE_STORE_OPERATORS_ALIGNED(sizeof(V))
    };
#else
    template<typename V = Vector<float> >
    class STRUCT_ALIGN1(sizeof(V)) VectorAlignedBaseT
    {
        public:
            FREE_STORE_OPERATORS_ALIGNED(sizeof(V))
    } STRUCT_ALIGN2(sizeof(V));
#endif
} // namespace AVX
} // namespace Vc
} // namespace ROOT
#include "undomacros.h"

#endif // AVX_TYPES_H
