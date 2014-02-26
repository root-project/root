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

#ifndef VC_COMMON_VECTORTUPLE_H
#define VC_COMMON_VECTORTUPLE_H

#include "macros.h"

namespace ROOT {
namespace Vc
{
namespace Common
{

template<size_t StructSize, typename V> struct InterleavedMemoryReadAccess;

template<int Length, typename V> struct VectorTuple;
template<typename V> struct VectorTuple<2, V>
{
    typedef typename V::EntryType T;
    typedef V &VC_RESTRICT Reference;
    Reference l, r;

    Vc_ALWAYS_INLINE VectorTuple(Reference a, Reference b)
        : l(a), r(b)
    {
    }

    Vc_ALWAYS_INLINE VectorTuple<3, V> operator,(V &a) const
    {
        return VectorTuple<3, V>(*this, a);
    }

    Vc_ALWAYS_INLINE VectorTuple<3, const V> operator,(const V &a) const
    {
        return VectorTuple<3, const V>(*this, a);
    }

    template<size_t StructSize>
    Vc_ALWAYS_INLINE void operator=(const InterleavedMemoryReadAccess<StructSize, V> &access) const
    {
        VC_STATIC_ASSERT(2 <= StructSize, You_are_trying_to_extract_more_data_from_the_struct_than_it_has);
        access.deinterleave(l, r);
    }
};

template<typename V> struct VectorTuple<2, const V>
{
    typedef typename V::EntryType T;
    typedef const V &VC_RESTRICT Reference;
    Reference l, r;

    Vc_ALWAYS_INLINE VectorTuple(Reference a, Reference b)
        : l(a), r(b)
    {
    }

    Vc_ALWAYS_INLINE VectorTuple<3, const V> operator,(const V &a) const
    {
        return VectorTuple<3, const V>(*this, a);
    }
};

#define _VC_VECTORTUPLE_SPECIALIZATION(LENGTH, parameters) \
template<typename V> struct VectorTuple<LENGTH, V> \
{ \
    typedef typename V::EntryType T; \
    typedef V &VC_RESTRICT Reference; \
    const VectorTuple<LENGTH - 1, V> &l; \
    Reference r; \
 \
    Vc_ALWAYS_INLINE VectorTuple(const VectorTuple<LENGTH - 1, V> &tuple, Reference a) \
        : l(tuple), r(a) \
    { \
    } \
 \
    Vc_ALWAYS_INLINE VectorTuple<LENGTH + 1, V> operator,(V &a) const \
    { \
        return VectorTuple<LENGTH + 1, V>(*this, a); \
    } \
 \
    template<size_t StructSize> \
    Vc_ALWAYS_INLINE void operator=(const InterleavedMemoryReadAccess<StructSize, V> &access) const \
    { \
        VC_STATIC_ASSERT(LENGTH <= StructSize, You_are_trying_to_extract_more_data_from_the_struct_than_it_has); \
        access.deinterleave parameters; \
    } \
}; \
template<typename V> struct VectorTuple<LENGTH, const V> \
{ \
    typedef typename V::EntryType T; \
    typedef const V &VC_RESTRICT Reference; \
    const VectorTuple<LENGTH - 1, const V> &l; \
    Reference r; \
 \
    Vc_ALWAYS_INLINE VectorTuple(const VectorTuple<LENGTH - 1, const V> &tuple, Reference a) \
        : l(tuple), r(a) \
    { \
    } \
 \
    Vc_ALWAYS_INLINE VectorTuple<LENGTH + 1, const V> operator,(const V &a) const \
    { \
        return VectorTuple<LENGTH + 1, const V>(*this, a); \
    } \
}
_VC_VECTORTUPLE_SPECIALIZATION(3, (l.l, l.r, r));
_VC_VECTORTUPLE_SPECIALIZATION(4, (l.l.l, l.l.r, l.r, r));
_VC_VECTORTUPLE_SPECIALIZATION(5, (l.l.l.l, l.l.l.r, l.l.r, l.r, r));
_VC_VECTORTUPLE_SPECIALIZATION(6, (l.l.l.l.l, l.l.l.l.r, l.l.l.r, l.l.r, l.r, r));
_VC_VECTORTUPLE_SPECIALIZATION(7, (l.l.l.l.l.l, l.l.l.l.l.r, l.l.l.l.r, l.l.l.r, l.l.r, l.r, r));
_VC_VECTORTUPLE_SPECIALIZATION(8, (l.l.l.l.l.l.l, l.l.l.l.l.l.r, l.l.l.l.l.r, l.l.l.l.r, l.l.l.r, l.l.r, l.r, r));
//        VC_STATIC_ASSERT(false, You_are_gathering_too_many_vectors__This_is_not_implemented);

} // namespace Common

#ifdef VC_IMPL_Scalar
namespace Scalar
#elif defined VC_IMPL_SSE
namespace SSE
#elif defined VC_IMPL_AVX
namespace AVX
#endif
{

template<typename T>
Vc_ALWAYS_INLINE Common::VectorTuple<2, Vc::Vector<T> > operator,(Vc::Vector<T> &a, Vc::Vector<T> &b)
{
    return Common::VectorTuple<2, Vc::Vector<T> >(a, b);
}

template<typename T>
Vc_ALWAYS_INLINE Common::VectorTuple<2, const Vc::Vector<T> > operator,(const Vc::Vector<T> &a, const Vc::Vector<T> &b)
{
    return Common::VectorTuple<2, const Vc::Vector<T> >(a, b);
}

} // namespace Scalar/SSE/AVX

} // namespace Vc
} // namespace ROOT

#include "undomacros.h"

#endif // VC_COMMON_VECTORTUPLE_H
