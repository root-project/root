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

#ifndef VC_COMMON_INTERLEAVEDMEMORY_H
#define VC_COMMON_INTERLEAVEDMEMORY_H

#include "macros.h"

namespace ROOT {
namespace Vc
{
namespace Common
{

namespace Internal
{
template<typename A, typename B> struct CopyConst { typedef B Type; };
template<typename A, typename B> struct CopyConst<const A, B> { typedef const B Type; };

template<typename S, typename X, typename R> struct EnableInterleaves { typedef R Type; };
template<typename S, typename X, typename R> struct EnableInterleaves<const S, X, R>;
}  // namespace Internal

/**
 * \internal
 */
template<typename V> struct InterleavedMemoryAccessBase
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    typedef typename V::AsArg VArg;
    typedef T Ta Vc_MAY_ALIAS;
    const I m_indexes;
    Ta *const m_data;

    Vc_ALWAYS_INLINE InterleavedMemoryAccessBase(typename I::AsArg indexes, Ta *data)
        : m_indexes(indexes), m_data(data)
    {
    }

    // implementations of the following are in {scalar,sse,avx}/interleavedmemory.tcc
    void deinterleave(V &v0, V &v1) const;
    void deinterleave(V &v0, V &v1, V &v2) const;
    void deinterleave(V &v0, V &v1, V &v2, V &v3) const;
    void deinterleave(V &v0, V &v1, V &v2, V &v3, V &v4) const;
    void deinterleave(V &v0, V &v1, V &v2, V &v3, V &v4, V &v5) const;
    void deinterleave(V &v0, V &v1, V &v2, V &v3, V &v4, V &v5, V &v6) const;
    void deinterleave(V &v0, V &v1, V &v2, V &v3, V &v4, V &v5, V &v6, V &v7) const;

    void interleave(VArg v0, VArg v1);
    void interleave(VArg v0, VArg v1, VArg v2);
    void interleave(VArg v0, VArg v1, VArg v2, VArg v3);
    void interleave(VArg v0, VArg v1, VArg v2, VArg v3, VArg v4);
    void interleave(VArg v0, VArg v1, VArg v2, VArg v3, VArg v4, VArg v5);
    void interleave(VArg v0, VArg v1, VArg v2, VArg v3, VArg v4, VArg v5, VArg v6);
    void interleave(VArg v0, VArg v1, VArg v2, VArg v3, VArg v4, VArg v5, VArg v6, VArg v7);
};

/**
 * \internal
 */
// delay execution of the deinterleaving gather until operator=
template<size_t StructSize, typename V> struct InterleavedMemoryReadAccess : public InterleavedMemoryAccessBase<V>
{
    typedef InterleavedMemoryAccessBase<V> Base;
    typedef typename Base::Ta Ta;
    typedef typename Base::I I;

    Vc_ALWAYS_INLINE InterleavedMemoryReadAccess(const Ta *data, typename I::AsArg indexes)
        : Base(indexes * I(StructSize), const_cast<Ta *>(data)) // this needs to be refactored to properly keep the constness
    {
    }
};

/**
 * \internal
 */
template<size_t StructSize, typename V> struct InterleavedMemoryAccess : public InterleavedMemoryReadAccess<StructSize, V>
{
    typedef InterleavedMemoryAccessBase<V> Base;
    typedef typename Base::Ta Ta;
    typedef typename Base::I I;

    Vc_ALWAYS_INLINE InterleavedMemoryAccess(Ta *data, typename I::AsArg indexes)
        : InterleavedMemoryReadAccess<StructSize, V>(data, indexes)
    {
    }

#define _VC_SCATTER_ASSIGNMENT(LENGTH, parameters) \
    Vc_ALWAYS_INLINE void operator=(const VectorTuple<LENGTH, V> &rhs) \
    { \
        VC_STATIC_ASSERT(LENGTH <= StructSize, You_are_trying_to_scatter_more_data_into_the_struct_than_it_has); \
        this->interleave parameters ; \
    } \
    Vc_ALWAYS_INLINE void operator=(const VectorTuple<LENGTH, const V> &rhs) \
    { \
        VC_STATIC_ASSERT(LENGTH <= StructSize, You_are_trying_to_scatter_more_data_into_the_struct_than_it_has); \
        checkIndexesUnique(); \
        this->interleave parameters ; \
    }
    _VC_SCATTER_ASSIGNMENT(2, (rhs.l, rhs.r))
    _VC_SCATTER_ASSIGNMENT(3, (rhs.l.l, rhs.l.r, rhs.r));
    _VC_SCATTER_ASSIGNMENT(4, (rhs.l.l.l, rhs.l.l.r, rhs.l.r, rhs.r));
    _VC_SCATTER_ASSIGNMENT(5, (rhs.l.l.l.l, rhs.l.l.l.r, rhs.l.l.r, rhs.l.r, rhs.r));
    _VC_SCATTER_ASSIGNMENT(6, (rhs.l.l.l.l.l, rhs.l.l.l.l.r, rhs.l.l.l.r, rhs.l.l.r, rhs.l.r, rhs.r));
    _VC_SCATTER_ASSIGNMENT(7, (rhs.l.l.l.l.l.l, rhs.l.l.l.l.l.r, rhs.l.l.l.l.r, rhs.l.l.l.r, rhs.l.l.r, rhs.l.r, rhs.r));
    _VC_SCATTER_ASSIGNMENT(8, (rhs.l.l.l.l.l.l.l, rhs.l.l.l.l.l.l.r, rhs.l.l.l.l.l.r, rhs.l.l.l.l.r, rhs.l.l.l.r, rhs.l.l.r, rhs.l.r, rhs.r));
#undef _VC_SCATTER_ASSIGNMENT

private:
#ifdef NDEBUG
    Vc_ALWAYS_INLINE void checkIndexesUnique() const {}
#else
    void checkIndexesUnique() const
    {
        const I test = Base::m_indexes.sorted();
        VC_ASSERT(I::Size == 1 || (test == test.rotated(1)).isEmpty())
    }
#endif
};

#ifdef DOXYGEN
} // namespace Common
// in doxygen InterleavedMemoryWrapper should appear in the Vc namespace (see the using statement
// below)
#endif

/**
 * Wraps a pointer to memory with convenience functions to access it via vectors.
 *
 * \param S The type of the struct.
 * \param V The type of the vector to be returned when read. This should reflect the type of the
 * members inside the struct.
 *
 * \see operator[]
 * \ingroup Utilities
 * \headerfile interleavedmemory.h <Vc/Memory>
 */
template<typename S, typename V> class InterleavedMemoryWrapper
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    typedef typename V::AsArg VArg;
    typedef typename I::AsArg IndexType;
    typedef InterleavedMemoryAccess<sizeof(S) / sizeof(T), V> Access;
    typedef InterleavedMemoryReadAccess<sizeof(S) / sizeof(T), V> ReadAccess;
    typedef typename Internal::CopyConst<S, T>::Type Ta Vc_MAY_ALIAS;
    Ta *const m_data;

    VC_STATIC_ASSERT((sizeof(S) / sizeof(T)) * sizeof(T) == sizeof(S), InterleavedMemoryAccess_does_not_support_packed_structs);

public:
    /**
     * Constructs the wrapper object.
     *
     * \param s A pointer to a C-array.
     */
    Vc_ALWAYS_INLINE InterleavedMemoryWrapper(S *s)
        : m_data(reinterpret_cast<Ta *>(s))
    {
    }

    /**
     * Interleaved scatter/gather access.
     *
     * Assuming you have a struct of floats and a vector of \p indexes into the array, this function
     * can be used to access the struct entries as vectors using the minimal number of store or load
     * instructions.
     *
     * \param indexes Vector of indexes that determine the gather locations.
     *
     * \return A special (magic) object that executes the loads and deinterleave on assignment to a
     * vector tuple.
     *
     * Example:
     * \code
     * struct Foo {
     *   float x, y, z;
     * };
     *
     * void fillWithBar(Foo *_data, uint_v indexes)
     * {
     *   Vc::InterleavedMemoryWrapper<Foo, float_v> data(_data);
     *   const float_v x = bar(1);
     *   const float_v y = bar(2);
     *   const float_v z = bar(3);
     *   data[indexes] = (x, y, z);
     *   // it's also possible to just store a subset at the front of the struct:
     *   data[indexes] = (x, y);
     *   // if you want to store a single entry, use scatter:
     *   z.scatter(_data, &Foo::x, indexes);
     * }
     *
     * float_v normalizeStuff(Foo *_data, uint_v indexes)
     * {
     *   Vc::InterleavedMemoryWrapper<Foo, float_v> data(_data);
     *   float_v x, y, z;
     *   (x, y, z) = data[indexes];
     *   // it is also possible to just load a subset from the front of the struct:
     *   // (x, y) = data[indexes];
     *   return Vc::sqrt(x * x + y * y + z * z);
     * }
     * \endcode
     *
     * You may think of the gather operation (or scatter as the inverse) like this:
\verbatim
             Memory: {x0 y0 z0 x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 x5 y5 z5 x6 y6 z6 x7 y7 z7 x8 y8 z8}
            indexes: [5, 0, 1, 7]
Result in (x, y, z): ({x5 x0 x1 x7}, {y5 y0 y1 y7}, {z5 z0 z1 z7})
\endverbatim
     *
     * \warning If \p indexes contains non-unique entries on scatter, the result is undefined. If
     * \c NDEBUG is not defined the implementation will assert that the \p indexes entries are unique.
     */
#ifdef DOXYGEN
    Vc_ALWAYS_INLINE Access operator[](IndexType indexes)
#else
    // need to SFINAE disable this for objects that wrap constant data
    template <typename U>
    Vc_ALWAYS_INLINE typename Internal::EnableInterleaves<S, U, Access>::Type operator[](
        VC_ALIGNED_PARAMETER(U) indexes)
#endif
    {
        return Access(m_data, indexes);
    }

    /// const overload (gathers only) of the above function
    Vc_ALWAYS_INLINE ReadAccess operator[](VC_ALIGNED_PARAMETER(IndexType) indexes) const
    {
        return ReadAccess(m_data, indexes);
    }

    /// alias of the above function
    Vc_ALWAYS_INLINE ReadAccess gather(VC_ALIGNED_PARAMETER(IndexType) indexes) const
    {
        return operator[](indexes);
    }

    //Vc_ALWAYS_INLINE Access scatter(I indexes, VArg v0, VArg v1);
};
#ifndef DOXYGEN
} // namespace Common

using Common::InterleavedMemoryWrapper;
#endif

} // namespace Vc
} // namespace ROOT

#include "undomacros.h"

#endif // VC_COMMON_INTERLEAVEDMEMORY_H
