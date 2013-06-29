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

#ifndef VC_SCALAR_INTERLEAVEDMEMORY_TCC
#define VC_SCALAR_INTERLEAVEDMEMORY_TCC

#include "macros.h"
namespace ROOT {
namespace Vc
{
namespace Common
{

template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1)
{
    m_data[m_indexes.data() + 0] = v0.data();
    m_data[m_indexes.data() + 1] = v1.data();
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2)
{
    m_data[m_indexes.data() + 0] = v0.data();
    m_data[m_indexes.data() + 1] = v1.data();
    m_data[m_indexes.data() + 2] = v2.data();
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2, const typename V::AsArg v3)
{
    m_data[m_indexes.data() + 0] = v0.data();
    m_data[m_indexes.data() + 1] = v1.data();
    m_data[m_indexes.data() + 2] = v2.data();
    m_data[m_indexes.data() + 3] = v3.data();
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2, const typename V::AsArg v3, const typename V::AsArg v4)
{
    m_data[m_indexes.data() + 0] = v0.data();
    m_data[m_indexes.data() + 1] = v1.data();
    m_data[m_indexes.data() + 2] = v2.data();
    m_data[m_indexes.data() + 3] = v3.data();
    m_data[m_indexes.data() + 4] = v4.data();
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2, const typename V::AsArg v3, const typename V::AsArg v4,
        const typename V::AsArg v5)
{
    m_data[m_indexes.data() + 0] = v0.data();
    m_data[m_indexes.data() + 1] = v1.data();
    m_data[m_indexes.data() + 2] = v2.data();
    m_data[m_indexes.data() + 3] = v3.data();
    m_data[m_indexes.data() + 4] = v4.data();
    m_data[m_indexes.data() + 5] = v5.data();
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2, const typename V::AsArg v3, const typename V::AsArg v4,
        const typename V::AsArg v5, const typename V::AsArg v6)
{
    m_data[m_indexes.data() + 0] = v0.data();
    m_data[m_indexes.data() + 1] = v1.data();
    m_data[m_indexes.data() + 2] = v2.data();
    m_data[m_indexes.data() + 3] = v3.data();
    m_data[m_indexes.data() + 4] = v4.data();
    m_data[m_indexes.data() + 5] = v5.data();
    m_data[m_indexes.data() + 6] = v6.data();
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2, const typename V::AsArg v3, const typename V::AsArg v4,
        const typename V::AsArg v5, const typename V::AsArg v6, const typename V::AsArg v7)
{
    m_data[m_indexes.data() + 0] = v0.data();
    m_data[m_indexes.data() + 1] = v1.data();
    m_data[m_indexes.data() + 2] = v2.data();
    m_data[m_indexes.data() + 3] = v3.data();
    m_data[m_indexes.data() + 4] = v4.data();
    m_data[m_indexes.data() + 5] = v5.data();
    m_data[m_indexes.data() + 6] = v6.data();
    m_data[m_indexes.data() + 7] = v7.data();
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::deinterleave(V &v0, V &v1) const/*{{{*/
{
    v0.data() = m_data[m_indexes.data() + 0];
    v1.data() = m_data[m_indexes.data() + 1];
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::deinterleave(V &v0, V &v1, V &v2) const/*{{{*/
{
    v0.data() = m_data[m_indexes.data() + 0];
    v1.data() = m_data[m_indexes.data() + 1];
    v2.data() = m_data[m_indexes.data() + 2];
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::deinterleave(V &v0, V &v1, V &v2, V &v3) const/*{{{*/
{
    v0.data() = m_data[m_indexes.data() + 0];
    v1.data() = m_data[m_indexes.data() + 1];
    v2.data() = m_data[m_indexes.data() + 2];
    v3.data() = m_data[m_indexes.data() + 3];
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::deinterleave(V &v0, V &v1, V &v2, V &v3, V &v4) const/*{{{*/
{
    v0.data() = m_data[m_indexes.data() + 0];
    v1.data() = m_data[m_indexes.data() + 1];
    v2.data() = m_data[m_indexes.data() + 2];
    v3.data() = m_data[m_indexes.data() + 3];
    v4.data() = m_data[m_indexes.data() + 4];
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::deinterleave(V &v0, V &v1, V &v2, V &v3, V &v4, V &v5) const/*{{{*/
{
    v0.data() = m_data[m_indexes.data() + 0];
    v1.data() = m_data[m_indexes.data() + 1];
    v2.data() = m_data[m_indexes.data() + 2];
    v3.data() = m_data[m_indexes.data() + 3];
    v4.data() = m_data[m_indexes.data() + 4];
    v5.data() = m_data[m_indexes.data() + 5];
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::deinterleave(V &v0, V &v1, V &v2, V &v3, V &v4, V &v5, V &v6) const/*{{{*/
{
    v0.data() = m_data[m_indexes.data() + 0];
    v1.data() = m_data[m_indexes.data() + 1];
    v2.data() = m_data[m_indexes.data() + 2];
    v3.data() = m_data[m_indexes.data() + 3];
    v4.data() = m_data[m_indexes.data() + 4];
    v5.data() = m_data[m_indexes.data() + 5];
    v6.data() = m_data[m_indexes.data() + 6];
}/*}}}*/
template<typename V> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V>::deinterleave(V &v0, V &v1, V &v2, V &v3, V &v4, V &v5, V &v6, V &v7) const/*{{{*/
{
    v0.data() = m_data[m_indexes.data() + 0];
    v1.data() = m_data[m_indexes.data() + 1];
    v2.data() = m_data[m_indexes.data() + 2];
    v3.data() = m_data[m_indexes.data() + 3];
    v4.data() = m_data[m_indexes.data() + 4];
    v5.data() = m_data[m_indexes.data() + 5];
    v6.data() = m_data[m_indexes.data() + 6];
    v7.data() = m_data[m_indexes.data() + 7];
}/*}}}*/

} // namespace Common
} // namespace Vc
} // namespace ROOT
#include "undomacros.h"

#endif // VC_SCALAR_INTERLEAVEDMEMORY_TCC

// vim: foldmethod=marker
