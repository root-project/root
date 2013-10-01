/*  This file is part of the Vc library.

    Copyright (C) 2010-2011 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_COMMON_ALIASINGENTRYHELPER_H
#define VC_COMMON_ALIASINGENTRYHELPER_H

#include "macros.h"

namespace ROOT {
namespace Vc
{
namespace Common
{

template<class StorageType> class AliasingEntryHelper
{
    private:
        typedef typename StorageType::EntryType T;
#ifdef VC_ICC
        StorageType *const m_storage;
        const int m_index;
    public:
        Vc_ALWAYS_INLINE AliasingEntryHelper(StorageType *d, int index) : m_storage(d), m_index(index) {}
        Vc_ALWAYS_INLINE AliasingEntryHelper(const AliasingEntryHelper &rhs) : m_storage(rhs.m_storage), m_index(rhs.m_index) {}
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator=(const AliasingEntryHelper &rhs) {
            m_storage->assign(m_index, rhs);
            return *this;
        }

        Vc_ALWAYS_INLINE AliasingEntryHelper &operator  =(T x) { m_storage->assign(m_index, x); return *this; }
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator +=(T x) { m_storage->assign(m_index, m_storage->m(m_index) + x); return *this; }
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator -=(T x) { m_storage->assign(m_index, m_storage->m(m_index) - x); return *this; }
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator /=(T x) { m_storage->assign(m_index, m_storage->m(m_index) / x); return *this; }
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator *=(T x) { m_storage->assign(m_index, m_storage->m(m_index) * x); return *this; }
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator |=(T x) { m_storage->assign(m_index, m_storage->m(m_index) | x); return *this; }
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator &=(T x) { m_storage->assign(m_index, m_storage->m(m_index) & x); return *this; }
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator ^=(T x) { m_storage->assign(m_index, m_storage->m(m_index) ^ x); return *this; }
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator %=(T x) { m_storage->assign(m_index, m_storage->m(m_index) % x); return *this; }
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator<<=(T x) { m_storage->assign(m_index, m_storage->m(m_index)<< x); return *this; }
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator>>=(T x) { m_storage->assign(m_index, m_storage->m(m_index)>> x); return *this; }

        Vc_ALWAYS_INLINE AliasingEntryHelper &operator++() { m_storage->assign(m_index, m_storage->m(m_index) + T(1)); return *this; }
        Vc_ALWAYS_INLINE T operator++(int) { T r = m_storage->m(m_index); m_storage->assign(m_index, m_storage->m(m_index) + T(1)); return r; }
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator--() { m_storage->assign(m_index, m_storage->m(m_index) - T(1)); return *this; }
        Vc_ALWAYS_INLINE T operator--(int) { T r = m_storage->m(m_index); m_storage->assign(m_index, m_storage->m(m_index) - T(1)); return r; }
#define m_data m_storage->read(m_index)
#else
        typedef T A Vc_MAY_ALIAS;
        A &m_data;
    public:
        template<typename T2>
        Vc_ALWAYS_INLINE AliasingEntryHelper(T2 &d) : m_data(reinterpret_cast<A &>(d)) {}

        Vc_ALWAYS_INLINE AliasingEntryHelper(A &d) : m_data(d) {}
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator=(const AliasingEntryHelper &rhs) {
            m_data = rhs.m_data;
            return *this;
        }

        Vc_ALWAYS_INLINE AliasingEntryHelper &operator =(T x) { m_data  = x; return *this; }
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator+=(T x) { m_data += x; return *this; }
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator-=(T x) { m_data -= x; return *this; }
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator/=(T x) { m_data /= x; return *this; }
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator*=(T x) { m_data *= x; return *this; }
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator|=(T x) { m_data |= x; return *this; }
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator&=(T x) { m_data &= x; return *this; }
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator^=(T x) { m_data ^= x; return *this; }
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator%=(T x) { m_data %= x; return *this; }
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator<<=(T x) { m_data <<= x; return *this; }
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator>>=(T x) { m_data >>= x; return *this; }

        Vc_ALWAYS_INLINE AliasingEntryHelper &operator++() { ++m_data; return *this; }
        Vc_ALWAYS_INLINE T operator++(int) { T r = m_data; ++m_data; return r; }
        Vc_ALWAYS_INLINE AliasingEntryHelper &operator--() { --m_data; return *this; }
        Vc_ALWAYS_INLINE T operator--(int) { T r = m_data; --m_data; return r; }
#endif

        Vc_ALWAYS_INLINE Vc_PURE operator const T() const { return m_data; }

        Vc_ALWAYS_INLINE Vc_PURE bool operator==(T x) const { return static_cast<T>(m_data) == x; }
        Vc_ALWAYS_INLINE Vc_PURE bool operator!=(T x) const { return static_cast<T>(m_data) != x; }
        Vc_ALWAYS_INLINE Vc_PURE bool operator<=(T x) const { return static_cast<T>(m_data) <= x; }
        Vc_ALWAYS_INLINE Vc_PURE bool operator>=(T x) const { return static_cast<T>(m_data) >= x; }
        Vc_ALWAYS_INLINE Vc_PURE bool operator< (T x) const { return static_cast<T>(m_data) <  x; }
        Vc_ALWAYS_INLINE Vc_PURE bool operator> (T x) const { return static_cast<T>(m_data) >  x; }

        Vc_ALWAYS_INLINE Vc_PURE T operator-() const { return -static_cast<T>(m_data); }
        Vc_ALWAYS_INLINE Vc_PURE T operator~() const { return ~static_cast<T>(m_data); }
        Vc_ALWAYS_INLINE Vc_PURE T operator+(T x) const { return static_cast<T>(m_data) + x; }
        Vc_ALWAYS_INLINE Vc_PURE T operator-(T x) const { return static_cast<T>(m_data) - x; }
        Vc_ALWAYS_INLINE Vc_PURE T operator/(T x) const { return static_cast<T>(m_data) / x; }
        Vc_ALWAYS_INLINE Vc_PURE T operator*(T x) const { return static_cast<T>(m_data) * x; }
        Vc_ALWAYS_INLINE Vc_PURE T operator|(T x) const { return static_cast<T>(m_data) | x; }
        Vc_ALWAYS_INLINE Vc_PURE T operator&(T x) const { return static_cast<T>(m_data) & x; }
        Vc_ALWAYS_INLINE Vc_PURE T operator^(T x) const { return static_cast<T>(m_data) ^ x; }
        Vc_ALWAYS_INLINE Vc_PURE T operator%(T x) const { return static_cast<T>(m_data) % x; }
        //T operator<<(T x) const { return static_cast<T>(m_data) << x; }
        //T operator>>(T x) const { return static_cast<T>(m_data) >> x; }
#ifdef m_data
#undef m_data
#endif
};

} // namespace Common
} // namespace Vc
} // namespace ROOT

#include "undomacros.h"

#endif // VC_COMMON_ALIASINGENTRYHELPER_H
