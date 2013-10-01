/*  This file is part of the Vc library.

    Copyright (C) 2010-2012 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_COMMON_STORAGE_H
#define VC_COMMON_STORAGE_H

#include "aliasingentryhelper.h"
#include "macros.h"
#include "types.h"

namespace ROOT {
namespace Vc
{
namespace Common
{

template<typename _VectorType, typename _EntryType, typename VectorTypeBase = _VectorType> class VectorMemoryUnion
{
    public:
        typedef _VectorType VectorType;
        typedef _EntryType EntryType;
        typedef EntryType AliasingEntryType Vc_MAY_ALIAS;
        Vc_ALWAYS_INLINE VectorMemoryUnion() { assertCorrectAlignment(&v()); }
#if defined VC_ICC || defined VC_MSVC
        Vc_ALWAYS_INLINE VectorMemoryUnion(const VectorType &x) { data.v = x; assertCorrectAlignment(&data.v); }
        Vc_ALWAYS_INLINE VectorMemoryUnion &operator=(const VectorType &x) {
            data.v = x; return *this;
        }

        Vc_ALWAYS_INLINE Vc_PURE VectorType &v() { return reinterpret_cast<VectorType &>(data.v); }
        Vc_ALWAYS_INLINE Vc_PURE const VectorType &v() const { return reinterpret_cast<const VectorType &>(data.v); }

#if defined VC_ICC
        Vc_ALWAYS_INLINE Vc_PURE AliasingEntryHelper<VectorMemoryUnion> m(size_t index) {
            return AliasingEntryHelper<VectorMemoryUnion>(this, index);
        }
        Vc_ALWAYS_INLINE void assign(size_t index, EntryType x) {
            data.m[index] = x;
        }
        Vc_ALWAYS_INLINE Vc_PURE EntryType read(size_t index) const {
            return data.m[index];
        }
#else
        Vc_ALWAYS_INLINE Vc_PURE EntryType &m(size_t index) {
            return data.m[index];
        }
#endif

        Vc_ALWAYS_INLINE Vc_PURE EntryType m(size_t index) const {
            return data.m[index];
        }

#ifdef VC_COMPILE_BENCHMARKS
    public:
#endif
    private:
        union VectorScalarUnion {
            VectorTypeBase v;
            EntryType m[sizeof(VectorTypeBase)/sizeof(EntryType)];
        } data;
#else
        Vc_ALWAYS_INLINE VectorMemoryUnion(VectorType x) : data(x) { assertCorrectAlignment(&data); }
        Vc_ALWAYS_INLINE VectorMemoryUnion &operator=(VectorType x) {
            data = x; return *this;
        }

        Vc_ALWAYS_INLINE Vc_PURE VectorType &v() { return data; }
        Vc_ALWAYS_INLINE Vc_PURE const VectorType &v() const { return data; }

        Vc_ALWAYS_INLINE Vc_PURE AliasingEntryType &m(size_t index) {
            return reinterpret_cast<AliasingEntryType *>(&data)[index];
        }

        Vc_ALWAYS_INLINE Vc_PURE EntryType m(size_t index) const {
            return reinterpret_cast<const AliasingEntryType *>(&data)[index];
        }

    private:
#ifdef VC_COMPILE_BENCHMARKS
    public:
#endif
        VectorType data;
#endif
};

#if VC_GCC == 0x40700 || (VC_GCC >= 0x40600 && VC_GCC <= 0x40603)
// workaround bug 52736 in GCC
template<typename T, typename V> static Vc_ALWAYS_INLINE Vc_CONST T &vectorMemoryUnionAliasedMember(V *data, size_t index) {
    if (__builtin_constant_p(index) && index == 0) {
        T *ret;
        asm("mov %1,%0" : "=r"(ret) : "r"(data));
        return *ret;
    } else {
        return reinterpret_cast<T *>(data)[index];
    }
}
template<> Vc_ALWAYS_INLINE Vc_PURE VectorMemoryUnion<__m128d, double>::AliasingEntryType &VectorMemoryUnion<__m128d, double>::m(size_t index) {
    return vectorMemoryUnionAliasedMember<AliasingEntryType>(&data, index);
}
template<> Vc_ALWAYS_INLINE Vc_PURE VectorMemoryUnion<__m128i, long long>::AliasingEntryType &VectorMemoryUnion<__m128i, long long>::m(size_t index) {
    return vectorMemoryUnionAliasedMember<AliasingEntryType>(&data, index);
}
template<> Vc_ALWAYS_INLINE Vc_PURE VectorMemoryUnion<__m128i, unsigned long long>::AliasingEntryType &VectorMemoryUnion<__m128i, unsigned long long>::m(size_t index) {
    return vectorMemoryUnionAliasedMember<AliasingEntryType>(&data, index);
}
#endif

} // namespace Common
} // namespace Vc
} // namespace ROOT

#include "undomacros.h"

#endif // VC_COMMON_STORAGE_H
