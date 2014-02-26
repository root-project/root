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

#ifndef VC_SCALAR_DEINTERLEAVE_H
#define VC_SCALAR_DEINTERLEAVE_H

#include "macros.h"

namespace ROOT {
namespace Vc
{
namespace Internal
{

template<> struct HelperImpl<Vc::ScalarImpl>
{
    template<typename V, typename M, typename A>
    static Vc_ALWAYS_INLINE void deinterleave(V &a, V &b, const M *mem, A)
    {
        a = mem[0];
        b = mem[1];
    }

    static Vc_ALWAYS_INLINE void prefetchForOneRead(const void *) {}
    static Vc_ALWAYS_INLINE void prefetchForModify(const void *) {}
    static Vc_ALWAYS_INLINE void prefetchClose(const void *) {}
    static Vc_ALWAYS_INLINE void prefetchMid(const void *) {}
    static Vc_ALWAYS_INLINE void prefetchFar(const void *) {}

    template<Vc::MallocAlignment A>
    static Vc_ALWAYS_INLINE_L void *malloc(size_t n) Vc_ALWAYS_INLINE_R;
    static Vc_ALWAYS_INLINE_L void free(void *p) Vc_ALWAYS_INLINE_R;
};

} // namespace Scalar
} // namespace Vc
} // namespace ROOT

#include "helperimpl.tcc"
#include "undomacros.h"

#endif // VC_SCALAR_DEINTERLEAVE_H
