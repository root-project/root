/*  This file is part of the Vc library.

    Copyright (C) 2009-2010 Matthias Kretz <kretz@kde.org>

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

#include "../common/macros.h"

#ifndef VC_SSE_MACROS_H
#define VC_SSE_MACROS_H
#undef VC_SSE_UNDOMACROS_H

#ifndef _M128
# define _M128 __m128
#endif

#ifndef _M128I
# define _M128I __m128i
#endif

#ifndef _M128D
# define _M128D __m128d
#endif

#define STORE_VECTOR(type, name, vec) \
    union { __m128i p; type v[16 / sizeof(type)]; } CAT(u, __LINE__); \
    _mm_store_si128(&CAT(u, __LINE__).p, vec); \
    const type *const name = &CAT(u, __LINE__).v[0]

#if defined(VC_IMPL_SSE4_1) && !defined(VC_DISABLE_PTEST)
#define VC_USE_PTEST
#endif

#endif // VC_SSE_MACROS_H
