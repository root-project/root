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

#ifndef VC_SSE_UNDOMACROS_H
#define VC_SSE_UNDOMACROS_H
#undef VC_SSE_MACROS_H

#undef STORE_VECTOR

#ifdef VC_USE_PTEST
#undef VC_USE_PTEST
#endif

#endif // VC_SSE_UNDOMACROS_H

#include "../common/undomacros.h"
