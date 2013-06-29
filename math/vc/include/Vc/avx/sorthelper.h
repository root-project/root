/*  This file is part of the Vc library.

    Copyright (C) 2011 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_AVX_SORTHELPER_H
#define VC_AVX_SORTHELPER_H

#include "types.h"

namespace ROOT {
namespace Vc
{
namespace AVX
{
template<typename T> struct SortHelper
{
    typedef typename VectorTypeHelper<T>::Type VectorType;
#ifdef VC_PASSING_VECTOR_BY_VALUE_IS_BROKEN
    typedef const VectorType & VTArg;
#else
    typedef const VectorType VTArg;
#endif
    static VectorType sort(VTArg);
    static void sort(VectorType &, VectorType &);
};
} // namespace AVX
} // namespace Vc
} // namespace ROOT

#endif // VC_AVX_SORTHELPER_H
