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

#ifndef VC_COMMON_DEINTERLEAVE_H
#define VC_COMMON_DEINTERLEAVE_H

#include "macros.h"

namespace ROOT {
namespace Vc
{

/**
 * \ingroup Vectors
 *
 * Loads two vectors of values from an interleaved array.
 *
 * \param a, b The vectors to load the values from memory into.
 * \param memory The memory location where to read the next 2 * V::Size values from
 * \param align Either pass Vc::Aligned or Vc::Unaligned. It defaults to Vc::Aligned if nothing is
 * specified.
 *
 * If you store your data as
 * \code
 * struct { float x, y; } m[1000];
 * \endcode
 * then the deinterleave function allows you to read \p Size concurrent x and y values like this:
 * \code
 * Vc::float_v x, y;
 * Vc::deinterleave(&x, &y, &m[10], Vc::Unaligned);
 * \endcode
 * This code will load m[10], m[12], m[14], ... into \p x and m[11], m[13], m[15], ... into \p y.
 *
 * The deinterleave function supports the following type combinations:
\verbatim
  V \  M | float | double | ushort | short | uint | int
=========|=======|========|========|=======|======|=====
 float_v |   X   |        |    X   |   X   |      |
---------|-------|--------|--------|-------|------|-----
sfloat_v |   X   |        |    X   |   X   |      |
---------|-------|--------|--------|-------|------|-----
double_v |       |    X   |        |       |      |
---------|-------|--------|--------|-------|------|-----
   int_v |       |        |        |   X   |      |  X
---------|-------|--------|--------|-------|------|-----
  uint_v |       |        |    X   |       |   X  |
---------|-------|--------|--------|-------|------|-----
 short_v |       |        |        |   X   |      |
---------|-------|--------|--------|-------|------|-----
ushort_v |       |        |    X   |       |      |
\endverbatim
 */
template<typename V, typename M, typename A> Vc_ALWAYS_INLINE void deinterleave(V *a, V *b,
        const M *memory, A align)
{
    Internal::Helper::deinterleave(*a, *b, memory, align);
}

// documented as default for align above
template<typename V, typename M> Vc_ALWAYS_INLINE void deinterleave(V *a, V *b,
        const M *memory)
{
    Internal::Helper::deinterleave(*a, *b, memory, Aligned);
}

} // namespace Vc
} // namespace ROOT

#include "undomacros.h"

#endif // VC_COMMON_DEINTERLEAVE_H
