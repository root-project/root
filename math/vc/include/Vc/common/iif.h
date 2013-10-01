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

#ifndef VC_COMMON_IIF_H
#define VC_COMMON_IIF_H

#include "macros.h"

namespace ROOT {
namespace Vc
{
/**
 * Function to mimic the ternary operator '?:'.
 *
 * \param condition  Determines which values are returned. This is analog to the first argument to
 *                   the ternary operator.
 * \param trueValue  The values to return where \p condition is \c true.
 * \param falseValue The values to return where \p condition is \c false.
 * \return A combination of entries from \p trueValue and \p falseValue, according to \p condition.
 *
 * So instead of the scalar variant
 * \code
 * float x = a > 1.f ? b : b + c;
 * \endcode
 * you'd write
 * \code
 * float_v x = Vc::iif (a > 1.f, b, b + c);
 * \endcode
 */
#ifndef VC_MSVC
template<typename T> static Vc_ALWAYS_INLINE Vector<T> iif (typename Vector<T>::Mask condition, Vector<T> trueValue, Vector<T> falseValue)
{
#else
template<typename T> static Vc_ALWAYS_INLINE Vector<T> iif (const typename Vector<T>::Mask &condition, const Vector<T> &trueValue, const Vector<T> &_falseValue)
{
    Vector<T> falseValue(_falseValue);
#endif
    falseValue(condition) = trueValue;
    return falseValue;
}
} // namespace Vc
} // namespace ROOT

#include "undomacros.h"

#endif // VC_COMMON_IIF_H
