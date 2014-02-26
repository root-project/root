/*  This file is part of the Vc library.

    Copyright (C) 2009-2012 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_COMMON_TRIGONOMETRIC_H
#define VC_COMMON_TRIGONOMETRIC_H

#ifndef VC__USE_NAMESPACE
#error "Do not include Vc/common/trigonometric.h outside of Vc itself"
#endif

#include "macros.h"
namespace ROOT {
namespace Vc
{
namespace
{
    using Vc::VC__USE_NAMESPACE::Vector;
} // namespace

namespace Internal
{
template<Vc::Implementation Impl> struct MapImpl { enum Dummy { Value = Impl }; };
template<> struct MapImpl<Vc::SSE42Impl> { enum Dummy { Value = MapImpl<Vc::SSE41Impl>::Value }; };
typedef ImplementationT<MapImpl<VC_IMPL>::Value
#if defined(VC_IMPL_XOP) && defined(VC_IMPL_FMA4)
    + Vc::XopInstructions
    + Vc::Fma4Instructions
#endif
    > TrigonometricImplementation;
} // namespace Internal

template<typename Impl> struct Trigonometric
{
    template<typename T> static Vector<T> sin(const Vector<T> &_x);
    template<typename T> static Vector<T> cos(const Vector<T> &_x);
    template<typename T> static void sincos(const Vector<T> &_x, Vector<T> *_sin, Vector<T> *_cos);
    template<typename T> static Vector<T> asin (const Vector<T> &_x);
    template<typename T> static Vector<T> atan (const Vector<T> &_x);
    template<typename T> static Vector<T> atan2(const Vector<T> &y, const Vector<T> &x);
};
namespace VC__USE_NAMESPACE
#undef VC__USE_NAMESPACE
{
    template<typename T> static Vc_ALWAYS_INLINE Vc_PURE Vector<T> sin(const Vector<T> &_x) {
        return Vc::Trigonometric<Vc::Internal::TrigonometricImplementation>::sin(_x);
    }
    template<typename T> static Vc_ALWAYS_INLINE Vc_PURE Vector<T> cos(const Vector<T> &_x) {
        return Vc::Trigonometric<Vc::Internal::TrigonometricImplementation>::cos(_x);
    }
    template<typename T> static Vc_ALWAYS_INLINE void sincos(const Vector<T> &_x, Vector<T> *_sin, Vector<T> *_cos) {
        Vc::Trigonometric<Vc::Internal::TrigonometricImplementation>::sincos(_x, _sin, _cos);
    }
    template<typename T> static Vc_ALWAYS_INLINE Vc_PURE Vector<T> asin (const Vector<T> &_x) {
        return Vc::Trigonometric<Vc::Internal::TrigonometricImplementation>::asin(_x);
    }
    template<typename T> static Vc_ALWAYS_INLINE Vc_PURE Vector<T> atan (const Vector<T> &_x) {
        return Vc::Trigonometric<Vc::Internal::TrigonometricImplementation>::atan(_x);
    }
    template<typename T> static Vc_ALWAYS_INLINE Vc_PURE Vector<T> atan2(const Vector<T> &y, const Vector<T> &x) {
        return Vc::Trigonometric<Vc::Internal::TrigonometricImplementation>::atan2(y, x);
    }
} // namespace VC__USE_NAMESPACE
} // namespace Vc
} // namespace ROOT

#include "undomacros.h"
#endif // VC_COMMON_TRIGONOMETRIC_H
