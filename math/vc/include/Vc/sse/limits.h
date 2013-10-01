/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_SSE_LIMITS_H
#define VC_SSE_LIMITS_H

#include "intrinsics.h"
#include "types.h"
#include "macros.h"

namespace std
{
template<> struct numeric_limits< ::ROOT::Vc::SSE::ushort_v> : public numeric_limits<unsigned short>
{
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::ushort_v max()           _VC_NOEXCEPT { return ::ROOT::Vc::SSE::_mm_setallone_si128(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::ushort_v min()           _VC_NOEXCEPT { return ::ROOT::Vc::SSE::ushort_v::Zero(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::ushort_v lowest()        _VC_NOEXCEPT { return min(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::ushort_v epsilon()       _VC_NOEXCEPT { return ::ROOT::Vc::SSE::ushort_v::Zero(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::ushort_v round_error()   _VC_NOEXCEPT { return ::ROOT::Vc::SSE::ushort_v::Zero(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::ushort_v infinity()      _VC_NOEXCEPT { return ::ROOT::Vc::SSE::ushort_v::Zero(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::ushort_v quiet_NaN()     _VC_NOEXCEPT { return ::ROOT::Vc::SSE::ushort_v::Zero(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::ushort_v signaling_NaN() _VC_NOEXCEPT { return ::ROOT::Vc::SSE::ushort_v::Zero(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::ushort_v denorm_min()    _VC_NOEXCEPT { return ::ROOT::Vc::SSE::ushort_v::Zero(); }
};
template<> struct numeric_limits< ::ROOT::Vc::SSE::short_v> : public numeric_limits<short>
{
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::short_v max()           _VC_NOEXCEPT { return _mm_srli_epi16(::ROOT::Vc::SSE::_mm_setallone_si128(), 1); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::short_v min()           _VC_NOEXCEPT { return ::ROOT::Vc::SSE::_mm_setmin_epi16(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::short_v lowest()        _VC_NOEXCEPT { return min(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::short_v epsilon()       _VC_NOEXCEPT { return ::ROOT::Vc::SSE::short_v::Zero(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::short_v round_error()   _VC_NOEXCEPT { return ::ROOT::Vc::SSE::short_v::Zero(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::short_v infinity()      _VC_NOEXCEPT { return ::ROOT::Vc::SSE::short_v::Zero(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::short_v quiet_NaN()     _VC_NOEXCEPT { return ::ROOT::Vc::SSE::short_v::Zero(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::short_v signaling_NaN() _VC_NOEXCEPT { return ::ROOT::Vc::SSE::short_v::Zero(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::short_v denorm_min()    _VC_NOEXCEPT { return ::ROOT::Vc::SSE::short_v::Zero(); }
};
template<> struct numeric_limits< ::ROOT::Vc::SSE::uint_v> : public numeric_limits<unsigned int>
{
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::uint_v max()           _VC_NOEXCEPT { return ::ROOT::Vc::SSE::_mm_setallone_si128(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::uint_v min()           _VC_NOEXCEPT { return ::ROOT::Vc::SSE::uint_v::Zero(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::uint_v lowest()        _VC_NOEXCEPT { return min(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::uint_v epsilon()       _VC_NOEXCEPT { return ::ROOT::Vc::SSE::uint_v::Zero(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::uint_v round_error()   _VC_NOEXCEPT { return ::ROOT::Vc::SSE::uint_v::Zero(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::uint_v infinity()      _VC_NOEXCEPT { return ::ROOT::Vc::SSE::uint_v::Zero(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::uint_v quiet_NaN()     _VC_NOEXCEPT { return ::ROOT::Vc::SSE::uint_v::Zero(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::uint_v signaling_NaN() _VC_NOEXCEPT { return ::ROOT::Vc::SSE::uint_v::Zero(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::uint_v denorm_min()    _VC_NOEXCEPT { return ::ROOT::Vc::SSE::uint_v::Zero(); }
};
template<> struct numeric_limits< ::ROOT::Vc::SSE::int_v> : public numeric_limits<int>
{
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::int_v max()           _VC_NOEXCEPT { return _mm_srli_epi32(::ROOT::Vc::SSE::_mm_setallone_si128(), 1); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::int_v min()           _VC_NOEXCEPT { return ::ROOT::Vc::SSE::_mm_setmin_epi32(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::int_v lowest()        _VC_NOEXCEPT { return min(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::int_v epsilon()       _VC_NOEXCEPT { return ::ROOT::Vc::SSE::int_v::Zero(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::int_v round_error()   _VC_NOEXCEPT { return ::ROOT::Vc::SSE::int_v::Zero(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::int_v infinity()      _VC_NOEXCEPT { return ::ROOT::Vc::SSE::int_v::Zero(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::int_v quiet_NaN()     _VC_NOEXCEPT { return ::ROOT::Vc::SSE::int_v::Zero(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::int_v signaling_NaN() _VC_NOEXCEPT { return ::ROOT::Vc::SSE::int_v::Zero(); }
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::SSE::int_v denorm_min()    _VC_NOEXCEPT { return ::ROOT::Vc::SSE::int_v::Zero(); }
};
} // namespace std

#include "undomacros.h"

#endif // VC_SSE_LIMITS_H
