/*  This file is part of the Vc library.

    Copyright (C) 2009-2011 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_AVX_LIMITS_H
#define VC_AVX_LIMITS_H

#include "intrinsics.h"
#include "types.h"

namespace std
{
#define _VC_NUM_LIM(T, _max, _min) \
template<> struct numeric_limits< ::ROOT::Vc::AVX::Vector<T> > : public numeric_limits<T> \
{ \
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::AVX::Vector<T> max()           _VC_NOEXCEPT { return _max; } \
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::AVX::Vector<T> min()           _VC_NOEXCEPT { return _min; } \
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::AVX::Vector<T> lowest()        _VC_NOEXCEPT { return min(); } \
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::AVX::Vector<T> epsilon()       _VC_NOEXCEPT { return ::ROOT::Vc::AVX::Vector<T>::Zero(); } \
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::AVX::Vector<T> round_error()   _VC_NOEXCEPT { return ::ROOT::Vc::AVX::Vector<T>::Zero(); } \
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::AVX::Vector<T> infinity()      _VC_NOEXCEPT { return ::ROOT::Vc::AVX::Vector<T>::Zero(); } \
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::AVX::Vector<T> quiet_NaN()     _VC_NOEXCEPT { return ::ROOT::Vc::AVX::Vector<T>::Zero(); } \
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::AVX::Vector<T> signaling_NaN() _VC_NOEXCEPT { return ::ROOT::Vc::AVX::Vector<T>::Zero(); } \
    static Vc_INTRINSIC Vc_CONST ::ROOT::Vc::AVX::Vector<T> denorm_min()    _VC_NOEXCEPT { return ::ROOT::Vc::AVX::Vector<T>::Zero(); } \
}

#ifndef VC_IMPL_AVX2
namespace {
    using ::ROOT::Vc::AVX::_mm256_srli_epi32;
}
#endif
_VC_NUM_LIM(unsigned short, ::ROOT::Vc::AVX::_mm_setallone_si128(), _mm_setzero_si128());
_VC_NUM_LIM(         short, _mm_srli_epi16(::ROOT::Vc::AVX::_mm_setallone_si128(), 1), ::ROOT::Vc::AVX::_mm_setmin_epi16());
_VC_NUM_LIM(  unsigned int, ::ROOT::Vc::AVX::_mm256_setallone_si256(), _mm256_setzero_si256());
_VC_NUM_LIM(           int, _mm256_srli_epi32(::ROOT::Vc::AVX::_mm256_setallone_si256(), 1), ::ROOT::Vc::AVX::_mm256_setmin_epi32());
#undef _VC_NUM_LIM

} // namespace std

#endif // VC_AVX_LIMITS_H
