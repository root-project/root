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

#ifndef VC_SSE_DEINTERLEAVE_H
#define VC_SSE_DEINTERLEAVE_H

#include "macros.h"

namespace ROOT {
namespace Vc
{
namespace Internal
{

template<> struct HelperImpl<Vc::SSE2Impl>
{
    typedef SSE::Vector<float> float_v;
    typedef SSE::Vector<SSE::float8> sfloat_v;
    typedef SSE::Vector<double> double_v;
    typedef SSE::Vector<int> int_v;
    typedef SSE::Vector<unsigned int> uint_v;
    typedef SSE::Vector<short> short_v;
    typedef SSE::Vector<unsigned short> ushort_v;

    template<typename A> static void deinterleave(float_v &, float_v &, const float *, A);
    template<typename A> static void deinterleave(float_v &, float_v &, const short *, A);
    template<typename A> static void deinterleave(float_v &, float_v &, const unsigned short *, A);

    template<typename A> static void deinterleave(sfloat_v &, sfloat_v &, const float *, A);
    template<typename A> static void deinterleave(sfloat_v &, sfloat_v &, const short *, A);
    template<typename A> static void deinterleave(sfloat_v &, sfloat_v &, const unsigned short *, A);

    template<typename A> static void deinterleave(double_v &, double_v &, const double *, A);

    template<typename A> static void deinterleave(int_v &, int_v &, const int *, A);
    template<typename A> static void deinterleave(int_v &, int_v &, const short *, A);

    template<typename A> static void deinterleave(uint_v &, uint_v &, const unsigned int *, A);
    template<typename A> static void deinterleave(uint_v &, uint_v &, const unsigned short *, A);

    template<typename A> static void deinterleave(short_v &, short_v &, const short *, A);

    template<typename A> static void deinterleave(ushort_v &, ushort_v &, const unsigned short *, A);

    static Vc_ALWAYS_INLINE_L void prefetchForOneRead(const void *addr) Vc_ALWAYS_INLINE_R;
    static Vc_ALWAYS_INLINE_L void prefetchForModify(const void *addr) Vc_ALWAYS_INLINE_R;
    static Vc_ALWAYS_INLINE_L void prefetchClose(const void *addr) Vc_ALWAYS_INLINE_R;
    static Vc_ALWAYS_INLINE_L void prefetchMid(const void *addr) Vc_ALWAYS_INLINE_R;
    static Vc_ALWAYS_INLINE_L void prefetchFar(const void *addr) Vc_ALWAYS_INLINE_R;

    template<Vc::MallocAlignment A>
    static Vc_ALWAYS_INLINE_L void *malloc(size_t n) Vc_ALWAYS_INLINE_R;
    static Vc_ALWAYS_INLINE_L void free(void *p) Vc_ALWAYS_INLINE_R;
};

template<> struct HelperImpl<SSE3Impl> : public HelperImpl<SSE2Impl> {};
template<> struct HelperImpl<SSSE3Impl> : public HelperImpl<SSE3Impl> {};
template<> struct HelperImpl<SSE41Impl> : public HelperImpl<SSSE3Impl> {};
template<> struct HelperImpl<SSE42Impl> : public HelperImpl<SSE41Impl> {};


} // namespace Internal
} // namespace Vc
} // namespace ROOT

#include "deinterleave.tcc"
#include "prefetches.tcc"
#include "helperimpl.tcc"
#include "undomacros.h"

#endif // VC_SSE_DEINTERLEAVE_H
