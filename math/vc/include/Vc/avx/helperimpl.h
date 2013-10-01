/*  This file is part of the Vc library.

    Copyright (C) 2011-2012 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_AVX_HELPERIMPL_H
#define VC_AVX_HELPERIMPL_H

#include "macros.h"

namespace ROOT {
namespace Vc
{
namespace Internal
{

template<> struct HelperImpl<Vc::AVXImpl>
{
    typedef AVX::Vector<float> float_v;
    typedef AVX::Vector<sfloat> sfloat_v;
    typedef AVX::Vector<double> double_v;
    typedef AVX::Vector<int> int_v;
    typedef AVX::Vector<unsigned int> uint_v;
    typedef AVX::Vector<short> short_v;
    typedef AVX::Vector<unsigned short> ushort_v;

    template<typename A> static void deinterleave(float_v &, float_v &, const float *, A);
    template<typename A> static void deinterleave(float_v &, float_v &, const short *, A);
    template<typename A> static void deinterleave(float_v &, float_v &, const unsigned short *, A);

    template<typename A, typename MemT> static void deinterleave(sfloat_v &, sfloat_v &, const MemT *, A);

    template<typename A> static void deinterleave(double_v &, double_v &, const double *, A);

    template<typename A> static void deinterleave(int_v &, int_v &, const int *, A);
    template<typename A> static void deinterleave(int_v &, int_v &, const short *, A);

    template<typename A> static void deinterleave(uint_v &, uint_v &, const unsigned int *, A);
    template<typename A> static void deinterleave(uint_v &, uint_v &, const unsigned short *, A);

    template<typename A> static void deinterleave(short_v &, short_v &, const short *, A);

    template<typename A> static void deinterleave(ushort_v &, ushort_v &, const unsigned short *, A);

    template<typename V, typename M, typename A>
        static Vc_ALWAYS_INLINE_L void deinterleave(V &VC_RESTRICT a, V &VC_RESTRICT b,
                V &VC_RESTRICT c, const M *VC_RESTRICT memory, A align) Vc_ALWAYS_INLINE_R;

    template<typename V, typename M, typename A>
        static Vc_ALWAYS_INLINE_L void deinterleave(V &VC_RESTRICT a, V &VC_RESTRICT b,
                V &VC_RESTRICT c, V &VC_RESTRICT d,
                const M *VC_RESTRICT memory, A align) Vc_ALWAYS_INLINE_R;

    template<typename V, typename M, typename A>
        static Vc_ALWAYS_INLINE_L void deinterleave(V &VC_RESTRICT a, V &VC_RESTRICT b,
                V &VC_RESTRICT c, V &VC_RESTRICT d, V &VC_RESTRICT e,
                const M *VC_RESTRICT memory, A align) Vc_ALWAYS_INLINE_R;

    template<typename V, typename M, typename A>
        static Vc_ALWAYS_INLINE_L void deinterleave(V &VC_RESTRICT a, V &VC_RESTRICT b,
                V &VC_RESTRICT c, V &VC_RESTRICT d, V &VC_RESTRICT e,
                V &VC_RESTRICT f, const M *VC_RESTRICT memory, A align) Vc_ALWAYS_INLINE_R;

    template<typename V, typename M, typename A>
        static Vc_ALWAYS_INLINE_L void deinterleave(V &VC_RESTRICT a, V &VC_RESTRICT b,
                V &VC_RESTRICT c, V &VC_RESTRICT d, V &VC_RESTRICT e,
                V &VC_RESTRICT f, V &VC_RESTRICT g, V &VC_RESTRICT h,
                const M *VC_RESTRICT memory, A align) Vc_ALWAYS_INLINE_R;

    static Vc_ALWAYS_INLINE_L void prefetchForOneRead(const void *addr) Vc_ALWAYS_INLINE_R;
    static Vc_ALWAYS_INLINE_L void prefetchForModify(const void *addr) Vc_ALWAYS_INLINE_R;
    static Vc_ALWAYS_INLINE_L void prefetchClose(const void *addr) Vc_ALWAYS_INLINE_R;
    static Vc_ALWAYS_INLINE_L void prefetchMid(const void *addr) Vc_ALWAYS_INLINE_R;
    static Vc_ALWAYS_INLINE_L void prefetchFar(const void *addr) Vc_ALWAYS_INLINE_R;

    template<Vc::MallocAlignment A>
    static Vc_ALWAYS_INLINE_L void *malloc(size_t n) Vc_ALWAYS_INLINE_R;
    static Vc_ALWAYS_INLINE_L void free(void *p) Vc_ALWAYS_INLINE_R;
};

} // namespace Internal
} // namespace Vc
} // namespace ROOT

#include "deinterleave.tcc"
#include "prefetches.tcc"
#include "helperimpl.tcc"
#include "undomacros.h"

#endif // VC_AVX_HELPERIMPL_H
