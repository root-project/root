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

namespace ROOT {
namespace Vc
{
namespace AVX
{

template<> Vc_ALWAYS_INLINE Mask<4, 32>::Mask(const Mask<8, 32> &m)
    : k(concat(_mm_unpacklo_ps(lo128(m.data()), lo128(m.data())),
                _mm_unpackhi_ps(lo128(m.data()), lo128(m.data()))))
{
}

template<> Vc_ALWAYS_INLINE Mask<8, 32>::Mask(const Mask<4, 32> &m)
    // aabb ccdd -> abcd 0000
    : k(concat(Mem::shuffle<X0, X2, Y0, Y2>(lo128(m.data()), hi128(m.data())),
                _mm_setzero_ps()))
{
}

template<unsigned int Size> Vc_ALWAYS_INLINE Vc_PURE int Mask<Size, 32u>::shiftMask() const
{
    return _mm256_movemask_epi8(dataI());
}
template<unsigned int Size> Vc_ALWAYS_INLINE Vc_PURE int Mask<Size, 16u>::shiftMask() const
{
    return _mm_movemask_epi8(dataI());
}

template<> Vc_ALWAYS_INLINE Vc_PURE int Mask< 4, 32>::toInt() const { return _mm256_movemask_pd(dataD()); }
template<> Vc_ALWAYS_INLINE Vc_PURE int Mask< 8, 32>::toInt() const { return _mm256_movemask_ps(data ()); }
template<> Vc_ALWAYS_INLINE Vc_PURE int Mask< 8, 16>::toInt() const { return _mm_movemask_epi8(_mm_packs_epi16(dataI(), _mm_setzero_si128())); }
template<> Vc_ALWAYS_INLINE Vc_PURE int Mask<16, 16>::toInt() const { return _mm_movemask_epi8(dataI()); }

template<> Vc_ALWAYS_INLINE Vc_PURE bool Mask< 4, 32>::operator[](int index) const { return toInt() & (1 << index); }
template<> Vc_ALWAYS_INLINE Vc_PURE bool Mask< 8, 32>::operator[](int index) const { return toInt() & (1 << index); }
template<> Vc_ALWAYS_INLINE Vc_PURE bool Mask< 8, 16>::operator[](int index) const { return shiftMask() & (1 << 2 * index); }
template<> Vc_ALWAYS_INLINE Vc_PURE bool Mask<16, 16>::operator[](int index) const { return toInt() & (1 << index); }

#ifndef VC_IMPL_POPCNT
static Vc_ALWAYS_INLINE Vc_CONST unsigned int _mm_popcnt_u32(unsigned int n) {
    n = (n & 0x55555555U) + ((n >> 1) & 0x55555555U);
    n = (n & 0x33333333U) + ((n >> 2) & 0x33333333U);
    n = (n & 0x0f0f0f0fU) + ((n >> 4) & 0x0f0f0f0fU);
    //n = (n & 0x00ff00ffU) + ((n >> 8) & 0x00ff00ffU);
    //n = (n & 0x0000ffffU) + ((n >>16) & 0x0000ffffU);
    return n;
}
#endif
template<unsigned int Size> Vc_ALWAYS_INLINE Vc_PURE int Mask<Size, 32u>::count() const { return _mm_popcnt_u32(toInt()); }
template<unsigned int Size> Vc_ALWAYS_INLINE Vc_PURE int Mask<Size, 16u>::count() const { return _mm_popcnt_u32(toInt()); }
template<unsigned int Size> Vc_ALWAYS_INLINE Vc_PURE int Mask<Size, 32u>::firstOne() const { return _bit_scan_forward(toInt()); }
template<unsigned int Size> Vc_ALWAYS_INLINE Vc_PURE int Mask<Size, 16u>::firstOne() const { return _bit_scan_forward(toInt()); }

} // namespace AVX
} // namespace Vc
} // namespace ROOT
