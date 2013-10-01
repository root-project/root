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

namespace ROOT {
namespace Vc
{
namespace SSE
{

inline void deinterleave(Vector<float> &a, Vector<float> &b)
{
    const _M128 tmp0 = _mm_unpacklo_ps(a.data(), b.data());
    const _M128 tmp1 = _mm_unpackhi_ps(a.data(), b.data());
    a.data() = _mm_unpacklo_ps(tmp0, tmp1);
    b.data() = _mm_unpackhi_ps(tmp0, tmp1);
}

inline void deinterleave(Vector<float> &a, Vector<float> &b, Vector<short>::AsArg tmp)
{
    a.data() = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_slli_epi32(tmp.data(), 16), 16));
    b.data() = _mm_cvtepi32_ps(_mm_srai_epi32(tmp.data(), 16));
}

inline void deinterleave(Vector<float> &a, Vector<float> &b, Vector<unsigned short>::AsArg tmp)
{
    a.data() = _mm_cvtepi32_ps(_mm_srli_epi32(_mm_slli_epi32(tmp.data(), 16), 16));
    b.data() = _mm_cvtepi32_ps(_mm_srli_epi32(tmp.data(), 16));
}

inline void deinterleave(Vector<float8> &a, Vector<float8> &b)
{
    _M128 tmp0 = _mm_unpacklo_ps(a.data()[0], a.data()[1]);
    _M128 tmp1 = _mm_unpackhi_ps(a.data()[0], a.data()[1]);
    _M128 tmp2 = _mm_unpacklo_ps(b.data()[0], b.data()[1]);
    _M128 tmp3 = _mm_unpackhi_ps(b.data()[0], b.data()[1]);
    a.data()[0] = _mm_unpacklo_ps(tmp0, tmp1);
    b.data()[0] = _mm_unpackhi_ps(tmp0, tmp1);
    a.data()[1] = _mm_unpacklo_ps(tmp2, tmp3);
    b.data()[1] = _mm_unpackhi_ps(tmp2, tmp3);
}

inline void deinterleave(Vector<float8> &a, Vector<float8> &b, Vector<short>::AsArg tmp0, Vector<short>::AsArg tmp1)
{
    a.data()[0] = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_slli_epi32(tmp0.data(), 16), 16));
    b.data()[0] = _mm_cvtepi32_ps(_mm_srai_epi32(tmp0.data(), 16));
    a.data()[1] = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_slli_epi32(tmp1.data(), 16), 16));
    b.data()[1] = _mm_cvtepi32_ps(_mm_srai_epi32(tmp1.data(), 16));
}

inline void deinterleave(Vector<float8> &a, Vector<float8> &b, Vector<unsigned short>::AsArg tmp0, Vector<unsigned short>::AsArg tmp1)
{
    a.data()[0] = _mm_cvtepi32_ps(_mm_srli_epi32(_mm_slli_epi32(tmp0.data(), 16), 16));
    b.data()[0] = _mm_cvtepi32_ps(_mm_srli_epi32(tmp0.data(), 16));
    a.data()[1] = _mm_cvtepi32_ps(_mm_srli_epi32(_mm_slli_epi32(tmp1.data(), 16), 16));
    b.data()[1] = _mm_cvtepi32_ps(_mm_srli_epi32(tmp1.data(), 16));
}

inline void deinterleave(Vector<double> &a, Vector<double> &b)
{
    _M128D tmp = _mm_unpacklo_pd(a.data(), b.data());
    b.data() = _mm_unpackhi_pd(a.data(), b.data());
    a.data() = tmp;
}

inline void deinterleave(Vector<int> &a, Vector<int> &b)
{
    const _M128I tmp0 = _mm_unpacklo_epi32(a.data(), b.data());
    const _M128I tmp1 = _mm_unpackhi_epi32(a.data(), b.data());
    a.data() = _mm_unpacklo_epi32(tmp0, tmp1);
    b.data() = _mm_unpackhi_epi32(tmp0, tmp1);
}

inline void deinterleave(Vector<unsigned int> &a, Vector<unsigned int> &b)
{
    const _M128I tmp0 = _mm_unpacklo_epi32(a.data(), b.data());
    const _M128I tmp1 = _mm_unpackhi_epi32(a.data(), b.data());
    a.data() = _mm_unpacklo_epi32(tmp0, tmp1);
    b.data() = _mm_unpackhi_epi32(tmp0, tmp1);
}

inline void deinterleave(Vector<short> &a, Vector<short> &b)
{
    _M128I tmp0 = _mm_unpacklo_epi16(a.data(), b.data()); // a0 a4 b0 b4 a1 a5 b1 b5
    _M128I tmp1 = _mm_unpackhi_epi16(a.data(), b.data()); // a2 a6 b2 b6 a3 a7 b3 b7
    _M128I tmp2 = _mm_unpacklo_epi16(tmp0, tmp1); // a0 a2 a4 a6 b0 b2 b4 b6
    _M128I tmp3 = _mm_unpackhi_epi16(tmp0, tmp1); // a1 a3 a5 a7 b1 b3 b5 b7
    a.data() = _mm_unpacklo_epi16(tmp2, tmp3);
    b.data() = _mm_unpackhi_epi16(tmp2, tmp3);
}

inline void deinterleave(Vector<unsigned short> &a, Vector<unsigned short> &b)
{
    _M128I tmp0 = _mm_unpacklo_epi16(a.data(), b.data()); // a0 a4 b0 b4 a1 a5 b1 b5
    _M128I tmp1 = _mm_unpackhi_epi16(a.data(), b.data()); // a2 a6 b2 b6 a3 a7 b3 b7
    _M128I tmp2 = _mm_unpacklo_epi16(tmp0, tmp1); // a0 a2 a4 a6 b0 b2 b4 b6
    _M128I tmp3 = _mm_unpackhi_epi16(tmp0, tmp1); // a1 a3 a5 a7 b1 b3 b5 b7
    a.data() = _mm_unpacklo_epi16(tmp2, tmp3);
    b.data() = _mm_unpackhi_epi16(tmp2, tmp3);
}

inline void deinterleave(Vector<int> &a, Vector<int> &b, Vector<short>::AsArg tmp)
{
    a.data() = _mm_srai_epi32(_mm_slli_epi32(tmp.data(), 16), 16);
    b.data() = _mm_srai_epi32(tmp.data(), 16);
}

inline void deinterleave(Vector<unsigned int> &a, Vector<unsigned int> &b, Vector<unsigned short>::AsArg tmp)
{
    a.data() = _mm_srli_epi32(_mm_slli_epi32(tmp.data(), 16), 16);
    b.data() = _mm_srli_epi32(tmp.data(), 16);
}

} // namespace SSE


namespace Internal
{

template<typename A> inline void HelperImpl<Vc::SSE2Impl>::deinterleave(
        float_v &a, float_v &b, const float *m, A align)
{
    a.load(m, align);
    b.load(m + float_v::Size, align);
    Vc::SSE::deinterleave(a, b);
}

template<typename A> inline void HelperImpl<Vc::SSE2Impl>::deinterleave(
        float_v &a, float_v &b, const short *m, A align)
{
    short_v tmp(m, align);
    Vc::SSE::deinterleave(a, b, tmp);
}

template<typename A> inline void HelperImpl<Vc::SSE2Impl>::deinterleave(
        float_v &a, float_v &b, const unsigned short *m, A align)
{
    ushort_v tmp(m, align);
    Vc::SSE::deinterleave(a, b, tmp);
}

template<typename A> inline void HelperImpl<Vc::SSE2Impl>::deinterleave(
        sfloat_v &a, sfloat_v &b, const float *m, A align)
{
    a.load(m, align);
    b.load(m + sfloat_v::Size, align);
    Vc::SSE::deinterleave(a, b);
}

template<typename A> inline void HelperImpl<Vc::SSE2Impl>::deinterleave(
        sfloat_v &a, sfloat_v &b, const short *m, A align)
{
    short_v tmp0(m, align);
    short_v tmp1(m + short_v::Size, align);
    Vc::SSE::deinterleave(a, b, tmp0, tmp1);
}

template<typename A> inline void HelperImpl<Vc::SSE2Impl>::deinterleave(
        sfloat_v &a, sfloat_v &b, const unsigned short *m, A align)
{
    ushort_v tmp0(m, align);
    ushort_v tmp1(m + short_v::Size, align);
    Vc::SSE::deinterleave(a, b, tmp0, tmp1);
}

template<typename A> inline void HelperImpl<Vc::SSE2Impl>::deinterleave(
        double_v &a, double_v &b, const double *m, A align)
{
    a.load(m, align);
    b.load(m + double_v::Size, align);
    Vc::SSE::deinterleave(a, b);
}

template<typename A> inline void HelperImpl<Vc::SSE2Impl>::deinterleave(
        int_v &a, int_v &b, const int *m, A align)
{
    a.load(m, align);
    b.load(m + int_v::Size, align);
    Vc::SSE::deinterleave(a, b);
}

template<typename A> inline void HelperImpl<Vc::SSE2Impl>::deinterleave(
        int_v &a, int_v &b, const short *m, A align)
{
    short_v tmp(m, align);
    Vc::SSE::deinterleave(a, b, tmp);
}

template<typename A> inline void HelperImpl<Vc::SSE2Impl>::deinterleave(
        uint_v &a, uint_v &b, const unsigned int *m, A align)
{
    a.load(m, align);
    b.load(m + uint_v::Size, align);
    Vc::SSE::deinterleave(a, b);
}

template<typename A> inline void HelperImpl<Vc::SSE2Impl>::deinterleave(
        uint_v &a, uint_v &b, const unsigned short *m, A align)
{
    ushort_v tmp(m, align);
    Vc::SSE::deinterleave(a, b, tmp);
}

template<typename A> inline void HelperImpl<Vc::SSE2Impl>::deinterleave(
        short_v &a, short_v &b, const short *m, A align)
{
    a.load(m, align);
    b.load(m + short_v::Size, align);
    Vc::SSE::deinterleave(a, b);
}

template<typename A> inline void HelperImpl<Vc::SSE2Impl>::deinterleave(
        ushort_v &a, ushort_v &b, const unsigned short *m, A align)
{
    a.load(m, align);
    b.load(m + ushort_v::Size, align);
    Vc::SSE::deinterleave(a, b);
}

} // namespace Internal
} // namespace Vc
} // namespace ROOT
