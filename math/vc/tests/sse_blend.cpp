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

#include "unittest.h"
#include <Vc/sse/intrinsics.h>

namespace std
{
ostream &operator<<(ostream &out, const __m128i &v)
{
    union {
        __m128i v;
        short m[8];
    } x = { v };

    out << "[" << x.m[0];
    for (int i = 1; i < 8; ++i) {
        out << ", " << x.m[i];
    }
    return out << "]";
}
} // namespace std

template<> inline bool unittest_compareHelper<__m128i, __m128i>(const __m128i &a, const __m128i &b)
{
    return _mm_movemask_epi8(_mm_cmpeq_epi16(a, b)) == 0xffff;
}

void blendpd()
{
#ifdef VC_IMPL_SSE4_1
#define blend _mm_blend_pd
#else
#define blend Vc::SSE::mm_blend_pd
#endif
    __m128d a = _mm_set_pd(11, 10);
    __m128d b = _mm_set_pd(21, 20);

    COMPARE(_mm_movemask_pd(_mm_cmpeq_pd(blend(a, b, 0x0), a)), 0x3);
    COMPARE(_mm_movemask_pd(_mm_cmpeq_pd(blend(a, b, 0x1), _mm_set_pd(11, 20))), 0x3);
    COMPARE(_mm_movemask_pd(_mm_cmpeq_pd(blend(a, b, 0x2), _mm_set_pd(21, 10))), 0x3);
    COMPARE(_mm_movemask_pd(_mm_cmpeq_pd(blend(a, b, 0x3), b)), 0x3);
#undef blend
}
void blendps()
{
#ifdef VC_IMPL_SSE4_1
#define blend _mm_blend_ps
#else
#define blend Vc::SSE::mm_blend_ps
#endif
    __m128 a = _mm_set_ps(13, 12, 11, 10);
    __m128 b = _mm_set_ps(23, 22, 21, 20);

    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend(a, b, 0x0), a)), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend(a, b, 0x1), _mm_set_ps(13, 12, 11, 20))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend(a, b, 0x2), _mm_set_ps(13, 12, 21, 10))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend(a, b, 0x3), _mm_set_ps(13, 12, 21, 20))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend(a, b, 0x4), _mm_set_ps(13, 22, 11, 10))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend(a, b, 0x5), _mm_set_ps(13, 22, 11, 20))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend(a, b, 0x6), _mm_set_ps(13, 22, 21, 10))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend(a, b, 0x7), _mm_set_ps(13, 22, 21, 20))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend(a, b, 0x8), _mm_set_ps(23, 12, 11, 10))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend(a, b, 0x9), _mm_set_ps(23, 12, 11, 20))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend(a, b, 0xa), _mm_set_ps(23, 12, 21, 10))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend(a, b, 0xb), _mm_set_ps(23, 12, 21, 20))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend(a, b, 0xc), _mm_set_ps(23, 22, 11, 10))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend(a, b, 0xd), _mm_set_ps(23, 22, 11, 20))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend(a, b, 0xe), _mm_set_ps(23, 22, 21, 10))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend(a, b, 0xf), b)), 0xf);
#undef blend
}
void blendepi16()
{
#ifdef VC_IMPL_SSE4_1
#define blend _mm_blend_epi16
#else
#define blend Vc::SSE::mm_blend_epi16
#endif
    __m128i a = _mm_set_epi16(17, 16, 15, 14, 13, 12, 11, 10);
    __m128i b = _mm_set_epi16(27, 26, 25, 24, 23, 22, 21, 20);

#define CALL_2(_i, code) { enum { i = _i }; code } { enum { i = _i + 1 }; code }
#define CALL_4(_i, code) CALL_2(_i, code) CALL_2(_i + 2, code)
#define CALL_8(_i, code) CALL_4(_i, code) CALL_4(_i + 4, code)
#define CALL_16(_i, code) CALL_8(_i, code) CALL_8(_i + 8, code)
#define CALL_32(_i, code) CALL_16(_i, code) CALL_16(_i + 16, code)
#define CALL_64(_i, code) CALL_32(_i, code) CALL_32(_i + 32, code)
#define CALL_128(_i, code) CALL_64(_i, code) CALL_64(_i + 64, code)
#define CALL_256(code) CALL_128(0, code) CALL_128(128, code)
#define CALL_100(code) CALL_64(0, code) CALL_32(64, code) CALL_4(96, code)

    CALL_256(
        short r[8];
        for (int j = 0; j < 8; ++j) {
            r[j] = j + ((((i >> j) & 1) == 0) ? 10 : 20);
        }
        __m128i reference = _mm_set_epi16(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
        COMPARE_NOEQ(blend(a, b, i), reference);
    )
#undef blend
}

int main()
{
    runTest(blendpd);
    runTest(blendps);
    runTest(blendepi16);
}
