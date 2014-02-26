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

#ifndef VC_AVX_SHUFFLE_H
#define VC_AVX_SHUFFLE_H

#include "../sse/shuffle.h"
#include "macros.h"

namespace ROOT {
namespace Vc
{
        using AVX::m128;
        using AVX::m128d;
        using AVX::m128i;
        using AVX::m256;
        using AVX::m256d;
        using AVX::m256i;
        using AVX::param128;
        using AVX::param128d;
        using AVX::param128i;
        using AVX::param256;
        using AVX::param256d;
        using AVX::param256i;
    namespace Mem
    {
        template<VecPos L, VecPos H> static Vc_ALWAYS_INLINE m256 Vc_CONST permute128(param256 x) {
            VC_STATIC_ASSERT(L >= X0 && L <= X1, Incorrect_Range);
            VC_STATIC_ASSERT(H >= X0 && H <= X1, Incorrect_Range);
            return _mm256_permute2f128_ps(x, x, L + H * (1 << 4));
        }
        template<VecPos L, VecPos H> static Vc_ALWAYS_INLINE m256d Vc_CONST permute128(param256d x) {
            VC_STATIC_ASSERT(L >= X0 && L <= X1, Incorrect_Range);
            VC_STATIC_ASSERT(H >= X0 && H <= X1, Incorrect_Range);
            return _mm256_permute2f128_pd(x, x, L + H * (1 << 4));
        }
        template<VecPos L, VecPos H> static Vc_ALWAYS_INLINE m256i Vc_CONST permute128(param256i x) {
            VC_STATIC_ASSERT(L >= X0 && L <= X1, Incorrect_Range);
            VC_STATIC_ASSERT(H >= X0 && H <= X1, Incorrect_Range);
            return _mm256_permute2f128_si256(x, x, L + H * (1 << 4));
        }
        template<VecPos L, VecPos H> static Vc_ALWAYS_INLINE m256 Vc_CONST shuffle128(param256 x, param256 y) {
            VC_STATIC_ASSERT(L >= X0 && H >= X0, Incorrect_Range);
            VC_STATIC_ASSERT(L <= Y1 && H <= Y1, Incorrect_Range);
            return _mm256_permute2f128_ps(x, y, (L < Y0 ? L : L - Y0 + 2) + (H < Y0 ? H : H - Y0 + 2) * (1 << 4));
        }
        template<VecPos L, VecPos H> static Vc_ALWAYS_INLINE m256i Vc_CONST shuffle128(param256i x, param256i y) {
            VC_STATIC_ASSERT(L >= X0 && H >= X0, Incorrect_Range);
            VC_STATIC_ASSERT(L <= Y1 && H <= Y1, Incorrect_Range);
            return _mm256_permute2f128_si256(x, y, (L < Y0 ? L : L - Y0 + 2) + (H < Y0 ? H : H - Y0 + 2) * (1 << 4));
        }
        template<VecPos L, VecPos H> static Vc_ALWAYS_INLINE m256d Vc_CONST shuffle128(param256d x, param256d y) {
            VC_STATIC_ASSERT(L >= X0 && H >= X0, Incorrect_Range);
            VC_STATIC_ASSERT(L <= Y1 && H <= Y1, Incorrect_Range);
            return _mm256_permute2f128_pd(x, y, (L < Y0 ? L : L - Y0 + 2) + (H < Y0 ? H : H - Y0 + 2) * (1 << 4));
        }
        template<VecPos Dst0, VecPos Dst1, VecPos Dst2, VecPos Dst3> static Vc_ALWAYS_INLINE m256d Vc_CONST permute(param256d x) {
            VC_STATIC_ASSERT(Dst0 >= X0 && Dst1 >= X0 && Dst2 >= X2 && Dst3 >= X2, Incorrect_Range);
            VC_STATIC_ASSERT(Dst0 <= X1 && Dst1 <= X1 && Dst2 <= X3 && Dst3 <= X3, Incorrect_Range);
            return _mm256_permute_pd(x, Dst0 + Dst1 * 2 + (Dst2 - X2) * 4 + (Dst3 - X2) * 8);
        }
        template<VecPos Dst0, VecPos Dst1, VecPos Dst2, VecPos Dst3> static Vc_ALWAYS_INLINE m256 Vc_CONST permute(param256 x) {
            VC_STATIC_ASSERT(Dst0 >= X0 && Dst1 >= X0 && Dst2 >= X0 && Dst3 >= X0, Incorrect_Range);
            VC_STATIC_ASSERT(Dst0 <= X3 && Dst1 <= X3 && Dst2 <= X3 && Dst3 <= X3, Incorrect_Range);
            return _mm256_permute_ps(x, Dst0 + Dst1 * 4 + Dst2 * 16 + Dst3 * 64);
        }
        template<VecPos Dst0, VecPos Dst1, VecPos Dst2, VecPos Dst3> static Vc_ALWAYS_INLINE m256i Vc_CONST permute(param256i x) {
            return _mm256_castps_si256(permute<Dst0, Dst1, Dst2, Dst3>(_mm256_castsi256_ps(x)));
        }
        template<VecPos Dst0, VecPos Dst1, VecPos Dst2, VecPos Dst3> static Vc_ALWAYS_INLINE m256d Vc_CONST shuffle(param256d x, param256d y) {
            VC_STATIC_ASSERT(Dst0 >= X0 && Dst1 >= Y0 && Dst2 >= X2 && Dst3 >= Y2, Incorrect_Range);
            VC_STATIC_ASSERT(Dst0 <= X1 && Dst1 <= Y1 && Dst2 <= X3 && Dst3 <= Y3, Incorrect_Range);
            return _mm256_shuffle_pd(x, y, Dst0 + (Dst1 - Y0) * 2 + (Dst2 - X2) * 4 + (Dst3 - Y2) * 8);
        }
        template<VecPos Dst0, VecPos Dst1, VecPos Dst2, VecPos Dst3> static Vc_ALWAYS_INLINE m256 Vc_CONST shuffle(param256 x, param256 y) {
            VC_STATIC_ASSERT(Dst0 >= X0 && Dst1 >= X0 && Dst2 >= Y0 && Dst3 >= Y0, Incorrect_Range);
            VC_STATIC_ASSERT(Dst0 <= X3 && Dst1 <= X3 && Dst2 <= Y3 && Dst3 <= Y3, Incorrect_Range);
            return _mm256_shuffle_ps(x, y, Dst0 + Dst1 * 4 + (Dst2 - Y0) * 16 + (Dst3 - Y0) * 64);
        }
        template<VecPos Dst0, VecPos Dst1, VecPos Dst2, VecPos Dst3, VecPos Dst4, VecPos Dst5, VecPos Dst6, VecPos Dst7>
        static Vc_ALWAYS_INLINE m256 Vc_CONST blend(param256 x, param256 y) {
            VC_STATIC_ASSERT(Dst0 == X0 || Dst0 == Y0, Incorrect_Range);
            VC_STATIC_ASSERT(Dst1 == X1 || Dst1 == Y1, Incorrect_Range);
            VC_STATIC_ASSERT(Dst2 == X2 || Dst2 == Y2, Incorrect_Range);
            VC_STATIC_ASSERT(Dst3 == X3 || Dst3 == Y3, Incorrect_Range);
            VC_STATIC_ASSERT(Dst4 == X4 || Dst4 == Y4, Incorrect_Range);
            VC_STATIC_ASSERT(Dst5 == X5 || Dst5 == Y5, Incorrect_Range);
            VC_STATIC_ASSERT(Dst6 == X6 || Dst6 == Y6, Incorrect_Range);
            VC_STATIC_ASSERT(Dst7 == X7 || Dst7 == Y7, Incorrect_Range);
            return _mm256_blend_ps(x, y,
                    (Dst0 / Y0) *  1 + (Dst1 / Y1) *  2 +
                    (Dst2 / Y2) *  4 + (Dst3 / Y3) *  8 +
                    (Dst4 / Y4) * 16 + (Dst5 / Y5) * 32 +
                    (Dst6 / Y6) * 64 + (Dst7 / Y7) *128
                    );
        }
        template<VecPos Dst0, VecPos Dst1, VecPos Dst2, VecPos Dst3, VecPos Dst4, VecPos Dst5, VecPos Dst6, VecPos Dst7>
        static Vc_ALWAYS_INLINE m256i Vc_CONST blend(param256i x, param256i y) {
            return _mm256_castps_si256(blend<Dst0, Dst1, Dst2, Dst3, Dst4, Dst5, Dst6, Dst7>(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y)));
        }
        template<VecPos Dst> struct ScaleForBlend { enum { Value = Dst >= X4 ? Dst - X4 + Y0 : Dst }; };
        template<VecPos Dst0, VecPos Dst1, VecPos Dst2, VecPos Dst3, VecPos Dst4, VecPos Dst5, VecPos Dst6, VecPos Dst7>
        static Vc_ALWAYS_INLINE m256 Vc_CONST permute(param256 x) {
            VC_STATIC_ASSERT(Dst0 >= X0 && Dst0 <= X7, Incorrect_Range);
            VC_STATIC_ASSERT(Dst1 >= X0 && Dst1 <= X7, Incorrect_Range);
            VC_STATIC_ASSERT(Dst2 >= X0 && Dst2 <= X7, Incorrect_Range);
            VC_STATIC_ASSERT(Dst3 >= X0 && Dst3 <= X7, Incorrect_Range);
            VC_STATIC_ASSERT(Dst4 >= X0 && Dst4 <= X7, Incorrect_Range);
            VC_STATIC_ASSERT(Dst5 >= X0 && Dst5 <= X7, Incorrect_Range);
            VC_STATIC_ASSERT(Dst6 >= X0 && Dst6 <= X7, Incorrect_Range);
            VC_STATIC_ASSERT(Dst7 >= X0 && Dst7 <= X7, Incorrect_Range);
            if (Dst0 + X4 == Dst4 && Dst1 + X4 == Dst5 && Dst2 + X4 == Dst6 && Dst3 + X4 == Dst7) {
                return permute<Dst0, Dst1, Dst2, Dst3>(x);
            }
            const m128 loIn = _mm256_castps256_ps128(x);
            const m128 hiIn = _mm256_extractf128_ps(x, 1);
            m128 lo, hi;

            if (Dst0 < X4 && Dst1 < X4 && Dst2 < X4 && Dst3 < X4) {
                lo = _mm_permute_ps(loIn, Dst0 + Dst1 * 4 + Dst2 * 16 + Dst3 * 64);
            } else if (Dst0 >= X4 && Dst1 >= X4 && Dst2 >= X4 && Dst3 >= X4) {
                lo = _mm_permute_ps(hiIn, Dst0 + Dst1 * 4 + Dst2 * 16 + Dst3 * 64);
            } else if (Dst0 < X4 && Dst1 < X4 && Dst2 >= X4 && Dst3 >= X4) {
                lo = shuffle<Dst0, Dst1, Dst2 - X4 + Y0, Dst3 - X4 + Y0>(loIn, hiIn);
            } else if (Dst0 >= X4 && Dst1 >= X4 && Dst2 < X4 && Dst3 < X4) {
                lo = shuffle<Dst0 - X4, Dst1 - X4, Dst2 + Y0, Dst3 + Y0>(hiIn, loIn);
            } else if (Dst0 == X0 && Dst1 == X4 && Dst2 == X1 && Dst3 == X5) {
                lo = _mm_unpacklo_ps(loIn, hiIn);
            } else if (Dst0 == X4 && Dst1 == X0 && Dst2 == X5 && Dst3 == X1) {
                lo = _mm_unpacklo_ps(hiIn, loIn);
            } else if (Dst0 == X2 && Dst1 == X6 && Dst2 == X3 && Dst3 == X7) {
                lo = _mm_unpackhi_ps(loIn, hiIn);
            } else if (Dst0 == X6 && Dst1 == X2 && Dst2 == X7 && Dst3 == X3) {
                lo = _mm_unpackhi_ps(hiIn, loIn);
            } else if (Dst0 % X4 == 0 && Dst1 % X4 == 1 && Dst2 % X4 == 2 && Dst3 % X4 == 3) {
                lo = blend<ScaleForBlend<Dst0>::Value, ScaleForBlend<Dst1>::Value,
                   ScaleForBlend<Dst2>::Value, ScaleForBlend<Dst3>::Value>(loIn, hiIn);
            }

            if (Dst4 >= X4 && Dst5 >= X4 && Dst6 >= X4 && Dst7 >= X4) {
                hi = _mm_permute_ps(hiIn, (Dst4 - X4) + (Dst5 - X4) * 4 + (Dst6 - X4) * 16 + (Dst7 - X4) * 64);
            } else if (Dst4 < X4 && Dst5 < X4 && Dst6 < X4 && Dst7 < X4) {
                hi = _mm_permute_ps(loIn, (Dst4 - X4) + (Dst5 - X4) * 4 + (Dst6 - X4) * 16 + (Dst7 - X4) * 64);
            } else if (Dst4 < X4 && Dst5 < X4 && Dst6 >= X4 && Dst7 >= X4) {
                hi = shuffle<Dst4, Dst5, Dst6 - X4 + Y0, Dst7 - X4 + Y0>(loIn, hiIn);
            } else if (Dst4 >= X4 && Dst5 >= X4 && Dst6 < X4 && Dst7 < X4) {
                hi = shuffle<Dst4 - X4, Dst5 - X4, Dst6 + Y0, Dst7 + Y0>(hiIn, loIn);
            } else if (Dst4 == X0 && Dst5 == X4 && Dst6 == X1 && Dst7 == X5) {
                hi = _mm_unpacklo_ps(loIn, hiIn);
            } else if (Dst4 == X4 && Dst5 == X0 && Dst6 == X5 && Dst7 == X1) {
                hi = _mm_unpacklo_ps(hiIn, loIn);
            } else if (Dst4 == X2 && Dst5 == X6 && Dst6 == X3 && Dst7 == X7) {
                hi = _mm_unpackhi_ps(loIn, hiIn);
            } else if (Dst4 == X6 && Dst5 == X2 && Dst6 == X7 && Dst7 == X3) {
                hi = _mm_unpackhi_ps(hiIn, loIn);
            } else if (Dst4 % X4 == 0 && Dst5 % X4 == 1 && Dst6 % X4 == 2 && Dst7 % X4 == 3) {
                hi = blend<ScaleForBlend<Dst4>::Value, ScaleForBlend<Dst5>::Value,
                   ScaleForBlend<Dst6>::Value, ScaleForBlend<Dst7>::Value>(loIn, hiIn);
            }

            return _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
        }
    } // namespace Mem

    // little endian has the lo bits on the right and high bits on the left
    // with vectors this becomes greatly confusing:
    // Mem: abcd
    // Reg: dcba
    //
    // The shuffles and permutes above use memory ordering. The ones below use register ordering:
    namespace Reg
    {
        template<VecPos H, VecPos L> static Vc_ALWAYS_INLINE m256 Vc_CONST permute128(param256 x, param256 y) {
            VC_STATIC_ASSERT(L >= X0 && H >= X0, Incorrect_Range);
            VC_STATIC_ASSERT(L <= Y1 && H <= Y1, Incorrect_Range);
            return _mm256_permute2f128_ps(x, y, (L < Y0 ? L : L - Y0 + 2) + (H < Y0 ? H : H - Y0 + 2) * (1 << 4));
        }
        template<VecPos H, VecPos L> static Vc_ALWAYS_INLINE m256i Vc_CONST permute128(param256i x, param256i y) {
            VC_STATIC_ASSERT(L >= X0 && H >= X0, Incorrect_Range);
            VC_STATIC_ASSERT(L <= Y1 && H <= Y1, Incorrect_Range);
            return _mm256_permute2f128_si256(x, y, (L < Y0 ? L : L - Y0 + 2) + (H < Y0 ? H : H - Y0 + 2) * (1 << 4));
        }
        template<VecPos H, VecPos L> static Vc_ALWAYS_INLINE m256d Vc_CONST permute128(param256d x, param256d y) {
            VC_STATIC_ASSERT(L >= X0 && H >= X0, Incorrect_Range);
            VC_STATIC_ASSERT(L <= Y1 && H <= Y1, Incorrect_Range);
            return _mm256_permute2f128_pd(x, y, (L < Y0 ? L : L - Y0 + 2) + (H < Y0 ? H : H - Y0 + 2) * (1 << 4));
        }
        template<VecPos Dst3, VecPos Dst2, VecPos Dst1, VecPos Dst0> static Vc_ALWAYS_INLINE m256d Vc_CONST permute(param256d x) {
            VC_STATIC_ASSERT(Dst0 >= X0 && Dst1 >= X0 && Dst2 >= X2 && Dst3 >= X2, Incorrect_Range);
            VC_STATIC_ASSERT(Dst0 <= X1 && Dst1 <= X1 && Dst2 <= X3 && Dst3 <= X3, Incorrect_Range);
            return _mm256_permute_pd(x, Dst0 + Dst1 * 2 + (Dst2 - X2) * 4 + (Dst3 - X2) * 8);
        }
        template<VecPos Dst3, VecPos Dst2, VecPos Dst1, VecPos Dst0> static Vc_ALWAYS_INLINE m256 Vc_CONST permute(param256 x) {
            VC_STATIC_ASSERT(Dst0 >= X0 && Dst1 >= X0 && Dst2 >= X0 && Dst3 >= X0, Incorrect_Range);
            VC_STATIC_ASSERT(Dst0 <= X3 && Dst1 <= X3 && Dst2 <= X3 && Dst3 <= X3, Incorrect_Range);
            return _mm256_permute_ps(x, Dst0 + Dst1 * 4 + Dst2 * 16 + Dst3 * 64);
        }
        template<VecPos Dst1, VecPos Dst0> static Vc_ALWAYS_INLINE m128d Vc_CONST permute(param128d x) {
            VC_STATIC_ASSERT(Dst0 >= X0 && Dst1 >= X0, Incorrect_Range);
            VC_STATIC_ASSERT(Dst0 <= X1 && Dst1 <= X1, Incorrect_Range);
            return _mm_permute_pd(x, Dst0 + Dst1 * 2);
        }
        template<VecPos Dst3, VecPos Dst2, VecPos Dst1, VecPos Dst0> static Vc_ALWAYS_INLINE m128 Vc_CONST permute(param128 x) {
            VC_STATIC_ASSERT(Dst0 >= X0 && Dst1 >= X0 && Dst2 >= X0 && Dst3 >= X0, Incorrect_Range);
            VC_STATIC_ASSERT(Dst0 <= X3 && Dst1 <= X3 && Dst2 <= X3 && Dst3 <= X3, Incorrect_Range);
            return _mm_permute_ps(x, Dst0 + Dst1 * 4 + Dst2 * 16 + Dst3 * 64);
        }
        template<VecPos Dst3, VecPos Dst2, VecPos Dst1, VecPos Dst0> static Vc_ALWAYS_INLINE m256d Vc_CONST shuffle(param256d x, param256d y) {
            VC_STATIC_ASSERT(Dst0 >= X0 && Dst1 >= Y0 && Dst2 >= X2 && Dst3 >= Y2, Incorrect_Range);
            VC_STATIC_ASSERT(Dst0 <= X1 && Dst1 <= Y1 && Dst2 <= X3 && Dst3 <= Y3, Incorrect_Range);
            return _mm256_shuffle_pd(x, y, Dst0 + (Dst1 - Y0) * 2 + (Dst2 - X2) * 4 + (Dst3 - Y2) * 8);
        }
        template<VecPos Dst3, VecPos Dst2, VecPos Dst1, VecPos Dst0> static Vc_ALWAYS_INLINE m256 Vc_CONST shuffle(param256 x, param256 y) {
            VC_STATIC_ASSERT(Dst0 >= X0 && Dst1 >= X0 && Dst2 >= Y0 && Dst3 >= Y0, Incorrect_Range);
            VC_STATIC_ASSERT(Dst0 <= X3 && Dst1 <= X3 && Dst2 <= Y3 && Dst3 <= Y3, Incorrect_Range);
            return _mm256_shuffle_ps(x, y, Dst0 + Dst1 * 4 + (Dst2 - Y0) * 16 + (Dst3 - Y0) * 64);
        }
    } // namespace Reg
} // namespace Vc
} // namespace ROOT
#include "undomacros.h"

#endif // VC_AVX_SHUFFLE_H
