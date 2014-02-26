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

#ifndef VC_SSE_CONST_H
#define VC_SSE_CONST_H

#include "const_data.h"
#include "vector.h"
#include "macros.h"

namespace ROOT {
namespace Vc
{
namespace SSE
{
    template<typename T> class Vector;

    template<typename _T> struct Const
    {
        typedef Vector<_T> V;
        typedef typename V::EntryType T;
        typedef typename V::Mask M;
        enum Constants { Stride = 16 / sizeof(T) };

        static Vc_ALWAYS_INLINE Vc_CONST V _pi_4()        { return load(&c_trig<T>::data[0 * Stride]); }
        static Vc_ALWAYS_INLINE Vc_CONST V _pi_4_hi()     { return load(&c_trig<T>::data[1 * Stride]); }
        static Vc_ALWAYS_INLINE Vc_CONST V _pi_4_rem1()   { return load(&c_trig<T>::data[2 * Stride]); }
        static Vc_ALWAYS_INLINE Vc_CONST V _pi_4_rem2()   { return load(&c_trig<T>::data[3 * Stride]); }
        static Vc_ALWAYS_INLINE Vc_CONST V _1_16()        { return load(&c_trig<T>::data[4 * Stride]); }
        static Vc_ALWAYS_INLINE Vc_CONST V _16()          { return load(&c_trig<T>::data[5 * Stride]); }

        static Vc_ALWAYS_INLINE Vc_CONST V cosCoeff(int i) { return load(&c_trig<T>::data[( 8 + i) * Stride]); }
        static Vc_ALWAYS_INLINE Vc_CONST V sinCoeff(int i) { return load(&c_trig<T>::data[(14 + i) * Stride]); }
        static Vc_ALWAYS_INLINE Vc_CONST V atanP(int i)    { return load(&c_trig<T>::data[(24 + i) * Stride]); }
        static Vc_ALWAYS_INLINE Vc_CONST V atanQ(int i)    { return load(&c_trig<T>::data[(29 + i) * Stride]); }
        static Vc_ALWAYS_INLINE Vc_CONST V atanThrsHi()    { return load(&c_trig<T>::data[34 * Stride]); }
        static Vc_ALWAYS_INLINE Vc_CONST V atanThrsLo()    { return load(&c_trig<T>::data[35 * Stride]); }
        static Vc_ALWAYS_INLINE Vc_CONST V _pi_2_rem()     { return load(&c_trig<T>::data[36 * Stride]); }
        static Vc_ALWAYS_INLINE Vc_CONST V lossThreshold() { return load(&c_trig<T>::data[20 * Stride]); }
        static Vc_ALWAYS_INLINE Vc_CONST V _4_pi()         { return load(&c_trig<T>::data[21 * Stride]); }
        static Vc_ALWAYS_INLINE Vc_CONST V _pi_2()         { return load(&c_trig<T>::data[22 * Stride]); }
        static Vc_ALWAYS_INLINE Vc_CONST V _pi()           { return load(&c_trig<T>::data[23 * Stride]); }
        static Vc_ALWAYS_INLINE Vc_CONST V asinCoeff0(int i) { return load(&c_trig<T>::data[(40 + i) * Stride]); }
        static Vc_ALWAYS_INLINE Vc_CONST V asinCoeff1(int i) { return load(&c_trig<T>::data[(45 + i) * Stride]); }
        static Vc_ALWAYS_INLINE Vc_CONST V asinCoeff2(int i) { return load(&c_trig<T>::data[(49 + i) * Stride]); }
        static Vc_ALWAYS_INLINE Vc_CONST V asinCoeff3(int i) { return load(&c_trig<T>::data[(55 + i) * Stride]); }
        static Vc_ALWAYS_INLINE Vc_CONST V smallAsinInput()  { return load(&c_trig<T>::data[37 * Stride]); }
        static Vc_ALWAYS_INLINE Vc_CONST V largeAsinInput()  { return load(&c_trig<T>::data[38 * Stride]); }

        static Vc_ALWAYS_INLINE Vc_CONST M exponentMask() { return M(load(c_log<T>::d(1)).data()); }
        static Vc_ALWAYS_INLINE Vc_CONST V _1_2()         { return load(c_log<T>::d(18)); }
        static Vc_ALWAYS_INLINE Vc_CONST V _1_sqrt2()     { return load(c_log<T>::d(15)); }
        static Vc_ALWAYS_INLINE Vc_CONST V P(int i)       { return load(c_log<T>::d(2 + i)); }
        static Vc_ALWAYS_INLINE Vc_CONST V Q(int i)       { return load(c_log<T>::d(8 + i)); }
        static Vc_ALWAYS_INLINE Vc_CONST V min()          { return load(c_log<T>::d(14)); }
        static Vc_ALWAYS_INLINE Vc_CONST V ln2_small()    { return load(c_log<T>::d(17)); }
        static Vc_ALWAYS_INLINE Vc_CONST V ln2_large()    { return load(c_log<T>::d(16)); }
        static Vc_ALWAYS_INLINE Vc_CONST V neginf()       { return load(c_log<T>::d(13)); }
        static Vc_ALWAYS_INLINE Vc_CONST V log10_e()      { return load(c_log<T>::d(19)); }
        static Vc_ALWAYS_INLINE Vc_CONST V log2_e()       { return load(c_log<T>::d(20)); }

        static Vc_ALWAYS_INLINE_L Vc_CONST_L V highMask()         Vc_ALWAYS_INLINE_R Vc_CONST_R;
        static Vc_ALWAYS_INLINE_L Vc_CONST_L V highMask(int bits) Vc_ALWAYS_INLINE_R Vc_CONST_R;
    private:
        static Vc_ALWAYS_INLINE_L Vc_CONST_L V load(const T *mem) Vc_ALWAYS_INLINE_R Vc_CONST_R;
    };
    template<typename T> Vc_ALWAYS_INLINE Vc_CONST Vector<T> Const<T>::load(const T *mem) { return V(mem); }
    template<> Vc_ALWAYS_INLINE Vc_CONST sfloat_v Const<Vc::sfloat>::load(const float *mem) { return M256::dup(float_v(mem).data()); }

    template<> Vc_ALWAYS_INLINE Vc_CONST Vector<float> Const<float>::highMask() { return Vector<float>(reinterpret_cast<const float *>(&c_general::highMaskFloat)); }
    template<> Vc_ALWAYS_INLINE Vc_CONST Vector<double> Const<double>::highMask() { return Vector<double>(reinterpret_cast<const double *>(&c_general::highMaskDouble)); }
    template<> Vc_ALWAYS_INLINE Vc_CONST Vector<float> Const<float>::highMask(int bits) { return _mm_castsi128_ps(_mm_slli_epi32(_mm_setallone_si128(), bits)); }
    template<> Vc_ALWAYS_INLINE Vc_CONST Vector<double> Const<double>::highMask(int bits) { return _mm_castsi128_pd(_mm_slli_epi64(_mm_setallone_si128(), bits)); }
    template<> Vc_ALWAYS_INLINE Vc_CONST Vector<float8> Const<float8>::highMask(int bits) {
        return M256::dup(Const<float>::highMask(bits).data());
    }
    template<> Vc_ALWAYS_INLINE Vc_CONST Vector<float8> Const<float8>::P(int i) {
        return M256::dup(Const<float>::P(i).data());
    }
    template<> Vc_ALWAYS_INLINE Vc_CONST Vector<float8> Const<float8>::Q(int i) {
        return M256::dup(Const<float>::Q(i).data());
    }
    template<> Vc_ALWAYS_INLINE Vc_CONST Vector<float8>::Mask Const<float8>::exponentMask() {
        return M256::dup(Const<float>::exponentMask().data());
    }
} // namespace SSE
} // namespace Vc
} // namespace ROOT

#include "undomacros.h"

#endif // VC_SSE_CONST_H
