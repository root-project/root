/*  This file is part of the Vc library. {{{

    Copyright (C) 2012 Matthias Kretz <kretz@kde.org>

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

}}}*/

#ifndef VC_AVX_CONST_DATA_H
#define VC_AVX_CONST_DATA_H

#include "macros.h"
namespace ROOT {
namespace Vc
{
namespace AVX
{

ALIGN(64) extern const unsigned int   _IndexesFromZero32[8];
ALIGN(16) extern const unsigned short _IndexesFromZero16[8];
ALIGN(16) extern const unsigned char  _IndexesFromZero8[16];

struct STRUCT_ALIGN1(64) c_general
{
    static const float oneFloat;
    static const unsigned int absMaskFloat[2];
    static const unsigned int signMaskFloat[2];
    static const unsigned int highMaskFloat;
    static const unsigned short minShort[2];
    static const unsigned short one16[2];
    static const float _2power31;
    static const double oneDouble;
    static const unsigned long long frexpMask;
    static const unsigned long long highMaskDouble;
} STRUCT_ALIGN2(64);

template<typename T> struct c_trig
{
    ALIGN(64) static const T data[];
};

template<typename T> struct c_log
{
    typedef float floatAlias Vc_MAY_ALIAS;
    static Vc_ALWAYS_INLINE float d(int i) { return *reinterpret_cast<const floatAlias *>(&data[i]); }
    ALIGN(64) static const unsigned int data[];
};

template<> struct c_log<double>
{
    enum VectorSize { Size = 16 / sizeof(double) };
    typedef double doubleAlias Vc_MAY_ALIAS;
    static Vc_ALWAYS_INLINE double d(int i) { return *reinterpret_cast<const doubleAlias *>(&data[i]); }
    ALIGN(64) static const unsigned long long data[];
};

} // namespace AVX
} // namespace Vc
} // namespace ROOT

#include "undomacros.h"

#endif // VC_AVX_CONST_DATA_H
