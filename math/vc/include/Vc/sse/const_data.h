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

#ifndef VC_SSE_CONST_DATA_H
#define VC_SSE_CONST_DATA_H

#include "macros.h"

namespace ROOT {
namespace Vc
{
namespace SSE
{

ALIGN(16) extern const unsigned int   _IndexesFromZero4[4];
ALIGN(16) extern const unsigned short _IndexesFromZero8[8];
ALIGN(16) extern const unsigned char  _IndexesFromZero16[16];

struct c_general
{
    ALIGN(64) static const unsigned int allone[4];
    ALIGN(16) static const unsigned short one16[8];
    ALIGN(16) static const unsigned int one32[4];
    ALIGN(16) static const float oneFloat[4];
    ALIGN(16) static const double oneDouble[2];
    ALIGN(16) static const int absMaskFloat[4];
    ALIGN(16) static const long long absMaskDouble[2];
    ALIGN(16) static const unsigned int signMaskFloat[4];
    ALIGN(16) static const unsigned int highMaskFloat[4];
    ALIGN(16) static const unsigned long long signMaskDouble[2];
    ALIGN(16) static const unsigned long long highMaskDouble[2];
    ALIGN(16) static const short minShort[8];
    ALIGN(16) static const unsigned long long frexpMask[2];
};

template<typename T> struct c_trig
{
    ALIGN(64) static const T data[];
};

template<typename T> struct c_log
{
    enum VectorSize { Size = 16 / sizeof(T) };
    static Vc_ALWAYS_INLINE Vc_CONST const float *d(int i) { return reinterpret_cast<const  float *>(&data[i * Size]); }
    ALIGN(64) static const unsigned int data[];
};

template<> struct c_log<double>
{
    enum VectorSize { Size = 16 / sizeof(double) };
    static Vc_ALWAYS_INLINE Vc_CONST const double *d(int i) { return reinterpret_cast<const double *>(&data[i * Size]); }
    ALIGN(64) static const unsigned long long data[];
};

} // namespace SSE
} // namespace Vc
} // namespace ROOT

#include "undomacros.h"

#endif // VC_SSE_CONST_DATA_H
