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

#ifndef VC_AVX_DEBUG_H
#define VC_AVX_DEBUG_H

#ifndef NDEBUG
#include "vectorbase.h"
#include <iostream>
#include <iomanip>
#endif

namespace ROOT {
namespace Vc
{
namespace AVX
{

#ifdef NDEBUG
class DebugStream
{
    public:
        DebugStream(const char *, const char *, int) {}
        template<typename T> inline DebugStream &operator<<(const T &) { return *this; }
};
#else
class DebugStream
{
    private:
        template<typename T, typename V> static void printVector(V _x)
        {
            enum { Size = sizeof(V) / sizeof(T) };
            union { V v; T m[Size]; } x = { _x };
            std::cerr << '[' << std::setprecision(24) << x.m[0];
            for (int i = 1; i < Size; ++i) {
                std::cerr << ", " << std::setprecision(24) << x.m[i];
            }
            std::cerr << ']';
        }
    public:
        DebugStream(const char *func, const char *file, int line)
        {
            std::cerr << "\033[1;40;33mDEBUG: " << file << ':' << line << ' ' << func << ' ';
        }

        template<typename T> DebugStream &operator<<(const T &x) { std::cerr << x; return *this; }

        DebugStream &operator<<(__m128 x) {
            printVector<float, __m128>(x);
            return *this;
        }
        DebugStream &operator<<(__m256 x) {
            printVector<float, __m256>(x);
            return *this;
        }
        DebugStream &operator<<(__m128d x) {
            printVector<double, __m128d>(x);
            return *this;
        }
        DebugStream &operator<<(__m256d x) {
            printVector<double, __m256d>(x);
            return *this;
        }
        DebugStream &operator<<(__m128i x) {
            printVector<unsigned int, __m128i>(x);
            return *this;
        }
        DebugStream &operator<<(__m256i x) {
            printVector<unsigned int, __m256i>(x);
            return *this;
        }

        ~DebugStream()
        {
            std::cerr << "\033[0m" << std::endl;
        }
};
#endif

#define VC_DEBUG ::ROOT::Vc::AVX::DebugStream(__PRETTY_FUNCTION__, __FILE__, __LINE__)

} // namespace AVX
} // namespace Vc
} // namespace ROOT

#endif // VC_AVX_DEBUG_H
