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
#include <iostream>

using namespace Vc;

enum {
    VectorSizeFactor = short_v::Size / int_v::Size
};

void testSigned()
{
    for (int start = -32000; start < 32000; start += 5) {
        int_v a[VectorSizeFactor];
        for (int i = 0; i < VectorSizeFactor; ++i) {
            a[i] = int_v(IndexesFromZero) + int_v::Size * i + start;
        }
        short_v b(a);
        COMPARE(b, short_v(IndexesFromZero) + start);

        // false positive: warning: ‘c’ is used uninitialized in this function
        int_v c[VectorSizeFactor];
        b.expand(c);
        for (int i = 0; i < VectorSizeFactor; ++i) {
            COMPARE(c[i], int_v(IndexesFromZero) + int_v::Size * i + start);
        }
    }
}

void testUnsigned()
{
#if defined(VC_IMPL_SSE4_1) || defined(VC_IMPL_AVX)
    for (unsigned int start = 0; start < 64000; start += 5) {
#else
    for (unsigned int start = 0; start < 32000; start += 5) {
#endif
        uint_v a[VectorSizeFactor];
        for (unsigned int i = 0; i < VectorSizeFactor; ++i) {
            a[i] = uint_v(IndexesFromZero) + uint_v::Size * i + start;
        }
        ushort_v b(a);
        COMPARE(b, ushort_v(IndexesFromZero) + start);

        // false positive: warning: ‘c’ is used uninitialized in this function
        uint_v c[VectorSizeFactor];
        b.expand(c);
        for (unsigned int i = 0; i < VectorSizeFactor; ++i) {
            COMPARE(c[i], uint_v(IndexesFromZero) + uint_v::Size * i + start);
        }
    }
    for (unsigned int start = 32000; start < 64000; start += 5) {
        ushort_v b(IndexesFromZero);
        b += start;
        COMPARE(b, ushort_v(IndexesFromZero) + start);

        // false positive: warning: ‘c’ may be used uninitialized in this function
        uint_v c[VectorSizeFactor];
        b.expand(c);
        for (unsigned int i = 0; i < VectorSizeFactor; ++i) {
            COMPARE(c[i], uint_v(IndexesFromZero) + uint_v::Size * i + start);
        }
    }
}

int main()
{
    runTest(testSigned);
    runTest(testUnsigned);
    return 0;
}
