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

#include "unittest.h"

using namespace Vc;

enum Swizzle {
    BADC, CDAB, AAAA, BBBB, CCCC, DDDD, BCAD, BCDA, DABC, ACBD, DBCA, DCBA
};

template<typename V> V scalarSwizzle(VC_ALIGNED_PARAMETER(V) v, Swizzle s)
{
    V r = v;
    for (int i = 0; i + 4 <= V::Size; i += 4) {
        switch (s) {
        case BADC:
            r[i + 0] = v[i + 1];
            r[i + 1] = v[i + 0];
            r[i + 2] = v[i + 3];
            r[i + 3] = v[i + 2];
            break;
        case CDAB:
            r[i + 0] = v[i + 2];
            r[i + 1] = v[i + 3];
            r[i + 2] = v[i + 0];
            r[i + 3] = v[i + 1];
            break;
        case AAAA:
            r[i + 0] = v[i + 0];
            r[i + 1] = v[i + 0];
            r[i + 2] = v[i + 0];
            r[i + 3] = v[i + 0];
            break;
        case BBBB:
            r[i + 0] = v[i + 1];
            r[i + 1] = v[i + 1];
            r[i + 2] = v[i + 1];
            r[i + 3] = v[i + 1];
            break;
        case CCCC:
            r[i + 0] = v[i + 2];
            r[i + 1] = v[i + 2];
            r[i + 2] = v[i + 2];
            r[i + 3] = v[i + 2];
            break;
        case DDDD:
            r[i + 0] = v[i + 3];
            r[i + 1] = v[i + 3];
            r[i + 2] = v[i + 3];
            r[i + 3] = v[i + 3];
            break;
        case BCAD:
            r[i + 0] = v[i + 1];
            r[i + 1] = v[i + 2];
            r[i + 2] = v[i + 0];
            r[i + 3] = v[i + 3];
            break;
        case BCDA:
            r[i + 0] = v[i + 1];
            r[i + 1] = v[i + 2];
            r[i + 2] = v[i + 3];
            r[i + 3] = v[i + 0];
            break;
        case DABC:
            r[i + 0] = v[i + 3];
            r[i + 1] = v[i + 0];
            r[i + 2] = v[i + 1];
            r[i + 3] = v[i + 2];
            break;
        case ACBD:
            r[i + 0] = v[i + 0];
            r[i + 1] = v[i + 2];
            r[i + 2] = v[i + 1];
            r[i + 3] = v[i + 3];
            break;
        case DBCA:
            r[i + 0] = v[i + 3];
            r[i + 1] = v[i + 1];
            r[i + 2] = v[i + 2];
            r[i + 3] = v[i + 0];
            break;
        case DCBA:
            r[i + 0] = v[i + 3];
            r[i + 1] = v[i + 2];
            r[i + 2] = v[i + 1];
            r[i + 3] = v[i + 0];
            break;
        }
    }
    return r;
}

template<typename V> void testSwizzle()
{
    for (int i = 0; i < 100; ++i) {
        const V test = V::Random();
        COMPARE(test.abcd(), test);
        COMPARE(test.badc(), scalarSwizzle(test, BADC));
        COMPARE(test.cdab(), scalarSwizzle(test, CDAB));
        COMPARE(test.aaaa(), scalarSwizzle(test, AAAA));
        COMPARE(test.bbbb(), scalarSwizzle(test, BBBB));
        COMPARE(test.cccc(), scalarSwizzle(test, CCCC));
        COMPARE(test.dddd(), scalarSwizzle(test, DDDD));
        COMPARE(test.bcad(), scalarSwizzle(test, BCAD));
        COMPARE(test.bcda(), scalarSwizzle(test, BCDA));
        COMPARE(test.dabc(), scalarSwizzle(test, DABC));
        COMPARE(test.acbd(), scalarSwizzle(test, ACBD));
        COMPARE(test.dbca(), scalarSwizzle(test, DBCA));
        COMPARE(test.dcba(), scalarSwizzle(test, DCBA));
    }
}

int main(int argc, char **argv)
{
    initTest(argc, argv);

#if VC_DOUBLE_V_SIZE >= 4 || VC_DOUBLE_V_SIZE == 1
    runTest(testSwizzle<double_v>);
#endif
    runTest(testSwizzle<float_v>);
    runTest(testSwizzle<sfloat_v>);
    runTest(testSwizzle<int_v>);
    runTest(testSwizzle<uint_v>);
    runTest(testSwizzle<short_v>);
    runTest(testSwizzle<ushort_v>);

    return 0;
}
