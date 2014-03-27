/*  This file is part of the Vc library.

    Copyright (C) 2009-2011 Matthias Kretz <kretz@kde.org>

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
#include "vectormemoryhelper.h"
#include <cmath>

using namespace Vc;

template<typename Vec> void testInc()
{
    VectorMemoryHelper<Vec> mem(2);
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    T *data = mem;
    for (int borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (int i = 0; i < Vec::Size; ++i) {
            data[i] = static_cast<T>(i);
            data[i + Vec::Size] = data[i] + static_cast<T>(data[i] < border ? 1 : 0);
        }
        Vec a(&data[0]);
        Vec b(&data[Vec::Size]);
        Mask m = a < border;
        Vec aa(a);
        COMPARE(aa(m)++, a) << ", border: " << border << ", m: " << m;
        COMPARE(aa, b) << ", border: " << border << ", m: " << m;
        COMPARE(++a(m), b) << ", border: " << border << ", m: " << m;
        COMPARE(a, b) << ", border: " << border << ", m: " << m;
    }
}

template<typename Vec> void testDec()
{
    VectorMemoryHelper<Vec> mem(2);
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    T *data = mem;
    for (int borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (int i = 0; i < Vec::Size; ++i) {
            data[i] = static_cast<T>(i + 1);
            data[i + Vec::Size] = data[i] - static_cast<T>(data[i] < border ? 1 : 0);
        }
        Vec a(&data[0]);
        Vec b(&data[Vec::Size]);
        Mask m = a < border;
        Vec aa(a);
        COMPARE(aa(m)--, a);
        COMPARE(--a(m), b);
        COMPARE(a, b);
        COMPARE(aa, b);
    }
}

template<typename Vec> void testPlusEq()
{
    VectorMemoryHelper<Vec> mem(2);
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    T *data = mem;
    for (int borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (int i = 0; i < Vec::Size; ++i) {
            data[i] = static_cast<T>(i + 1);
            data[i + Vec::Size] = data[i] + static_cast<T>(data[i] < border ? 2 : 0);
        }
        Vec a(&data[0]);
        Vec b(&data[Vec::Size]);
        Mask m = a < border;
        COMPARE(a(m) += static_cast<T>(2), b);
        COMPARE(a, b);
    }
}

template<typename Vec> void testMinusEq()
{
    VectorMemoryHelper<Vec> mem(2);
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    T *data = mem;
    for (int borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (int i = 0; i < Vec::Size; ++i) {
            data[i] = static_cast<T>(i + 2);
            data[i + Vec::Size] = data[i] - static_cast<T>(data[i] < border ? 2 : 0);
        }
        Vec a(&data[0]);
        Vec b(&data[Vec::Size]);
        Mask m = a < border;
        COMPARE(a(m) -= static_cast<T>(2), b);
        COMPARE(a, b);
    }
}

template<typename Vec> void testTimesEq()
{
    VectorMemoryHelper<Vec> mem(2);
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    T *data = mem;
    for (int borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (int i = 0; i < Vec::Size; ++i) {
            data[i] = static_cast<T>(i);
            data[i + Vec::Size] = data[i] * static_cast<T>(data[i] < border ? 2 : 1);
        }
        Vec a(&data[0]);
        Vec b(&data[Vec::Size]);
        Mask m = a < border;
        COMPARE(a(m) *= static_cast<T>(2), b);
        COMPARE(a, b);
    }
}

template<typename Vec> void testDivEq()
{
    VectorMemoryHelper<Vec> mem(2);
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    T *data = mem;
    for (int borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (int i = 0; i < Vec::Size; ++i) {
            data[i] = static_cast<T>(5 * i);
            data[i + Vec::Size] = data[i] / static_cast<T>(data[i] < border ? 3 : 1);
        }
        Vec a(&data[0]);
        Vec b(&data[Vec::Size]);
        Mask m = a < border;
        COMPARE(a(m) /= static_cast<T>(3), b);
        COMPARE(a, b);
    }
}

template<typename Vec> void testAssign()
{
    VectorMemoryHelper<Vec> mem(2);
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    T *data = mem;
    for (int borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (int i = 0; i < Vec::Size; ++i) {
            data[i] = static_cast<T>(i);
            data[i + Vec::Size] = data[i] + static_cast<T>(data[i] < border ? 2 : 0);
        }
        Vec a(&data[0]);
        Vec b(&data[Vec::Size]);
        Mask m = a < border;
        COMPARE(a(m) = b, b);
        COMPARE(a, b);
    }
}

template<typename Vec> void testZero()
{
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    typedef typename Vec::IndexType I;

    for (int cut = 0; cut < Vec::Size; ++cut) {
        const Mask mask(I(Vc::IndexesFromZero) < cut);
        //std::cout << mask << std::endl;

        const T aa = 4;
        Vec a(aa);
        Vec b(Vc::Zero);

        b(!mask) = a;
        a.setZero(mask);

        COMPARE(a, b);
    }
}

template<typename Vec> void testCount()
{
    for_all_masks(Vec, m) {
        int count = 0;
        for (int i = 0; i < Vec::Size; ++i) {
            if (m[i]) {
                ++count;
            }
        }
        COMPARE(m.count(), count) << ", m = " << m;
    }
}

template<typename Vec> void testFirstOne()
{
    typedef typename Vec::IndexType I;
    typedef typename Vec::Mask M;

    for (int i = 0; i < Vec::Size; ++i) {
        const M mask(I(Vc::IndexesFromZero) == i);
        COMPARE(mask.firstOne(), i);
    }
}

template<typename M1, typename M2> void testLogicalOperatorsImpl()
{
    VERIFY((M1(true) && M2(true)).isFull());
    VERIFY((M1(true) && M2(false)).isEmpty());
    VERIFY((M1(true) || M2(true)).isFull());
    VERIFY((M1(true) || M2(false)).isFull());
    VERIFY((M1(false) || M2(false)).isEmpty());
}

template<typename M1, typename M2> void testBinaryOperatorsImpl()
{
    testLogicalOperatorsImpl<M1, M2>();

    VERIFY((M1(true) & M2(true)).isFull());
    VERIFY((M1(true) & M2(false)).isEmpty());
    VERIFY((M1(true) | M2(true)).isFull());
    VERIFY((M1(true) | M2(false)).isFull());
    VERIFY((M1(false) | M2(false)).isEmpty());
    VERIFY((M1(true) ^ M2(true)).isEmpty());
    VERIFY((M1(true) ^ M2(false)).isFull());
}

void testBinaryOperators()
{
    testLogicalOperatorsImpl< short_m, sfloat_m>();
    testLogicalOperatorsImpl<ushort_m, sfloat_m>();
    testLogicalOperatorsImpl<sfloat_m,  short_m>();
    testLogicalOperatorsImpl<sfloat_m, ushort_m>();

    testBinaryOperatorsImpl< short_m,  short_m>();
    testBinaryOperatorsImpl< short_m, ushort_m>();
    testBinaryOperatorsImpl<ushort_m,  short_m>();
    testBinaryOperatorsImpl<ushort_m, ushort_m>();
    testBinaryOperatorsImpl<sfloat_m, sfloat_m>();

    testBinaryOperatorsImpl<   int_m,    int_m>();
    testBinaryOperatorsImpl<   int_m,   uint_m>();
    testBinaryOperatorsImpl<   int_m,  float_m>();
    testBinaryOperatorsImpl<  uint_m,    int_m>();
    testBinaryOperatorsImpl<  uint_m,   uint_m>();
    testBinaryOperatorsImpl<  uint_m,  float_m>();
    testBinaryOperatorsImpl< float_m,    int_m>();
    testBinaryOperatorsImpl< float_m,   uint_m>();
    testBinaryOperatorsImpl< float_m,  float_m>();

    testBinaryOperatorsImpl<double_m, double_m>();
}

#ifdef VC_IMPL_SSE
void testFloat8GatherMask()
{
    Memory<short_v, short_v::Size * 256> data;
    short_v::Memory andMemory;
    for (int i = 0; i < short_v::Size; ++i) {
        andMemory[i] = 1 << i;
    }
    const short_v andMask(andMemory);

    for (unsigned int i = 0; i < data.vectorsCount(); ++i) {
        data.vector(i) = andMask & i;
    }

    for (unsigned int i = 0; i < data.vectorsCount(); ++i) {
        const short_m mask = data.vector(i) == short_v::Zero();

        SSE::Float8GatherMask
            gatherMaskA(mask),
            gatherMaskB(static_cast<sfloat_m>(mask));
        COMPARE(gatherMaskA.toInt(), gatherMaskB.toInt());
    }
}
#endif

int main(int argc, char **argv)
{
    initTest(argc, argv);

    testAllTypes(testInc);
    testAllTypes(testDec);
    testAllTypes(testPlusEq);
    testAllTypes(testMinusEq);
    testAllTypes(testTimesEq);
    testAllTypes(testDivEq);
    testAllTypes(testAssign);
    testAllTypes(testZero);
    testAllTypes(testCount);
    testAllTypes(testFirstOne);
    runTest(testBinaryOperators);

#ifdef VC_IMPL_SSE
    runTest(testFloat8GatherMask);
#endif

    return 0;
}
