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

using namespace Vc;

template<typename V, unsigned int Size, template<typename V2, unsigned int Size2> class TestClass> struct TestWrapper
{
    static inline void run()
    {
        TestWrapper<V, Size/2, TestClass>::run();
        TestClass<V, Size>::test();
        TestClass<V, Size - 1>::test();
    }
};
template<typename V, template<typename V2, unsigned int Size> class TestClass> struct TestWrapper<V, 1, TestClass> {
    static inline void run() {}
};

template<typename V, unsigned int Size> struct TestEntries { static void test() {
    typedef typename V::EntryType T;
    const T x = Size;
    Memory<V, Size> m;
    const Memory<V, Size> &m2 = m;
    Memory<V> m3(Size);
    for (unsigned int i = 0; i < Size; ++i) {
        m[i] = x;
        m3[i] = x;
    }
    for (unsigned int i = 0; i < Size; ++i) {
        COMPARE(m[i], x);
        COMPARE(m2[i], x);
        COMPARE(m3[i], x);
    }
    for (unsigned int i = 0; i < Size; ++i) {
        COMPARE(m.entries()[i], x);
        COMPARE(m2.entries()[i], x);
        COMPARE(m3.entries()[i], x);
    }
    const T *ptr = m2;
    for (unsigned int i = 0; i < Size; ++i) {
        COMPARE(ptr[i], x);
    }
    ptr = m3;
    for (unsigned int i = 0; i < Size; ++i) {
        COMPARE(ptr[i], x);
    }
}};

template<typename V, unsigned int Size> struct TestEntries2D { static void test() {
    typedef typename V::EntryType T;
    const T x = Size;
    Memory<V, Size, Size> m;
    const Memory<V, Size, Size> &m2 = m;

    for (size_t i = 0; i < Size; ++i) {
        for (size_t j = 0; j < Size; ++j) {
            m[i][j] = x + i + j;
        }
    }
    for (size_t i = 0; i < Size; ++i) {
        for (size_t j = 0; j < Size; ++j) {
            COMPARE(m[i][j], T(x + i + j));
            COMPARE(m2[i][j], T(x + i + j));
        }
    }
    for (size_t i = 0; i < Size; ++i) {
        for (size_t j = 0; j < Size; ++j) {
            COMPARE(m[i].entries()[j], T(x + i + j));
            COMPARE(m2[i].entries()[j], T(x + i + j));
        }
    }
    for (size_t i = 0; i < Size; ++i) {
        const T *ptr = m2[i];
        for (size_t j = 0; j < Size; ++j) {
            COMPARE(ptr[j], T(x + i + j));
        }
    }
}};

template<typename V, unsigned int Size> struct TestVectors { static void test()
{
    const V startX(V::IndexType::IndexesFromZero() + Size);
    Memory<V, Size> m;
    const Memory<V, Size> &m2 = m;
    Memory<V> m3(Size);
    V x = startX;
    for (unsigned int i = 0; i < m.vectorsCount(); ++i, x += V::Size) {
        m.vector(i) = x;
        m3.vector(i) = x;
    }
    x = startX;
    unsigned int i;
    for (i = 0; i + 1 < m.vectorsCount(); ++i) {
        COMPARE(V(m.vector(i)), x);
        COMPARE(V(m2.vector(i)), x);
        COMPARE(V(m3.vector(i)), x);
        for (int shift = 0; shift < V::Size; ++shift, ++x) {
            COMPARE(V(m.vector(i, shift)), x);
            COMPARE(V(m2.vector(i, shift)), x);
            COMPARE(V(m3.vector(i, shift)), x);
        }
    }
    COMPARE(V(m.vector(i)), x);
    COMPARE(V(m2.vector(i)), x);
    COMPARE(V(m3.vector(i)), x);
}};

template<typename V, unsigned int Size> struct TestVectors2D { static void test()
{
    const V startX(V::IndexType::IndexesFromZero() + Size);
    Memory<V, Size, Size> m;
    const Memory<V, Size, Size> &m2 = m;
    V x = startX;
    for (size_t i = 0; i < m.rowsCount(); ++i, x += V::Size) {
        Memory<V, Size> &mrow = m[i];
        for (size_t j = 0; j < mrow.vectorsCount(); ++j, x += V::Size) {
            mrow.vector(j) = x;
        }
    }
    x = startX;
    for (size_t i = 0; i < m.rowsCount(); ++i, x += V::Size) {
        Memory<V, Size> &mrow = m[i];
        const Memory<V, Size> &m2row = m2[i];
        size_t j;
        for (j = 0; j < mrow.vectorsCount() - 1; ++j) {
            COMPARE(V(mrow.vector(j)), x);
            COMPARE(V(m2row.vector(j)), x);
            for (int shift = 0; shift < V::Size; ++shift, ++x) {
                COMPARE(V(mrow.vector(j, shift)), x);
                COMPARE(V(m2row.vector(j, shift)), x);
            }
        }
        COMPARE(V(mrow.vector(j)), x) << i << " " << j;
        COMPARE(V(m2row.vector(j)), x);
        x += V::Size;
    }
}};

template<typename V, unsigned int Size> struct TestVectorReorganization { static void test()
{
    typename V::Memory init;
    for (unsigned int i = 0; i < V::Size; ++i) {
        init[i] = i;
    }
    V x(init);
    Memory<V, Size> m;
    Memory<V> m3(Size);
    for (unsigned int i = 0; i < m.vectorsCount(); ++i) {
        m.vector(i) = x;
        m3.vector(i) = x;
        x += V::Size;
    }
    ///////////////////////////////////////////////////////////////////////////
    x = V(init);
    for (unsigned int i = 0; i < m.vectorsCount(); ++i) {
        COMPARE(V(m.vector(i)), x);
        COMPARE(V(m3.vector(i)), x);
        x += V::Size;
    }
    ///////////////////////////////////////////////////////////////////////////
    x = V(init);
    unsigned int indexes[Size];
    for (unsigned int i = 0; i < Size; ++i) {
        indexes[i] = i;
    }
    for (unsigned int i = 0; i + V::Size < Size; ++i) {
        COMPARE(m.gather(&indexes[i]), x);
        COMPARE(m3.gather(&indexes[i]), x);
        x += 1;
    }
    ///////////////////////////////////////////////////////////////////////////
    for (unsigned int i = 0; i < V::Size; ++i) {
        init[i] = i * 2;
    }
    x = V(init);
    for (unsigned int i = 0; i < Size; ++i) {
        indexes[i] = (i * 2) % Size;
    }
    for (unsigned int i = 0; i + V::Size < Size; ++i) {
        COMPARE(m.gather(&indexes[i]), x);
        COMPARE(m3.gather(&indexes[i]), x);
        x += 2;
        x(x >= Size) -= Size;
    }
}};

template<typename V> void testEntries()
{
    TestWrapper<V, 128, TestEntries>::run();
}

template<typename V> void testEntries2D()
{
    TestWrapper<V, 32, TestEntries2D>::run();
}

template<typename V> void testVectors()
{
    TestWrapper<V, 128, TestVectors>::run();
}

template<typename V> void testVectors2D()
{
    TestWrapper<V, 32, TestVectors2D>::run();
}

template<typename V> void testVectorReorganization()
{
    TestWrapper<V, 128, TestVectorReorganization>::run();
}

template<typename V> void memoryOperators()
{
    Memory<V, 129> m1, m2;
    m1.setZero();
    m2.setZero();
    VERIFY(m1 == m2);
    VERIFY(!(m1 != m2));
    VERIFY(!(m1 < m2));
    VERIFY(!(m1 > m2));
    m1 += m2;
    VERIFY(m1 == m2);
    VERIFY(m1 <= m2);
    VERIFY(m1 >= m2);
    m1 += 1;
    VERIFY(m1 != m2);
    VERIFY(m1 > m2);
    VERIFY(m1 >= m2);
    VERIFY(m2 < m1);
    VERIFY(m2 <= m1);
    VERIFY(!(m1 == m2));
    VERIFY(!(m1 <= m2));
    VERIFY(!(m2 >= m1));
    m2 += m1;
    VERIFY(m1 == m2);
    m2 *= 2;
    m1 += 1;
    VERIFY(m1 == m2);
    m2 /= 2;
    m1 -= 1;
    VERIFY(m1 == m2);
    m1 *= m2;
    VERIFY(m1 == m2);
    m1 /= m2;
    VERIFY(m1 == m2);
    m1 -= m2;
    m2 -= m2;
    VERIFY(m1 == m2);
}

template<typename V> void testCCtor()
{
    Memory<V> m1(5);
    for (size_t i = 0; i < m1.entriesCount(); ++i) {
        m1[i] = i;
    }
    Memory<V> m2(m1);
    for (size_t i = 0; i < m1.entriesCount(); ++i) {
        m1[i] += 1;
    }
    for (size_t i = 0; i < m1.entriesCount(); ++i) {
        COMPARE(m1[i], m2[i] + 1);
    }
}

template<typename V> void testCopyAssignment()
{
    typedef typename V::EntryType T;

    Memory<V, 99> m1;
    m1.setZero();

    Memory<V, 99> m2(m1);
    for (size_t i = 0; i < m2.entriesCount(); ++i) {
        COMPARE(m2[i], T(0));
        m2[i] += 1;
    }
    m1 = m2;
    for (size_t i = 0; i < m2.entriesCount(); ++i) {
        COMPARE(m1[i], T(1));
    }
}

int main()
{
    testAllTypes(testEntries);
    testAllTypes(testEntries2D);
    testAllTypes(testVectors);
    testAllTypes(testVectors2D);
    testAllTypes(testVectorReorganization);
    testAllTypes(memoryOperators);
    testAllTypes(testCCtor);
    testAllTypes(testCopyAssignment);

    return 0;
}
