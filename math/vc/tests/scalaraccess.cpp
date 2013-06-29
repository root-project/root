/*  This file is part of the Vc library.

    Copyright (C) 2010-2012 Matthias Kretz <kretz@kde.org>

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

template<typename V> void reads()
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;

    V a = V::Zero();
    const T zero = 0;
    for (int i = 0; i < V::Size; ++i) {
        const T x = a[i];
        COMPARE(x, zero);
    }
    a = static_cast<V>(I::IndexesFromZero());
    for (int i = 0; i < V::Size; ++i) {
        const T x = a[i];
        const T y = i;
        COMPARE(x, y);
    }
}

template<typename V, size_t Index>
inline void readsConstantIndexTest(VC_ALIGNED_PARAMETER(V) a, VC_ALIGNED_PARAMETER(V) b)
{
    typedef typename V::EntryType T;
    {
        const T x = a[Index];
        const T zero = 0;
        COMPARE(x, zero) << Index;
    }{
        const T x = b[Index];
        const T y = Index;
        COMPARE(x, y) << Index;
    }
}

template<typename V, size_t Index>
struct ReadsConstantIndex
{
    ReadsConstantIndex(VC_ALIGNED_PARAMETER(V) a, VC_ALIGNED_PARAMETER(V) b)
    {
        readsConstantIndexTest<V, Index>(a, b);
        ReadsConstantIndex<V, Index - 1>(a, b);
    }
};


template<typename V>
struct ReadsConstantIndex<V, 0>
{
    ReadsConstantIndex(VC_ALIGNED_PARAMETER(V) a, VC_ALIGNED_PARAMETER(V) b)
    {
        readsConstantIndexTest<V, 0>(a, b);
    }
};

template<typename V> void readsConstantIndex()
{
    typedef typename V::IndexType I;

    V a = V::Zero();
    V b = static_cast<V>(I::IndexesFromZero());
    ReadsConstantIndex<V, V::Size - 1>(a, b);
}

template<typename V> void writes()
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;

    V a;
    for (int i = 0; i < V::Size; ++i) {
        a[i] = static_cast<T>(i);
    }
    V b = static_cast<V>(I::IndexesFromZero());
    COMPARE(a, b);

    const T one = 1;
    const T two = 2;

    if (V::Size == 1) {
        a(a == 0) += one;
        a[0] += one;
        a(a == 0) += one;
        COMPARE(a, V(2));
    } else if (V::Size == 4) {
        a(a == 1) += two;
        a[2] += one;
        a(a == 3) += one;
        b(b == 1) += one;
        b(b == 2) += one;
        b(b == 3) += one;
        COMPARE(a, b);
    } else if (V::Size == 8 || V::Size == 16) {
        a(a == 2) += two;
        a[3] += one;
        a(a == 4) += one;
        b(b == 2) += one;
        b(b == 3) += one;
        b(b == 4) += one;
        COMPARE(a, b);
    } else if (V::Size == 2) { // a = [0, 1]; b = [0, 1]
        a(a == 0) += two;      // a = [2, 1]
        a[1] += one;           // a = [2, 2]
        a(a == 2) += one;      // a = [3, 3]
        b(b == 0) += one;      // b = [1, 1]
        b(b == 1) += one;      // b = [2, 2]
        b(b == 2) += one;      // b = [3, 3]
        COMPARE(a, b);
    } else {
        FAIL() << "unsupported Vector::Size";
    }
}

int main(int argc, char **argv)
{
    initTest(argc, argv);

    testAllTypes(reads);
    testAllTypes(writes);
    testAllTypes(readsConstantIndex);
    //testAllTypes(writesConstantIndex);

    return 0;
}
