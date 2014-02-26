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

template<typename Vec> void maskedGatherArray()
{
    typedef typename Vec::IndexType It;
    typedef typename Vec::EntryType T;

    T mem[Vec::Size];
    for (int i = 0; i < Vec::Size; ++i) {
        mem[i] = i + 1;
    }

    It indexes = It::IndexesFromZero();
    for_all_masks(Vec, m) {
        const Vec a(mem, indexes, m);
        for (int i = 0; i < Vec::Size; ++i) {
            COMPARE(a[i], m[i] ? mem[i] : 0) << " i = " << i << ", m = " << m;
        }

        T x = Vec::Size + 1;
        Vec b = x;
        b.gather(mem, indexes, m);
        for (int i = 0; i < Vec::Size; ++i) {
            COMPARE(b[i], m[i] ? mem[i] : x) << " i = " << i << ", m = " << m;
        }
    }
}

template<typename Vec> void gatherArray()
{
    typedef typename Vec::IndexType It;
    typedef typename Vec::EntryType T;
    typedef typename It::Mask M;

    const int count = 39999;
    T array[count];
    for (int i = 0; i < count; ++i) {
        array[i] = i + 1;
    }
    M mask;
    for (It i = It(IndexesFromZero); !(mask = (i < count)).isEmpty(); i += Vec::Size) {
        const Vec ii(i + 1);
        const typename Vec::Mask castedMask = static_cast<typename Vec::Mask>(mask);
        if (castedMask.isFull()) {
            Vec a(array, i);
            COMPARE(a, ii) << "\n       i: " << i;
            Vec b(Zero);
            b.gather(array, i);
            COMPARE(b, ii);
            COMPARE(a, b);
        }
        Vec b(Zero);
        b.gather(array, i, castedMask);
        COMPARE(castedMask, (b == ii)) << ", b = " << b << ", ii = " << ii << ", i = " << i;
        if (!castedMask.isFull()) {
            COMPARE(!castedMask, b == Vec(Zero));
        }
    }

    const typename Vec::Mask k(Zero);
    Vec a(One);
    a.gather(array, It(IndexesFromZero), k);
    COMPARE(a, Vec(One));
}

template<typename T> struct Struct
{
    T a;
    char x;
    T b;
    short y;
    T c;
    char z;
};

template<typename Vec> void gatherStruct()
{
    typedef typename Vec::IndexType It;
    typedef typename Vec::EntryType T;
    typedef Struct<T> S;
    const int count = 3999;
    S array[count];
    for (int i = 0; i < count; ++i) {
        array[i].a = i;
        array[i].b = i + 1;
        array[i].c = i + 2;
    }
    typename It::Mask mask;
    for (It i = It(IndexesFromZero); !(mask = (i < count)).isEmpty(); i += Vec::Size) {
        // if Vec is double_v the cast keeps only the lower two values, which is why the ==
        // comparison works
        const Vec i0(i);
        const Vec i1(i + 1);
        const Vec i2(i + 2);
        const typename Vec::Mask castedMask(mask);

        if (castedMask.isFull()) {
            Vec a(array, &S::a, i);
            COMPARE(a, i0) << "\ni: " << i;
            a.gather(array, &S::b, i);
            COMPARE(a, i1);
            a.gather(array, &S::c, i);
            COMPARE(a, i2);
        }

        Vec b(Zero);
        b.gather(array, &S::a, i, castedMask);
        COMPARE(castedMask, (b == i0));
        if (!castedMask.isFull()) {
            COMPARE(!castedMask, b == Vec(Zero));
        }
        b.gather(array, &S::b, i, castedMask);
        COMPARE(castedMask, (b == i1));
        if (!castedMask.isFull()) {
            COMPARE(!castedMask, b == Vec(Zero));
        }
        b.gather(array, &S::c, i, castedMask);
        COMPARE(castedMask, (b == i2));
        if (!castedMask.isFull()) {
            COMPARE(!castedMask, b == Vec(Zero));
        }
    }
}

template<typename T> struct Row
{
    T *data;
};

template<typename Vec> void gather2dim()
{
    typedef typename Vec::IndexType It;
    typedef typename Vec::EntryType T;
    const int count = 399;
    typedef Row<T> S;
    S array[count];
    for (int i = 0; i < count; ++i) {
        array[i].data = new T[count];
        for (int j = 0; j < count; ++j) {
            array[i].data[j] = 2 * i + j + 1;
        }
    }

    typename It::Mask mask;
    for (It i = It(IndexesFromZero); !(mask = (i < count)).isEmpty(); i += Vec::Size) {
        for (It j = It(IndexesFromZero); !(mask &= (j < count)).isEmpty(); j += Vec::Size) {
            const Vec i0(i * 2 + j + 1);
            const typename Vec::Mask castedMask(mask);

            Vec a(array, &S::data, i, j, castedMask);
            COMPARE(castedMask, castedMask && (a == i0)) << ", a = " << a << ", i0 = " << i0 << ", i = " << i << ", j = " << j;

            Vec b(Zero);
            b.gather(array, &S::data, i, j, castedMask);
            COMPARE(castedMask, (b == i0));
            if (!castedMask.isFull()) {
                COMPARE(!castedMask, b == Vec(Zero));
            } else {
                Vec c(array, &S::data, i, j);
                VERIFY((c == i0).isFull());

                Vec d(Zero);
                d.gather(array, &S::data, i, j);
                VERIFY((d == i0).isFull());
            }
        }
    }
    for (int i = 0; i < count; ++i) {
        delete[] array[i].data;
    }
}

int main(int argc, char **argv)
{
    initTest(argc, argv);

    testAllTypes(gatherArray);
    testAllTypes(maskedGatherArray);
#if defined(VC_CLANG) && VC_CLANG <= 0x030000
    // clang fails with:
    //  candidate template ignored: failed template argument deduction
    //  template<typename S1, typename IT> inline Vector(const S1 *array, const T S1::* member1, IT indexes, Mask mask = true)
#warning "Skipping compilation of tests gatherStruct and gather2dim because of clang bug"
#else
    testAllTypes(gatherStruct);
    testAllTypes(gather2dim);
#endif

    return 0;
}
