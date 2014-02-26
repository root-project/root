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
// includes {{{1
#include "unittest.h"
#include <iostream>
#include <cstring>

using namespace Vc;

template<typename Vec> void maskedScatterArray() //{{{1
{
    typedef typename Vec::IndexType It;
    typedef typename Vec::EntryType T;

    T mem[Vec::Size];
    const Vec v(It::IndexesFromZero() + 1);

    for_all_masks(Vec, m) {
        Vec::Zero().store(mem, Vc::Unaligned);
        v.scatter(&mem[0], It::IndexesFromZero(), m);

        for (int i = 0; i < Vec::Size; ++i) {
            COMPARE(mem[i], m[i] ? v[i] : T(0)) << " i = " << i << ", m = " << m;
        }
    }
}

template<typename Vec> void scatterArray() //{{{1
{
    typedef typename Vec::IndexType It;
    const int count = 31999;
    typename Vec::EntryType array[count], out[count];
    for (int i = 0; i < count; ++i) {
        array[i] = i - 100;
    }
    typename It::Mask mask;
    for (It i(IndexesFromZero); !(mask = (i < count)).isEmpty(); i += Vec::Size) {
        typename Vec::Mask castedMask(mask);
        if (castedMask.isFull()) {
            Vec a(array, i);
            a += Vec(One);
            a.scatter(out, i);
        } else {
            Vec a(array, i, castedMask);
            a += Vec(One);
            a.scatter(out, i, castedMask);
        }
    }
    for (int i = 0; i < count; ++i) {
        array[i] += 1;
        COMPARE(array[i], out[i]);
    }
    COMPARE(0, std::memcmp(array, out, count * sizeof(typename Vec::EntryType)));
}

template<typename T> struct Struct //{{{1
{
    T a;
    char x;
    T b;
    short y;
    T c;
    char z;
};

template<typename Vec> void scatterStruct() //{{{1
{
    typedef typename Vec::IndexType It;
    typedef Struct<typename Vec::EntryType> S;
    const int count = 3999;
    S array[count], out[count];
    memset(array, 0, count * sizeof(S));
    memset(out, 0, count * sizeof(S));
    for (int i = 0; i < count; ++i) {
        array[i].a = i;
        array[i].b = i + 1;
        array[i].c = i + 2;
    }
    typename It::Mask mask;
    for (It i(IndexesFromZero); !(mask = (i < count)).isEmpty(); i += Vec::Size) {
        typename Vec::Mask castedMask(mask);
        Vec a(array, &S::a, i, castedMask);
        Vec b(array, &S::b, i, castedMask);
        Vec c(array, &S::c, i, castedMask);
        a.scatter(out, &S::a, i, castedMask);
        b.scatter(out, &S::b, i, castedMask);
        c.scatter(out, &S::c, i, castedMask);
    }
    VERIFY(0 == memcmp(array, out, count * sizeof(S)));
}

template<typename T> struct Struct2 //{{{1
{
    char x;
    Struct<T> b;
    short y;
};

template<typename Vec> void scatterStruct2() //{{{1
{
    typedef typename Vec::IndexType It;
    typedef Struct2<typename Vec::EntryType> S1;
    typedef Struct<typename Vec::EntryType> S2;
    const int count = 97;
    S1 array[count], out[count];
    memset(array, 0, count * sizeof(S1));
    memset(out, 0, count * sizeof(S1));
    for (int i = 0; i < count; ++i) {
        array[i].b.a = i + 0;
        array[i].b.b = i + 1;
        array[i].b.c = i + 2;
    }
    typename It::Mask mask;
    for (It i(IndexesFromZero); !(mask = (i < count)).isEmpty(); i += Vec::Size) {
        typename Vec::Mask castedMask(mask);
        Vec a(array, &S1::b, &S2::a, i, castedMask);
        Vec b(array, &S1::b, &S2::b, i, castedMask);
        Vec c(array, &S1::b, &S2::c, i, castedMask);
        a.scatter(out, &S1::b, &S2::a, i, castedMask);
        b.scatter(out, &S1::b, &S2::b, i, castedMask);
        c.scatter(out, &S1::b, &S2::c, i, castedMask);
    }
    VERIFY(0 == memcmp(array, out, count * sizeof(S1)));
}

int main(int argc, char **argv) //{{{1
{
    initTest(argc, argv);

    runTest(scatterArray<int_v>);
    runTest(scatterArray<uint_v>);
    runTest(scatterArray<float_v>);
    runTest(scatterArray<double_v>);
    runTest(scatterArray<sfloat_v>);
    runTest(scatterArray<short_v>);
    runTest(scatterArray<ushort_v>);
    testAllTypes(maskedScatterArray);
#if defined(VC_CLANG) && VC_CLANG <= 0x030000
    // clang fails with:
    //  candidate template ignored: failed template argument deduction
    //  template<typename S1, typename IT> inline Vector(const S1 *array, const T S1::*
    //          member1, IT indexes, Mask mask = true)
#warning "Skipping compilation of tests scatterStruct and scatterStruct2 because of clang bug"
#else
    runTest(scatterStruct<int_v>);
    runTest(scatterStruct<uint_v>);
    runTest(scatterStruct<float_v>);
    runTest(scatterStruct<double_v>);
    runTest(scatterStruct<sfloat_v>);
    runTest(scatterStruct<short_v>);
    runTest(scatterStruct<ushort_v>);
    testAllTypes(scatterStruct2);
#endif
    return 0;
}

// vim: foldmethod=marker
