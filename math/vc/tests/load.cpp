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

template<typename Vec> unsigned long alignmentMask()
{
    if (Vec::Size == 1) {
        // on 32bit the maximal alignment is 4 Bytes, even for 8-Byte doubles.
        return std::min(sizeof(void*), sizeof(typename Vec::EntryType)) - 1;
    }
    // sizeof(SSE::sfloat_v) is too large
    // AVX::VectorAlignment is too large
    return std::min<unsigned long>(sizeof(Vec), VectorAlignment) - 1;
}

template<typename Vec> void checkAlignment()
{
    unsigned char i = 1;
    Vec a[10];
    unsigned long mask = alignmentMask<Vec>();
    for (i = 0; i < 10; ++i) {
        VERIFY((reinterpret_cast<size_t>(&a[i]) & mask) == 0) << "a = " << a << ", mask = " << mask;
    }
    const char *data = reinterpret_cast<const char *>(&a[0]);
    for (i = 0; i < 10; ++i) {
        VERIFY(&data[i * Vec::Size * sizeof(typename Vec::EntryType)] == reinterpret_cast<const char *>(&a[i]));
    }
}

void *hack_to_put_b_on_the_stack = 0;

template<typename Vec> void checkMemoryAlignment()
{
    typedef typename Vec::EntryType T;
    const T *b = 0;
    Vc::Memory<Vec, 10> a;
    b = a;
    hack_to_put_b_on_the_stack = &b;
    unsigned long mask = alignmentMask<Vec>();
    for (int i = 0; i < 10; ++i) {
        VERIFY((reinterpret_cast<size_t>(&b[i * Vec::Size]) & mask) == 0) << "b = " << b << ", mask = " << mask;
    }
}

template<typename Vec> void loadArray()
{
    typedef typename Vec::EntryType T;
    typedef typename Vec::IndexType I;

    enum loadArrayEnum { count = 256 * 1024 / sizeof(T) };
    Vc::Memory<Vec, count> array;
    for (int i = 0; i < count; ++i) {
        array[i] = i;
    }

    const I indexesFromZero(IndexesFromZero);

    const Vec offsets(indexesFromZero);
    for (int i = 0; i < count; i += Vec::Size) {
        const T *const addr = &array[i];
        Vec ii(i);
        ii += offsets;

        Vec a(addr);
        COMPARE(a, ii);

        Vec b = Vec::Zero();
        b.load(addr);
        COMPARE(b, ii);
    }
}

enum Enum {
    loadArrayShortCount = 32 * 1024,
    streamingLoadCount = 1024
};
template<typename Vec> void loadArrayShort()
{
    typedef typename Vec::EntryType T;

    Vc::Memory<Vec, loadArrayShortCount> array;
    for (int i = 0; i < loadArrayShortCount; ++i) {
        array[i] = i;
    }

    const Vec &offsets = static_cast<Vec>(ushort_v::IndexesFromZero());
    for (int i = 0; i < loadArrayShortCount; i += Vec::Size) {
        const T *const addr = &array[i];
        Vec ii(i);
        ii += offsets;

        Vec a(addr);
        COMPARE(a, ii);

        Vec b = Vec::Zero();
        b.load(addr);
        COMPARE(b, ii);
    }
}

template<typename Vec> void streamingLoad()
{
    typedef typename Vec::EntryType T;

    Vc::Memory<Vec, streamingLoadCount> data;
    data[0] = static_cast<T>(-streamingLoadCount/2);
    for (int i = 1; i < streamingLoadCount; ++i) {
        data[i] = data[i - 1];
        ++data[i];
    }

    Vec ref = data.firstVector();
    for (int i = 0; i < streamingLoadCount - Vec::Size; ++i, ++ref) {
        Vec v1, v2;
        if (0 == i % Vec::Size) {
            v1 = Vec(&data[i], Vc::Streaming | Vc::Aligned);
            v2.load (&data[i], Vc::Streaming | Vc::Aligned);
        } else {
            v1 = Vec(&data[i], Vc::Streaming | Vc::Unaligned);
            v2.load (&data[i], Vc::Streaming | Vc::Unaligned);
        }
        COMPARE(v1, ref);
        COMPARE(v2, ref);
    }
}

template<typename T> struct TypeInfo;
template<> struct TypeInfo<double        > { static const char *string() { return "double"; } };
template<> struct TypeInfo<float         > { static const char *string() { return "float"; } };
template<> struct TypeInfo<int           > { static const char *string() { return "int"; } };
template<> struct TypeInfo<unsigned int  > { static const char *string() { return "uint"; } };
template<> struct TypeInfo<short         > { static const char *string() { return "short"; } };
template<> struct TypeInfo<unsigned short> { static const char *string() { return "ushort"; } };
template<> struct TypeInfo<signed char   > { static const char *string() { return "schar"; } };
template<> struct TypeInfo<unsigned char > { static const char *string() { return "uchar"; } };
template<> struct TypeInfo<double_v      > { static const char *string() { return "double_v"; } };
template<> struct TypeInfo<float_v       > { static const char *string() { return "float_v"; } };
template<> struct TypeInfo<sfloat_v      > { static const char *string() { return "sfloat_v"; } };
template<> struct TypeInfo<int_v         > { static const char *string() { return "int_v"; } };
template<> struct TypeInfo<uint_v        > { static const char *string() { return "uint_v"; } };
template<> struct TypeInfo<short_v       > { static const char *string() { return "short_v"; } };
template<> struct TypeInfo<ushort_v      > { static const char *string() { return "ushort_v"; } };

template<typename T, typename Current = void> struct SupportedConversions { typedef void Next; };
template<> struct SupportedConversions<float, void>           { typedef double         Next; };
template<> struct SupportedConversions<float, double>         { typedef int            Next; };
template<> struct SupportedConversions<float, int>            { typedef unsigned int   Next; };
template<> struct SupportedConversions<float, unsigned int>   { typedef short          Next; };
template<> struct SupportedConversions<float, short>          { typedef unsigned short Next; };
template<> struct SupportedConversions<float, unsigned short> { typedef signed char    Next; };
template<> struct SupportedConversions<float, signed char>    { typedef unsigned char  Next; };
template<> struct SupportedConversions<float, unsigned char>  { typedef void           Next; };
template<> struct SupportedConversions<int  , void          > { typedef unsigned int   Next; };
template<> struct SupportedConversions<int  , unsigned int  > { typedef short          Next; };
template<> struct SupportedConversions<int  , short         > { typedef unsigned short Next; };
template<> struct SupportedConversions<int  , unsigned short> { typedef signed char    Next; };
template<> struct SupportedConversions<int  , signed char   > { typedef unsigned char  Next; };
template<> struct SupportedConversions<int  , unsigned char > { typedef void           Next; };
template<> struct SupportedConversions<unsigned int, void          > { typedef unsigned short Next; };
template<> struct SupportedConversions<unsigned int, unsigned short> { typedef unsigned char  Next; };
template<> struct SupportedConversions<unsigned int, unsigned char > { typedef void           Next; };
template<> struct SupportedConversions<unsigned short, void          > { typedef unsigned char  Next; };
template<> struct SupportedConversions<unsigned short, unsigned char > { typedef void           Next; };
template<> struct SupportedConversions<         short, void          > { typedef unsigned char  Next; };
template<> struct SupportedConversions<         short, unsigned char > { typedef signed char    Next; };
template<> struct SupportedConversions<         short,   signed char > { typedef void           Next; };

template<typename Vec, typename MemT> struct LoadCvt {
    static void test() {
        typedef typename Vec::EntryType VecT;
        MemT *data = Vc::malloc<MemT, Vc::AlignOnCacheline>(128);
        for (size_t i = 0; i < 128; ++i) {
            data[i] = static_cast<MemT>(i - 64);
        }

        for (size_t i = 0; i < 128 - Vec::Size + 1; ++i) {
            Vec v;
            if (i % (2 * Vec::Size) == 0) {
                v = Vec(&data[i]);
            } else if (i % Vec::Size == 0) {
                v = Vec(&data[i], Vc::Aligned);
            } else {
                v = Vec(&data[i], Vc::Unaligned);
            }
            for (size_t j = 0; j < Vec::Size; ++j) {
                COMPARE(v[j], static_cast<VecT>(data[i + j])) << " " << TypeInfo<MemT>::string();
            }
        }
        for (size_t i = 0; i < 128 - Vec::Size + 1; ++i) {
            Vec v;
            if (i % (2 * Vec::Size) == 0) {
                v.load(&data[i]);
            } else if (i % Vec::Size == 0) {
                v.load(&data[i], Vc::Aligned);
            } else {
                v.load(&data[i], Vc::Unaligned);
            }
            for (size_t j = 0; j < Vec::Size; ++j) {
                COMPARE(v[j], static_cast<VecT>(data[i + j])) << " " << TypeInfo<MemT>::string();
            }
        }
        for (size_t i = 0; i < 128 - Vec::Size + 1; ++i) {
            Vec v;
            if (i % (2 * Vec::Size) == 0) {
                v = Vec(&data[i], Vc::Streaming);
            } else if (i % Vec::Size == 0) {
                v = Vec(&data[i], Vc::Streaming | Vc::Aligned);
            } else {
                v = Vec(&data[i], Vc::Streaming | Vc::Unaligned);
            }
            for (size_t j = 0; j < Vec::Size; ++j) {
                COMPARE(v[j], static_cast<VecT>(data[i + j])) << " " << TypeInfo<MemT>::string();
            }
        }

        ADD_PASS() << "loadCvt: load " << TypeInfo<MemT>::string() << "* as " << TypeInfo<Vec>::string();
        LoadCvt<Vec, typename SupportedConversions<VecT, MemT>::Next>::test();
    }
};
template<typename Vec> struct LoadCvt<Vec, void> { static void test() {} };

template<typename Vec> void loadCvt()
{
    typedef typename Vec::EntryType T;
    LoadCvt<Vec, typename SupportedConversions<T>::Next>::test();
}

int main()
{
    runTest(checkAlignment<int_v>);
    runTest(checkAlignment<uint_v>);
    runTest(checkAlignment<float_v>);
    runTest(checkAlignment<double_v>);
    runTest(checkAlignment<short_v>);
    runTest(checkAlignment<ushort_v>);
    runTest(checkAlignment<sfloat_v>);
    testAllTypes(checkMemoryAlignment);
    runTest(loadArray<int_v>);
    runTest(loadArray<uint_v>);
    runTest(loadArray<float_v>);
    runTest(loadArray<double_v>);
    runTest(loadArray<sfloat_v>);
    runTest(loadArrayShort<short_v>);
    runTest(loadArrayShort<ushort_v>);

    testAllTypes(streamingLoad);

    testAllTypes(loadCvt);
    return 0;
}
