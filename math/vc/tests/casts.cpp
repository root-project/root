/*  This file is part of the Vc library.

    Copyright (C) 2010-2011 Matthias Kretz <kretz@kde.org>

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
#include <limits>
#include <algorithm>

using namespace Vc;

template<typename V1, typename V2> void testNumber(double n)
{
    typedef typename V1::EntryType T1;
    typedef typename V2::EntryType T2;

    // compare casts from T1 -> T2 with casts from V1 -> V2

    const T1 n1 = static_cast<T1>(n);
    //std::cerr << "n1 = " << n1 << ", static_cast<T2>(n1) = " << static_cast<T2>(n1) << std::endl;
    COMPARE(static_cast<V2>(V1(n1)), V2(static_cast<T2>(n1))) << "\n       n1: " << n1;
}

template<typename T> double maxHelper()
{
    return static_cast<double>(std::numeric_limits<T>::max());
}

template<> double maxHelper<int>()
{
    const int intDigits = std::numeric_limits<int>::digits;
    const int floatDigits = std::numeric_limits<float>::digits;
    return static_cast<double>(((int(1) << floatDigits) - 1) << (intDigits - floatDigits));
}

template<> double maxHelper<unsigned int>()
{
    const int intDigits = std::numeric_limits<unsigned int>::digits;
    const int floatDigits = std::numeric_limits<float>::digits;
    return static_cast<double>(((unsigned(1) << floatDigits) - 1) << (intDigits - floatDigits));
}

template<typename V1, typename V2> void testCast2()
{
    typedef typename V1::EntryType T1;
    typedef typename V2::EntryType T2;

    const double max = std::min(maxHelper<T1>(), maxHelper<T2>());
    const double min = std::max(
            std::numeric_limits<T1>::is_integer ?
                static_cast<double>(std::numeric_limits<T1>::min()) :
                static_cast<double>(-std::numeric_limits<T1>::max()),
            std::numeric_limits<T2>::is_integer ?
                static_cast<double>(std::numeric_limits<T2>::min()) :
                static_cast<double>(-std::numeric_limits<T2>::max())
                );

    testNumber<V1, V2>(0.);
    testNumber<V1, V2>(1.);
    testNumber<V1, V2>(2.);
    testNumber<V1, V2>(max);
    testNumber<V1, V2>(max / 4 + max / 2);
    testNumber<V1, V2>(max / 2);
    testNumber<V1, V2>(max / 4);
    testNumber<V1, V2>(min);
}

template<typename T> void testCast()
{
    testCast2<typename T::V1, typename T::V2>();
}

#define _CONCAT(A, B) A ## _ ## B
#define CONCAT(A, B) _CONCAT(A, B)
template<typename T1, typename T2>
struct T2Helper
{
    typedef T1 V1;
    typedef T2 V2;
};

void testFloatIndexesFromZero()
{
    Vc::float_v test(Vc::int_v::IndexesFromZero());
    for (int i = 0; i < float_v::Size; ++i) {
        COMPARE(test[i], float(i));
    }
}

int main(int argc, char **argv)
{
    initTest(argc, argv);

#define TEST(v1, v2) \
    typedef T2Helper<v1, v2> CONCAT(v1, v2); \
    runTest(testCast<CONCAT(v1, v2)>)

    TEST(float_v, float_v);
    TEST(float_v, int_v);
    TEST(float_v, uint_v);
    // needs special handling for different Size:
    //TEST(float_v, double_v);
    //TEST(float_v, short_v);
    //TEST(float_v, ushort_v);

    TEST(int_v, float_v);
    TEST(int_v, int_v);
    TEST(int_v, uint_v);

    TEST(uint_v, float_v);
    TEST(uint_v, int_v);
    TEST(uint_v, uint_v);

    TEST(ushort_v, sfloat_v);
    TEST(ushort_v, short_v);
    TEST(ushort_v, ushort_v);

    TEST(short_v, sfloat_v);
    TEST(short_v, short_v);
    TEST(short_v, ushort_v);

    TEST(sfloat_v, sfloat_v);
    TEST(sfloat_v, short_v);
    TEST(sfloat_v, ushort_v);
#undef TEST

    runTest(testFloatIndexesFromZero);

    return 0;
}
