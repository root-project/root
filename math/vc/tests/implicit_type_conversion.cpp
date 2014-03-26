/*{{{
    Copyright (C) 2012 Matthias Kretz <kretz@kde.org>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

}}}*/

#include "unittest.h"

//#define QUICK 1

using namespace Vc;

typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef long long longlong;
typedef unsigned long long ulonglong;

#ifdef QUICK
#define _TYPE_TEST(a, b, c)
#define _TYPE_TEST_ERR(a, b)
#else
#if defined(VC_GCC) && VC_GCC == 0x40801
// Skipping tests involving operator& because of a bug in GCC 4.8.1 (http://gcc.gnu.org/bugzilla/show_bug.cgi?id=57532)
#define _TYPE_TEST(a, b, c) \
    COMPARE(typeid(a() * b()), typeid(c)); \
    COMPARE(typeid(a() / b()), typeid(c)); \
    COMPARE(typeid(a() + b()), typeid(c)); \
    COMPARE(typeid(a() - b()), typeid(c)); \
    COMPARE(typeid(a() | b()), typeid(c)); \
    COMPARE(typeid(a() ^ b()), typeid(c)); \
    COMPARE(typeid(a() == b()), typeid(c::Mask)); \
    COMPARE(typeid(a() != b()), typeid(c::Mask)); \
    COMPARE(typeid(a() <= b()), typeid(c::Mask)); \
    COMPARE(typeid(a() >= b()), typeid(c::Mask)); \
    COMPARE(typeid(a() <  b()), typeid(c::Mask));
#else
#define _TYPE_TEST(a, b, c) \
    COMPARE(typeid(a() * b()), typeid(c)); \
    COMPARE(typeid(a() / b()), typeid(c)); \
    COMPARE(typeid(a() + b()), typeid(c)); \
    COMPARE(typeid(a() - b()), typeid(c)); \
    COMPARE(typeid(a() & b()), typeid(c)); \
    COMPARE(typeid(a() | b()), typeid(c)); \
    COMPARE(typeid(a() ^ b()), typeid(c)); \
    COMPARE(typeid(a() == b()), typeid(c::Mask)); \
    COMPARE(typeid(a() != b()), typeid(c::Mask)); \
    COMPARE(typeid(a() <= b()), typeid(c::Mask)); \
    COMPARE(typeid(a() >= b()), typeid(c::Mask)); \
    COMPARE(typeid(a() <  b()), typeid(c::Mask));
#endif
#define _TYPE_TEST_ERR(a, b) \
    COMPARE(typeid(a() *  b()), typeid(Vc::Error::invalid_operands_of_types<a, b>)); \
    COMPARE(typeid(a() /  b()), typeid(Vc::Error::invalid_operands_of_types<a, b>)); \
    COMPARE(typeid(a() +  b()), typeid(Vc::Error::invalid_operands_of_types<a, b>)); \
    COMPARE(typeid(a() -  b()), typeid(Vc::Error::invalid_operands_of_types<a, b>)); \
    COMPARE(typeid(a() &  b()), typeid(Vc::Error::invalid_operands_of_types<a, b>)); \
    COMPARE(typeid(a() |  b()), typeid(Vc::Error::invalid_operands_of_types<a, b>)); \
    COMPARE(typeid(a() ^  b()), typeid(Vc::Error::invalid_operands_of_types<a, b>)); \
    COMPARE(typeid(a() == b()), typeid(Vc::Error::invalid_operands_of_types<a, b>)); \
    COMPARE(typeid(a() != b()), typeid(Vc::Error::invalid_operands_of_types<a, b>)); \
    COMPARE(typeid(a() <= b()), typeid(Vc::Error::invalid_operands_of_types<a, b>)); \
    COMPARE(typeid(a() >= b()), typeid(Vc::Error::invalid_operands_of_types<a, b>)); \
    COMPARE(typeid(a() <  b()), typeid(Vc::Error::invalid_operands_of_types<a, b>));
#endif

#define TYPE_TEST(a, b, c) \
    _TYPE_TEST(a, b, c) \
    COMPARE(typeid(a() >  b()), typeid(c::Mask))

template<typename T>
struct TestImplicitCast {
    static bool test(const T &) { return  true; }
    static bool test(   ...   ) { return false; }
};

enum SomeEnum { EnumValue = 0 };
SomeEnum Enum() { return EnumValue; }

void testImplicitTypeConversions()
{
    VERIFY( TestImplicitCast<     int>::test(double()));
    VERIFY( TestImplicitCast<     int>::test( float()));
    VERIFY( TestImplicitCast<     int>::test(  Enum()));
    VERIFY( TestImplicitCast<     int>::test( short()));
    VERIFY( TestImplicitCast<     int>::test(ushort()));
    VERIFY( TestImplicitCast<     int>::test(  char()));
    VERIFY( TestImplicitCast<     int>::test(  uint()));
    VERIFY( TestImplicitCast<     int>::test(  long()));
    VERIFY( TestImplicitCast<     int>::test( ulong()));
    VERIFY( TestImplicitCast<     int>::test(  bool()));
    VERIFY( TestImplicitCast<double_v>::test(double()));
    VERIFY(!TestImplicitCast<double_v>::test( float()));
    VERIFY(!TestImplicitCast<double_v>::test(   int()));
    VERIFY( TestImplicitCast< float_v>::test( float()));
    VERIFY( TestImplicitCast<sfloat_v>::test( float()));
    VERIFY( TestImplicitCast<   int_v>::test(   int()));
    VERIFY( TestImplicitCast<  uint_v>::test(  uint()));
    VERIFY( TestImplicitCast< short_v>::test( short()));
    VERIFY( TestImplicitCast<ushort_v>::test(ushort()));

    TYPE_TEST( double_v,    double_v, double_v);
    TYPE_TEST( double_v,      double, double_v);
    TYPE_TEST( double_v,       float, double_v);
    TYPE_TEST( double_v,       short, double_v);
    TYPE_TEST( double_v,      ushort, double_v);
    TYPE_TEST( double_v,         int, double_v);
    TYPE_TEST( double_v,        uint, double_v);
    TYPE_TEST( double_v,        long, double_v);
    TYPE_TEST( double_v,       ulong, double_v);
    TYPE_TEST( double_v,    longlong, double_v);
    TYPE_TEST( double_v,   ulonglong, double_v);
    TYPE_TEST( double_v,        Enum, double_v);
    TYPE_TEST(   double,    double_v, double_v);
    TYPE_TEST(    float,    double_v, double_v);
    TYPE_TEST(    short,    double_v, double_v);
    TYPE_TEST(   ushort,    double_v, double_v);
    TYPE_TEST(      int,    double_v, double_v);
    TYPE_TEST(     uint,    double_v, double_v);
    TYPE_TEST(     long,    double_v, double_v);
    TYPE_TEST(    ulong,    double_v, double_v);
    TYPE_TEST( longlong,    double_v, double_v);
    TYPE_TEST(ulonglong,    double_v, double_v);
    // double_v done

    TYPE_TEST(  float_v,     float_v,  float_v);
    TYPE_TEST(  float_v,       float,  float_v);
    TYPE_TEST(  float_v,       short,  float_v);
    TYPE_TEST(  float_v,      ushort,  float_v);
    TYPE_TEST(  float_v,       int_v,  float_v);
    TYPE_TEST(  float_v,         int,  float_v);
    TYPE_TEST(  float_v,      uint_v,  float_v);
    TYPE_TEST(  float_v,        uint,  float_v);
    TYPE_TEST(  float_v,        long,  float_v);
    TYPE_TEST(  float_v,       ulong,  float_v);
    TYPE_TEST(  float_v,    longlong,  float_v);
    TYPE_TEST(  float_v,   ulonglong,  float_v);
    TYPE_TEST(    float,     float_v,  float_v);
    TYPE_TEST(    short,     float_v,  float_v);
    TYPE_TEST(   ushort,     float_v,  float_v);
    TYPE_TEST(    int_v,     float_v,  float_v);
    TYPE_TEST(      int,     float_v,  float_v);
    TYPE_TEST(   uint_v,     float_v,  float_v);
    TYPE_TEST(     uint,     float_v,  float_v);
    TYPE_TEST(     long,     float_v,  float_v);
    TYPE_TEST(    ulong,     float_v,  float_v);
    TYPE_TEST( longlong,     float_v,  float_v);
    TYPE_TEST(ulonglong,     float_v,  float_v);
    // double_v + float_v done

    TYPE_TEST( sfloat_v,    sfloat_v, sfloat_v);
    TYPE_TEST( sfloat_v,       float, sfloat_v);
    TYPE_TEST( sfloat_v,     short_v, sfloat_v);
    TYPE_TEST( sfloat_v,       short, sfloat_v);
    TYPE_TEST( sfloat_v,    ushort_v, sfloat_v);
    TYPE_TEST( sfloat_v,      ushort, sfloat_v);
    TYPE_TEST( sfloat_v,         int, sfloat_v);
    TYPE_TEST( sfloat_v,        uint, sfloat_v);
    TYPE_TEST( sfloat_v,        long, sfloat_v);
    TYPE_TEST( sfloat_v,       ulong, sfloat_v);
    TYPE_TEST( sfloat_v,    longlong, sfloat_v);
    TYPE_TEST( sfloat_v,   ulonglong, sfloat_v);
    TYPE_TEST( sfloat_v,    sfloat_v, sfloat_v);
    TYPE_TEST(    float,    sfloat_v, sfloat_v);
    TYPE_TEST(  short_v,    sfloat_v, sfloat_v);
    TYPE_TEST(    short,    sfloat_v, sfloat_v);
    TYPE_TEST( ushort_v,    sfloat_v, sfloat_v);
    TYPE_TEST(   ushort,    sfloat_v, sfloat_v);
    TYPE_TEST(      int,    sfloat_v, sfloat_v);
    TYPE_TEST(     uint,    sfloat_v, sfloat_v);
    TYPE_TEST(     long,    sfloat_v, sfloat_v);
    TYPE_TEST(    ulong,    sfloat_v, sfloat_v);
    TYPE_TEST( longlong,    sfloat_v, sfloat_v);
    TYPE_TEST(ulonglong,    sfloat_v, sfloat_v);
    // double_v + float_v + sfloat_v done

    TYPE_TEST(  short_v,     short_v,  short_v);
    TYPE_TEST(  short_v,       short,  short_v);
    TYPE_TEST(  short_v,    ushort_v, ushort_v);
    TYPE_TEST(  short_v,      ushort, ushort_v);
    TYPE_TEST(  short_v,         int,  short_v);
    TYPE_TEST(  short_v,        uint, ushort_v);
    TYPE_TEST(  short_v,        long,  short_v);
    TYPE_TEST(  short_v,       ulong, ushort_v);
    TYPE_TEST(  short_v,    longlong,  short_v);
    TYPE_TEST(  short_v,   ulonglong, ushort_v);
    TYPE_TEST(    short,     short_v,  short_v);
    TYPE_TEST( ushort_v,     short_v, ushort_v);
    TYPE_TEST(   ushort,     short_v, ushort_v);
    TYPE_TEST(      int,     short_v,  short_v);
    TYPE_TEST(     uint,     short_v, ushort_v);
    TYPE_TEST(     long,     short_v,  short_v);
    TYPE_TEST(    ulong,     short_v, ushort_v);
    TYPE_TEST( longlong,     short_v,  short_v);
    TYPE_TEST(ulonglong,     short_v, ushort_v);
    // double_v + float_v + sfloat_v + short_v done

    TYPE_TEST( ushort_v,       short, ushort_v);
    TYPE_TEST( ushort_v,    ushort_v, ushort_v);
    TYPE_TEST( ushort_v,      ushort, ushort_v);
    TYPE_TEST( ushort_v,         int, ushort_v);
    TYPE_TEST( ushort_v,        uint, ushort_v);
    TYPE_TEST( ushort_v,        long, ushort_v);
    TYPE_TEST( ushort_v,       ulong, ushort_v);
    TYPE_TEST( ushort_v,    longlong, ushort_v);
    TYPE_TEST( ushort_v,   ulonglong, ushort_v);
    TYPE_TEST(    short,    ushort_v, ushort_v);
    TYPE_TEST(   ushort,    ushort_v, ushort_v);
    TYPE_TEST(      int,    ushort_v, ushort_v);
    TYPE_TEST(     uint,    ushort_v, ushort_v);
    TYPE_TEST(     long,    ushort_v, ushort_v);
    TYPE_TEST(    ulong,    ushort_v, ushort_v);
    TYPE_TEST( longlong,    ushort_v, ushort_v);
    TYPE_TEST(ulonglong,    ushort_v, ushort_v);
    // double_v + float_v + sfloat_v + short_v + ushort_v done

    TYPE_TEST(    int_v,      ushort,   uint_v);
    TYPE_TEST(    int_v,       short,    int_v);
    TYPE_TEST(    int_v,       int_v,    int_v);
    TYPE_TEST(    int_v,         int,    int_v);
    TYPE_TEST(    int_v,      uint_v,   uint_v);
    TYPE_TEST(    int_v,        uint,   uint_v);
    TYPE_TEST(    int_v,        long,    int_v);
    TYPE_TEST(    int_v,       ulong,   uint_v);
    TYPE_TEST(    int_v,    longlong,    int_v);
    TYPE_TEST(    int_v,   ulonglong,   uint_v);
    TYPE_TEST(   ushort,       int_v,   uint_v);
    TYPE_TEST(    short,       int_v,    int_v);
    TYPE_TEST(      int,       int_v,    int_v);
    TYPE_TEST(   uint_v,       int_v,   uint_v);
    TYPE_TEST(     uint,       int_v,   uint_v);
    TYPE_TEST(     long,       int_v,    int_v);
    TYPE_TEST(    ulong,       int_v,   uint_v);
    TYPE_TEST( longlong,       int_v,    int_v);
    TYPE_TEST(ulonglong,       int_v,   uint_v);

    TYPE_TEST(   uint_v,       short,   uint_v);
    TYPE_TEST(   uint_v,      ushort,   uint_v);
    TYPE_TEST(   uint_v,       int_v,   uint_v);
    TYPE_TEST(   uint_v,         int,   uint_v);
    TYPE_TEST(   uint_v,      uint_v,   uint_v);
    TYPE_TEST(   uint_v,        uint,   uint_v);
    TYPE_TEST(   uint_v,        long,   uint_v);
    TYPE_TEST(   uint_v,       ulong,   uint_v);
    TYPE_TEST(   uint_v,    longlong,   uint_v);
    TYPE_TEST(   uint_v,   ulonglong,   uint_v);
    TYPE_TEST(    short,      uint_v,   uint_v);
    TYPE_TEST(   ushort,      uint_v,   uint_v);
    TYPE_TEST(    int_v,      uint_v,   uint_v);
    TYPE_TEST(      int,      uint_v,   uint_v);
    TYPE_TEST(     uint,      uint_v,   uint_v);
    TYPE_TEST(     long,      uint_v,   uint_v);
    TYPE_TEST(    ulong,      uint_v,   uint_v);
    TYPE_TEST( longlong,      uint_v,   uint_v);
    TYPE_TEST(ulonglong,      uint_v,   uint_v);
}

int main(int argc, char **argv)
{
    initTest(argc, argv);
    runTest(testImplicitTypeConversions);
    return 0;
}
