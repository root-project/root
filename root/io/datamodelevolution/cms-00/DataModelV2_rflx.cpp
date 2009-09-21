// Generated at Mon Sep 21 11:52:16 2009. Do not modify it

/*
GCC-XML version 0.9.0_20081002
Configuration settings:
  GCCXML_CONFIG="/afs/cern.ch/sw/lcg/external/gccxml/0.9.0_20081002/slc4_ia32_gcc34/share/gccxml-0.9/gccxml_config"
  GCCXML_COMPILER="c++"
  GCCXML_CXXFLAGS=" "
  GCCXML_EXECUTABLE="/afs/cern.ch/sw/lcg/external/gccxml/0.9.0_20081002/slc4_ia32_gcc34/bin/gccxml_cc1plus"
  GCCXML_CPP="/afs/cern.ch/sw/lcg/external/gccxml/0.9.0_20081002/slc4_ia32_gcc34/bin/gccxml_cc1plus"
  GCCXML_FLAGS="-D__DBL_MIN_EXP__='(-1021)' -D__FLT_MIN__='1.17549435e-38F' -D__CHAR_BIT__='8' -D__WCHAR_MAX__='2147483647' -D__DBL_DENORM_MIN__='4.9406564584124654e-324' -D__FLT_EVAL_METHOD__='2' -D__DBL_MIN_10_EXP__='(-307)' -D__FINITE_MATH_ONLY__='0' -D__GNUC_PATCHLEVEL__='3' -D__SHRT_MAX__='32767' -D__LDBL_MAX__='1.18973149535723176502e+4932L' -D__linux='1' -D__unix='1' -D__LDBL_MAX_EXP__='16384' -D__linux__='1' -D__SCHAR_MAX__='127' -D__USER_LABEL_PREFIX__='' -D__STDC_HOSTED__='1' -D__LDBL_HAS_INFINITY__='1' -D__DBL_DIG__='15' -D__FLT_EPSILON__='1.19209290e-7F' -D__GXX_WEAK__='1' -D__tune_i686__='1' -D__LDBL_MIN__='3.36210314311209350626e-4932L' -D__unix__='1' -D__DECIMAL_DIG__='21' -D__gnu_linux__='1' -D__LDBL_HAS_QUIET_NAN__='1' -D__GNUC__='3' -D__DBL_MAX__='1.7976931348623157e+308' -D__DBL_HAS_INFINITY__='1' -D__cplusplus='1' -D__DEPRECATED='1' -D__DBL_MAX_EXP__='1024' -D__GNUG__='3' -D__LONG_LONG_MAX__='9223372036854775807LL' -D__GXX_ABI_VERSION='1002' -D__FLT_MIN_EXP__='(-125)' -D__DBL_MIN__='2.2250738585072014e-308' -D__FLT_MIN_10_EXP__='(-37)' -D__DBL_HAS_QUIET_NAN__='1' -D__REGISTER_PREFIX__='' -D__NO_INLINE__='1' -D__i386='1' -D__FLT_MANT_DIG__='24' -D__VERSION__='"3.4.3"' -Di386='1' -Dunix='1' -D__i386__='1' -D__SIZE_TYPE__='unsigned int' -D__ELF__='1' -D__FLT_RADIX__='2' -D__LDBL_EPSILON__='1.08420217248550443401e-19L' -D__FLT_HAS_QUIET_NAN__='1' -D__FLT_MAX_10_EXP__='38' -D__LONG_MAX__='2147483647L' -D__FLT_HAS_INFINITY__='1' -Dlinux='1' -D__EXCEPTIONS='1' -D__LDBL_MANT_DIG__='64' -D__WCHAR_TYPE__='long int' -D__FLT_DIG__='6' -D__INT_MAX__='2147483647' -D__FLT_MAX_EXP__='128' -D__DBL_MANT_DIG__='53' -D__WINT_TYPE__='unsigned int' -D__LDBL_MIN_EXP__='(-16381)' -D__LDBL_MAX_10_EXP__='4932' -D__DBL_EPSILON__='2.2204460492503131e-16' -D__tune_pentiumpro__='1' -D__FLT_DENORM_MIN__='1.40129846e-45F' -D__FLT_MAX__='3.40282347e+38F' -D__GNUC_MINOR__='4' -D__DBL_MAX_10_EXP__='308' -D__LDBL_DENORM_MIN__='3.64519953188247460253e-4951L' -D__PTRDIFF_TYPE__='int' -D__LDBL_MIN_10_EXP__='(-4931)' -D__LDBL_DIG__='18' -D_GNU_SOURCE='1' -iwrapper"/afs/cern.ch/sw/lcg/external/gccxml/0.9.0_20081002/slc4_ia32_gcc34/share/gccxml-0.9/GCC/3.4" -isystem/afs/fnal.gov/ups/gcc/v3_4_3/Linux+2.6-2.3.4/bin/../lib/gcc/i686-pc-linux-gnu/3.4.3/../../../../include/c++/3.4.3 -isystem/afs/fnal.gov/ups/gcc/v3_4_3/Linux+2.6-2.3.4/bin/../lib/gcc/i686-pc-linux-gnu/3.4.3/../../../../include/c++/3.4.3/i686-pc-linux-gnu -isystem/afs/fnal.gov/ups/gcc/v3_4_3/Linux+2.6-2.3.4/bin/../lib/gcc/i686-pc-linux-gnu/3.4.3/../../../../include/c++/3.4.3/backward -isystem/afs/fnal.gov/ups/gcc/v3_4_3/Linux+2.6-2.3.4/bin/../lib/gcc/i686-pc-linux-gnu/3.4.3/include -isystem/usr/local/include -isystem/usr/include -include "gccxml_builtins.h" "
  GCCXML_USER_FLAGS=""
  GCCXML_ROOT="/afs/cern.ch/sw/lcg/external/gccxml/0.9.0_20081002/slc4_ia32_gcc34/share/gccxml-0.9"

Compiler info:
c++ (GCC) 3.4.3
Copyright (C) 2004 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

*/

#ifdef _WIN32
#pragma warning ( disable : 4786 )
#pragma warning ( disable : 4345 )
#elif defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ == 3
# pragma GCC diagnostic ignored "-Warray-bounds"
#endif
#include "DataModelV2.h"
#ifdef CONST
# undef CONST
#endif
#include "Reflex/Builder/ReflexBuilder.h"
#include <typeinfo>

#include "TBuffer.h"
#include "TVirtualObject.h"
#include <vector>
#include "TSchemaHelper.h"


namespace {
  ::Reflex::NamespaceBuilder nsb0( "std" );
  ::Reflex::Type type_void = ::Reflex::TypeBuilder("void");
  ::Reflex::Type type_1823 = ::Reflex::TypeBuilder("float");
  ::Reflex::Type type_1031 = ::Reflex::TypeBuilder("ClassAIns");
  ::Reflex::Type type_36 = ::Reflex::TypeBuilder("int");
  ::Reflex::Type type_29 = ::Reflex::TypeBuilder("ClassD");
  ::Reflex::Type type_3850 = ::Reflex::ReferenceBuilder(type_29);
  ::Reflex::Type type_29c = ::Reflex::ConstBuilder(type_29);
  ::Reflex::Type type_3852 = ::Reflex::ReferenceBuilder(type_29c);
  ::Reflex::Type type_1818 = ::Reflex::TypeBuilder("double");
  ::Reflex::Type type_776 = ::Reflex::TypeBuilder("ClassABase");
  ::Reflex::Type type_7084 = ::Reflex::ReferenceBuilder(type_776);
  ::Reflex::Type type_776c = ::Reflex::ConstBuilder(type_776);
  ::Reflex::Type type_7085 = ::Reflex::ReferenceBuilder(type_776c);
  ::Reflex::Type type_2653 = ::Reflex::TypeBuilder("bool");
  ::Reflex::Type type_994 = ::Reflex::TypeBuilder("ClassA");
  ::Reflex::Type type_4204 = ::Reflex::ReferenceBuilder(type_994);
  ::Reflex::Type type_994c = ::Reflex::ConstBuilder(type_994);
  ::Reflex::Type type_4206 = ::Reflex::ReferenceBuilder(type_994c);
  ::Reflex::Type type_113 = ::Reflex::TypeBuilder("void");
  ::Reflex::Type type_174 = ::Reflex::TypeBuilder("short");
  ::Reflex::Type type_995 = ::Reflex::TypeBuilder("ClassB");
  ::Reflex::Type type_4086 = ::Reflex::ReferenceBuilder(type_995);
  ::Reflex::Type type_995c = ::Reflex::ConstBuilder(type_995);
  ::Reflex::Type type_4088 = ::Reflex::ReferenceBuilder(type_995c);
  ::Reflex::Type type_996 = ::Reflex::TypeBuilder("ClassC");
  ::Reflex::Type type_3968 = ::Reflex::ReferenceBuilder(type_996);
  ::Reflex::Type type_996c = ::Reflex::ConstBuilder(type_996);
  ::Reflex::Type type_3970 = ::Reflex::ReferenceBuilder(type_996c);
  ::Reflex::Type type_7089 = ::Reflex::ReferenceBuilder(type_1031);
  ::Reflex::Type type_1031c = ::Reflex::ConstBuilder(type_1031);
  ::Reflex::Type type_7090 = ::Reflex::ReferenceBuilder(type_1031c);
  ::Reflex::Type type_1638 = ::Reflex::TypeBuilder("std::_Vector_base<ClassD*,std::allocator<ClassD*> >");
  ::Reflex::Type type_3786 = ::Reflex::PointerBuilder(type_29);
  ::Reflex::Type type_3788 = ::Reflex::PointerBuilder(type_3786);
  ::Reflex::Type type_3786c = ::Reflex::ConstBuilder(type_3786);
  ::Reflex::Type type_3790 = ::Reflex::PointerBuilder(type_3786c);
  ::Reflex::Type type_3792 = ::Reflex::ReferenceBuilder(type_3786);
  ::Reflex::Type type_3794 = ::Reflex::ReferenceBuilder(type_3786c);
  ::Reflex::Type type_2311 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<ClassD**,std::vector<ClassD*> >");
  ::Reflex::Type type_2312 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<ClassD* const*,std::vector<ClassD*> >");
  ::Reflex::Type type_1608 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<ClassD* const*,std::vector<ClassD*> > >");
  ::Reflex::Type type_1609 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<ClassD**,std::vector<ClassD*> > >");
  ::Reflex::Type type_122 = ::Reflex::TypeBuilder("unsigned int");
  ::Reflex::Type type_854 = ::Reflex::TypedefTypeBuilder("size_t", type_122);
  ::Reflex::Type type_684 = ::Reflex::TypedefTypeBuilder("ptrdiff_t", type_36);
  ::Reflex::Type type_1532 = ::Reflex::TypeBuilder("std::allocator<ClassD*>");
  ::Reflex::Type type_1532c = ::Reflex::ConstBuilder(type_1532);
  ::Reflex::Type type_7257 = ::Reflex::ReferenceBuilder(type_1532c);
  ::Reflex::Type type_1382 = ::Reflex::TypeBuilder("std::vector<ClassD*>");
  ::Reflex::Type type_1382c = ::Reflex::ConstBuilder(type_1382);
  ::Reflex::Type type_7258 = ::Reflex::ReferenceBuilder(type_1382c);
  ::Reflex::Type type_7259 = ::Reflex::ReferenceBuilder(type_1382);
  ::Reflex::Type type_1639 = ::Reflex::TypeBuilder("std::_Vector_base<ClassD,std::allocator<ClassD> >");
  ::Reflex::Type type_3848 = ::Reflex::PointerBuilder(type_29c);
  ::Reflex::Type type_2313 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<ClassD*,std::vector<ClassD> >");
  ::Reflex::Type type_2314 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<const ClassD*,std::vector<ClassD> >");
  ::Reflex::Type type_1610 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<const ClassD*,std::vector<ClassD> > >");
  ::Reflex::Type type_1611 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<ClassD*,std::vector<ClassD> > >");
  ::Reflex::Type type_1533 = ::Reflex::TypeBuilder("std::allocator<ClassD>");
  ::Reflex::Type type_1533c = ::Reflex::ConstBuilder(type_1533);
  ::Reflex::Type type_7261 = ::Reflex::ReferenceBuilder(type_1533c);
  ::Reflex::Type type_1383 = ::Reflex::TypeBuilder("std::vector<ClassD>");
  ::Reflex::Type type_1383c = ::Reflex::ConstBuilder(type_1383);
  ::Reflex::Type type_7262 = ::Reflex::ReferenceBuilder(type_1383c);
  ::Reflex::Type type_7263 = ::Reflex::ReferenceBuilder(type_1383);
  ::Reflex::Type type_1640 = ::Reflex::TypeBuilder("std::_Vector_base<ClassC*,std::allocator<ClassC*> >");
  ::Reflex::Type type_3904 = ::Reflex::PointerBuilder(type_996);
  ::Reflex::Type type_3906 = ::Reflex::PointerBuilder(type_3904);
  ::Reflex::Type type_3904c = ::Reflex::ConstBuilder(type_3904);
  ::Reflex::Type type_3908 = ::Reflex::PointerBuilder(type_3904c);
  ::Reflex::Type type_3910 = ::Reflex::ReferenceBuilder(type_3904);
  ::Reflex::Type type_3912 = ::Reflex::ReferenceBuilder(type_3904c);
  ::Reflex::Type type_2315 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<ClassC**,std::vector<ClassC*> >");
  ::Reflex::Type type_2316 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<ClassC* const*,std::vector<ClassC*> >");
  ::Reflex::Type type_1612 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<ClassC* const*,std::vector<ClassC*> > >");
  ::Reflex::Type type_1613 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<ClassC**,std::vector<ClassC*> > >");
  ::Reflex::Type type_1534 = ::Reflex::TypeBuilder("std::allocator<ClassC*>");
  ::Reflex::Type type_1534c = ::Reflex::ConstBuilder(type_1534);
  ::Reflex::Type type_7265 = ::Reflex::ReferenceBuilder(type_1534c);
  ::Reflex::Type type_1384 = ::Reflex::TypeBuilder("std::vector<ClassC*>");
  ::Reflex::Type type_1384c = ::Reflex::ConstBuilder(type_1384);
  ::Reflex::Type type_7266 = ::Reflex::ReferenceBuilder(type_1384c);
  ::Reflex::Type type_7267 = ::Reflex::ReferenceBuilder(type_1384);
  ::Reflex::Type type_1641 = ::Reflex::TypeBuilder("std::_Vector_base<ClassC,std::allocator<ClassC> >");
  ::Reflex::Type type_3966 = ::Reflex::PointerBuilder(type_996c);
  ::Reflex::Type type_2317 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<ClassC*,std::vector<ClassC> >");
  ::Reflex::Type type_2318 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<const ClassC*,std::vector<ClassC> >");
  ::Reflex::Type type_1614 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<const ClassC*,std::vector<ClassC> > >");
  ::Reflex::Type type_1615 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<ClassC*,std::vector<ClassC> > >");
  ::Reflex::Type type_1535 = ::Reflex::TypeBuilder("std::allocator<ClassC>");
  ::Reflex::Type type_1535c = ::Reflex::ConstBuilder(type_1535);
  ::Reflex::Type type_7269 = ::Reflex::ReferenceBuilder(type_1535c);
  ::Reflex::Type type_1385 = ::Reflex::TypeBuilder("std::vector<ClassC>");
  ::Reflex::Type type_1385c = ::Reflex::ConstBuilder(type_1385);
  ::Reflex::Type type_7270 = ::Reflex::ReferenceBuilder(type_1385c);
  ::Reflex::Type type_7271 = ::Reflex::ReferenceBuilder(type_1385);
  ::Reflex::Type type_1642 = ::Reflex::TypeBuilder("std::_Vector_base<ClassB*,std::allocator<ClassB*> >");
  ::Reflex::Type type_4022 = ::Reflex::PointerBuilder(type_995);
  ::Reflex::Type type_4024 = ::Reflex::PointerBuilder(type_4022);
  ::Reflex::Type type_4022c = ::Reflex::ConstBuilder(type_4022);
  ::Reflex::Type type_4026 = ::Reflex::PointerBuilder(type_4022c);
  ::Reflex::Type type_4028 = ::Reflex::ReferenceBuilder(type_4022);
  ::Reflex::Type type_4030 = ::Reflex::ReferenceBuilder(type_4022c);
  ::Reflex::Type type_2319 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<ClassB**,std::vector<ClassB*> >");
  ::Reflex::Type type_2320 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<ClassB* const*,std::vector<ClassB*> >");
  ::Reflex::Type type_1616 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<ClassB* const*,std::vector<ClassB*> > >");
  ::Reflex::Type type_1617 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<ClassB**,std::vector<ClassB*> > >");
  ::Reflex::Type type_1536 = ::Reflex::TypeBuilder("std::allocator<ClassB*>");
  ::Reflex::Type type_1536c = ::Reflex::ConstBuilder(type_1536);
  ::Reflex::Type type_7273 = ::Reflex::ReferenceBuilder(type_1536c);
  ::Reflex::Type type_1386 = ::Reflex::TypeBuilder("std::vector<ClassB*>");
  ::Reflex::Type type_1386c = ::Reflex::ConstBuilder(type_1386);
  ::Reflex::Type type_7274 = ::Reflex::ReferenceBuilder(type_1386c);
  ::Reflex::Type type_7275 = ::Reflex::ReferenceBuilder(type_1386);
  ::Reflex::Type type_1643 = ::Reflex::TypeBuilder("std::_Vector_base<ClassB,std::allocator<ClassB> >");
  ::Reflex::Type type_4084 = ::Reflex::PointerBuilder(type_995c);
  ::Reflex::Type type_2321 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<ClassB*,std::vector<ClassB> >");
  ::Reflex::Type type_2322 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<const ClassB*,std::vector<ClassB> >");
  ::Reflex::Type type_1618 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<const ClassB*,std::vector<ClassB> > >");
  ::Reflex::Type type_1619 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<ClassB*,std::vector<ClassB> > >");
  ::Reflex::Type type_1537 = ::Reflex::TypeBuilder("std::allocator<ClassB>");
  ::Reflex::Type type_1537c = ::Reflex::ConstBuilder(type_1537);
  ::Reflex::Type type_7277 = ::Reflex::ReferenceBuilder(type_1537c);
  ::Reflex::Type type_1387 = ::Reflex::TypeBuilder("std::vector<ClassB>");
  ::Reflex::Type type_1387c = ::Reflex::ConstBuilder(type_1387);
  ::Reflex::Type type_7278 = ::Reflex::ReferenceBuilder(type_1387c);
  ::Reflex::Type type_7279 = ::Reflex::ReferenceBuilder(type_1387);
  ::Reflex::Type type_1646 = ::Reflex::TypeBuilder("std::_Vector_base<ClassA*,std::allocator<ClassA*> >");
  ::Reflex::Type type_4140 = ::Reflex::PointerBuilder(type_994);
  ::Reflex::Type type_4142 = ::Reflex::PointerBuilder(type_4140);
  ::Reflex::Type type_4140c = ::Reflex::ConstBuilder(type_4140);
  ::Reflex::Type type_4144 = ::Reflex::PointerBuilder(type_4140c);
  ::Reflex::Type type_4146 = ::Reflex::ReferenceBuilder(type_4140);
  ::Reflex::Type type_4148 = ::Reflex::ReferenceBuilder(type_4140c);
  ::Reflex::Type type_2327 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<ClassA**,std::vector<ClassA*> >");
  ::Reflex::Type type_2328 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<ClassA* const*,std::vector<ClassA*> >");
  ::Reflex::Type type_1624 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<ClassA* const*,std::vector<ClassA*> > >");
  ::Reflex::Type type_1625 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<ClassA**,std::vector<ClassA*> > >");
  ::Reflex::Type type_1538 = ::Reflex::TypeBuilder("std::allocator<ClassA*>");
  ::Reflex::Type type_1538c = ::Reflex::ConstBuilder(type_1538);
  ::Reflex::Type type_7281 = ::Reflex::ReferenceBuilder(type_1538c);
  ::Reflex::Type type_1388 = ::Reflex::TypeBuilder("std::vector<ClassA*>");
  ::Reflex::Type type_1388c = ::Reflex::ConstBuilder(type_1388);
  ::Reflex::Type type_7282 = ::Reflex::ReferenceBuilder(type_1388c);
  ::Reflex::Type type_7283 = ::Reflex::ReferenceBuilder(type_1388);
  ::Reflex::Type type_1648 = ::Reflex::TypeBuilder("std::_Vector_base<ClassA,std::allocator<ClassA> >");
  ::Reflex::Type type_4202 = ::Reflex::PointerBuilder(type_994c);
  ::Reflex::Type type_2331 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<ClassA*,std::vector<ClassA> >");
  ::Reflex::Type type_2332 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<const ClassA*,std::vector<ClassA> >");
  ::Reflex::Type type_1628 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<const ClassA*,std::vector<ClassA> > >");
  ::Reflex::Type type_1629 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<ClassA*,std::vector<ClassA> > >");
  ::Reflex::Type type_1539 = ::Reflex::TypeBuilder("std::allocator<ClassA>");
  ::Reflex::Type type_1539c = ::Reflex::ConstBuilder(type_1539);
  ::Reflex::Type type_7285 = ::Reflex::ReferenceBuilder(type_1539c);
  ::Reflex::Type type_1389 = ::Reflex::TypeBuilder("std::vector<ClassA>");
  ::Reflex::Type type_1389c = ::Reflex::ConstBuilder(type_1389);
  ::Reflex::Type type_7286 = ::Reflex::ReferenceBuilder(type_1389c);
  ::Reflex::Type type_7287 = ::Reflex::ReferenceBuilder(type_1389);
  ::Reflex::Type type_1430 = ::Reflex::TypeBuilder("std::pair<int,double>");
  ::Reflex::Type type_4263 = ::Reflex::ReferenceBuilder(type_1430);
  ::Reflex::Type type_1430c = ::Reflex::ConstBuilder(type_1430);
  ::Reflex::Type type_4265 = ::Reflex::ReferenceBuilder(type_1430c);
  ::Reflex::Type type_36c = ::Reflex::ConstBuilder(type_36);
  ::Reflex::Type type_7343 = ::Reflex::ReferenceBuilder(type_36c);
  ::Reflex::Type type_1818c = ::Reflex::ConstBuilder(type_1818);
  ::Reflex::Type type_4323 = ::Reflex::ReferenceBuilder(type_1818c);
  ::Reflex::Type type_1644 = ::Reflex::TypeBuilder("std::_Vector_base<std::pair<int,double>,std::allocator<std::pair<int,double> > >");
  ::Reflex::Type type_4259 = ::Reflex::PointerBuilder(type_1430);
  ::Reflex::Type type_4261 = ::Reflex::PointerBuilder(type_1430c);
  ::Reflex::Type type_2323 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<std::pair<int,double>*,std::vector<std::pair<int,double> > >");
  ::Reflex::Type type_2324 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<const std::pair<int,double>*,std::vector<std::pair<int,double> > >");
  ::Reflex::Type type_1620 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<const std::pair<int,double>*,std::vector<std::pair<int,double> > > >");
  ::Reflex::Type type_1621 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<std::pair<int,double>*,std::vector<std::pair<int,double> > > >");
  ::Reflex::Type type_1540 = ::Reflex::TypeBuilder("std::allocator<std::pair<int,double> >");
  ::Reflex::Type type_1540c = ::Reflex::ConstBuilder(type_1540);
  ::Reflex::Type type_7289 = ::Reflex::ReferenceBuilder(type_1540c);
  ::Reflex::Type type_1390 = ::Reflex::TypeBuilder("std::vector<std::pair<int,double> >");
  ::Reflex::Type type_1390c = ::Reflex::ConstBuilder(type_1390);
  ::Reflex::Type type_7290 = ::Reflex::ReferenceBuilder(type_1390c);
  ::Reflex::Type type_7291 = ::Reflex::ReferenceBuilder(type_1390);
  ::Reflex::Type type_1645 = ::Reflex::TypeBuilder("std::_Vector_base<double,std::allocator<double> >");
  ::Reflex::Type type_2215 = ::Reflex::PointerBuilder(type_1818);
  ::Reflex::Type type_4319 = ::Reflex::PointerBuilder(type_1818c);
  ::Reflex::Type type_4321 = ::Reflex::ReferenceBuilder(type_1818);
  ::Reflex::Type type_2325 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<double*,std::vector<double> >");
  ::Reflex::Type type_2326 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<const double*,std::vector<double> >");
  ::Reflex::Type type_1622 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<const double*,std::vector<double> > >");
  ::Reflex::Type type_1623 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<double*,std::vector<double> > >");
  ::Reflex::Type type_1541 = ::Reflex::TypeBuilder("std::allocator<double>");
  ::Reflex::Type type_1541c = ::Reflex::ConstBuilder(type_1541);
  ::Reflex::Type type_7293 = ::Reflex::ReferenceBuilder(type_1541c);
  ::Reflex::Type type_1391 = ::Reflex::TypeBuilder("std::vector<double>");
  ::Reflex::Type type_1391c = ::Reflex::ConstBuilder(type_1391);
  ::Reflex::Type type_7294 = ::Reflex::ReferenceBuilder(type_1391c);
  ::Reflex::Type type_7295 = ::Reflex::ReferenceBuilder(type_1391);
  ::Reflex::Type type_1431 = ::Reflex::TypeBuilder("std::pair<int,float>");
  ::Reflex::Type type_4380 = ::Reflex::ReferenceBuilder(type_1431);
  ::Reflex::Type type_1431c = ::Reflex::ConstBuilder(type_1431);
  ::Reflex::Type type_4382 = ::Reflex::ReferenceBuilder(type_1431c);
  ::Reflex::Type type_1823c = ::Reflex::ConstBuilder(type_1823);
  ::Reflex::Type type_4440 = ::Reflex::ReferenceBuilder(type_1823c);
  ::Reflex::Type type_1647 = ::Reflex::TypeBuilder("std::_Vector_base<std::pair<int,float>,std::allocator<std::pair<int,float> > >");
  ::Reflex::Type type_4376 = ::Reflex::PointerBuilder(type_1431);
  ::Reflex::Type type_4378 = ::Reflex::PointerBuilder(type_1431c);
  ::Reflex::Type type_2329 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<std::pair<int,float>*,std::vector<std::pair<int,float> > >");
  ::Reflex::Type type_2330 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<const std::pair<int,float>*,std::vector<std::pair<int,float> > >");
  ::Reflex::Type type_1626 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<const std::pair<int,float>*,std::vector<std::pair<int,float> > > >");
  ::Reflex::Type type_1627 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<std::pair<int,float>*,std::vector<std::pair<int,float> > > >");
  ::Reflex::Type type_1542 = ::Reflex::TypeBuilder("std::allocator<std::pair<int,float> >");
  ::Reflex::Type type_1542c = ::Reflex::ConstBuilder(type_1542);
  ::Reflex::Type type_7297 = ::Reflex::ReferenceBuilder(type_1542c);
  ::Reflex::Type type_1392 = ::Reflex::TypeBuilder("std::vector<std::pair<int,float> >");
  ::Reflex::Type type_1392c = ::Reflex::ConstBuilder(type_1392);
  ::Reflex::Type type_7298 = ::Reflex::ReferenceBuilder(type_1392c);
  ::Reflex::Type type_7299 = ::Reflex::ReferenceBuilder(type_1392);
  ::Reflex::Type type_1649 = ::Reflex::TypeBuilder("std::_Vector_base<float,std::allocator<float> >");
  ::Reflex::Type type_2423 = ::Reflex::PointerBuilder(type_1823);
  ::Reflex::Type type_4436 = ::Reflex::PointerBuilder(type_1823c);
  ::Reflex::Type type_4438 = ::Reflex::ReferenceBuilder(type_1823);
  ::Reflex::Type type_2333 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<float*,std::vector<float> >");
  ::Reflex::Type type_2334 = ::Reflex::TypeBuilder("__gnu_cxx::__normal_iterator<const float*,std::vector<float> >");
  ::Reflex::Type type_1630 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<const float*,std::vector<float> > >");
  ::Reflex::Type type_1631 = ::Reflex::TypeBuilder("std::reverse_iterator<__gnu_cxx::__normal_iterator<float*,std::vector<float> > >");
  ::Reflex::Type type_1543 = ::Reflex::TypeBuilder("std::allocator<float>");
  ::Reflex::Type type_1543c = ::Reflex::ConstBuilder(type_1543);
  ::Reflex::Type type_7301 = ::Reflex::ReferenceBuilder(type_1543c);
  ::Reflex::Type type_1393 = ::Reflex::TypeBuilder("std::vector<float>");
  ::Reflex::Type type_1393c = ::Reflex::ConstBuilder(type_1393);
  ::Reflex::Type type_7302 = ::Reflex::ReferenceBuilder(type_1393c);
  ::Reflex::Type type_7303 = ::Reflex::ReferenceBuilder(type_1393);
} // unnamed namespace

#ifndef __CINT__

// Shadow classes to obtain the data member offsets 
namespace __shadow__ {
#ifdef __ClassD
#undef __ClassD
#endif
class __ClassD {
  public:
  __ClassD();
  virtual ~__ClassD() throw();
  float m_c;
  ::ClassAIns m_d;
  int m_e;
};
#ifdef __ClassABase
#undef __ClassABase
#endif
class __ClassABase {
  public:
  __ClassABase();
  virtual ~__ClassABase() throw();
  int m_a;
  double m_b;
};
#ifdef __ClassA
#undef __ClassA
#endif
class __ClassA : public ::ClassABase {
  public:
  __ClassA();
  virtual ~__ClassA() throw();
  float m_c;
  ::ClassAIns m_d;
  int m_e;
  int m_unit;
  bool m_md_set;
};
#ifdef __ClassB
#undef __ClassB
#endif
class __ClassB : public ::ClassA {
  public:
  __ClassB();
  virtual ~__ClassB() throw();
  short m_f;
  float m_g;
};
#ifdef __ClassC
#undef __ClassC
#endif
class __ClassC : public ::ClassABase {
  public:
  __ClassC();
  virtual ~__ClassC() throw();
  double m_f;
  float m_g;
};
#ifdef __ClassAIns
#undef __ClassAIns
#endif
class __ClassAIns {
  public:
  __ClassAIns();
  int m_a;
  double m_b;
  int m_unit;
};
#ifdef __std__vector_ClassDp_
#undef __std__vector_ClassDp_
#endif
class __std__vector_ClassDp_ : protected ::std::_Vector_base<ClassD*,std::allocator<ClassD*> > {
  public:
  __std__vector_ClassDp_();
};
#ifdef __std__vector_ClassD_
#undef __std__vector_ClassD_
#endif
class __std__vector_ClassD_ : protected ::std::_Vector_base<ClassD,std::allocator<ClassD> > {
  public:
  __std__vector_ClassD_();
};
#ifdef __std__vector_ClassCp_
#undef __std__vector_ClassCp_
#endif
class __std__vector_ClassCp_ : protected ::std::_Vector_base<ClassC*,std::allocator<ClassC*> > {
  public:
  __std__vector_ClassCp_();
};
#ifdef __std__vector_ClassC_
#undef __std__vector_ClassC_
#endif
class __std__vector_ClassC_ : protected ::std::_Vector_base<ClassC,std::allocator<ClassC> > {
  public:
  __std__vector_ClassC_();
};
#ifdef __std__vector_ClassBp_
#undef __std__vector_ClassBp_
#endif
class __std__vector_ClassBp_ : protected ::std::_Vector_base<ClassB*,std::allocator<ClassB*> > {
  public:
  __std__vector_ClassBp_();
};
#ifdef __std__vector_ClassB_
#undef __std__vector_ClassB_
#endif
class __std__vector_ClassB_ : protected ::std::_Vector_base<ClassB,std::allocator<ClassB> > {
  public:
  __std__vector_ClassB_();
};
#ifdef __std__vector_ClassAp_
#undef __std__vector_ClassAp_
#endif
class __std__vector_ClassAp_ : protected ::std::_Vector_base<ClassA*,std::allocator<ClassA*> > {
  public:
  __std__vector_ClassAp_();
};
#ifdef __std__vector_ClassA_
#undef __std__vector_ClassA_
#endif
class __std__vector_ClassA_ : protected ::std::_Vector_base<ClassA,std::allocator<ClassA> > {
  public:
  __std__vector_ClassA_();
};
#ifdef __std__pair_int_double_
#undef __std__pair_int_double_
#endif
struct __std__pair_int_double_ {
  public:
  __std__pair_int_double_();
  int first;
  double second;
};
#ifdef __std__vector_std__pair_int_double_s_
#undef __std__vector_std__pair_int_double_s_
#endif
class __std__vector_std__pair_int_double_s_ : protected ::std::_Vector_base<std::pair<int,double>,std::allocator<std::pair<int,double> > > {
  public:
  __std__vector_std__pair_int_double_s_();
};
#ifdef __std__vector_double_
#undef __std__vector_double_
#endif
class __std__vector_double_ : protected ::std::_Vector_base<double,std::allocator<double> > {
  public:
  __std__vector_double_();
};
#ifdef __std__pair_int_float_
#undef __std__pair_int_float_
#endif
struct __std__pair_int_float_ {
  public:
  __std__pair_int_float_();
  int first;
  float second;
};
#ifdef __std__vector_std__pair_int_float_s_
#undef __std__vector_std__pair_int_float_s_
#endif
class __std__vector_std__pair_int_float_s_ : protected ::std::_Vector_base<std::pair<int,float>,std::allocator<std::pair<int,float> > > {
  public:
  __std__vector_std__pair_int_float_s_();
};
#ifdef __std__vector_float_
#undef __std__vector_float_
#endif
class __std__vector_float_ : protected ::std::_Vector_base<float,std::allocator<float> > {
  public:
  __std__vector_float_();
};
}


#endif // __CINT__
namespace {
//------Stub functions for class ClassD -------------------------------
static void operator_1858( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::ClassD*)o)->operator=)(*(const ::ClassD*)arg[0]);
else   (((::ClassD*)o)->operator=)(*(const ::ClassD*)arg[0]);
}

static void constructor_1859( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::ClassD(*(const ::ClassD*)arg[0]);
  else ::new(mem) ::ClassD(*(const ::ClassD*)arg[0]);
}

static void constructor_1860( void* retaddr, void* mem, const std::vector<void*>&, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::ClassD();
  else ::new(mem) ::ClassD();
}

static void destructor_1861(void*, void * o, const std::vector<void*>&, void *) {
  (((::ClassD*)o)->::ClassD::~ClassD)();
}
static void method_newdel_29( void* retaddr, void*, const std::vector<void*>&, void*)
{
  static ::Reflex::NewDelFunctions s_funcs;
  s_funcs.fNew         = ::Reflex::NewDelFunctionsT< ::ClassD >::new_T;
  s_funcs.fNewArray    = ::Reflex::NewDelFunctionsT< ::ClassD >::newArray_T;
  s_funcs.fDelete      = ::Reflex::NewDelFunctionsT< ::ClassD >::delete_T;
  s_funcs.fDeleteArray = ::Reflex::NewDelFunctionsT< ::ClassD >::deleteArray_T;
  s_funcs.fDestructor  = ::Reflex::NewDelFunctionsT< ::ClassD >::destruct_T;
  if (retaddr) *(::Reflex::NewDelFunctions**)retaddr = &s_funcs;
}

//------Dictionary for class ClassD -------------------------------
void __ClassD_db_datamem(Reflex::Class*);
void __ClassD_db_funcmem(Reflex::Class*);
Reflex::GenreflexMemberBuilder __ClassD_datamem_bld(&__ClassD_db_datamem);
Reflex::GenreflexMemberBuilder __ClassD_funcmem_bld(&__ClassD_db_funcmem);
void __ClassD_dict() {
  ::Reflex::ClassBuilder("ClassD", typeid(::ClassD), sizeof(::ClassD), ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::VIRTUAL, ::Reflex::CLASS)
  .AddProperty("ClassVersion", "3")
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3850, type_3852), "operator=", operator_1858, 0, "", ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_3852), "ClassD", constructor_1859, 0, "", ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "ClassD", constructor_1860, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "~ClassD", destructor_1861, 0, 0, ::Reflex::PUBLIC | ::Reflex::VIRTUAL | ::Reflex::DESTRUCTOR )
  .AddFunctionMember<void*(void)>("__getNewDelFunctions", method_newdel_29, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddOnDemandDataMemberBuilder(&__ClassD_datamem_bld);
}

//------Delayed data member builder for class ClassD -------------------
void __ClassD_db_datamem(Reflex::Class* cl) {
  ::Reflex::ClassBuilder(cl)
  .AddDataMember(type_1823, "m_c", OffsetOf(__shadow__::__ClassD, m_c), ::Reflex::PRIVATE)
  .AddDataMember(type_1031, "m_d", OffsetOf(__shadow__::__ClassD, m_d), ::Reflex::PRIVATE)
  .AddDataMember(type_36, "m_e", OffsetOf(__shadow__::__ClassD, m_e), ::Reflex::PRIVATE);
}
//------Delayed function member builder for class ClassD -------------------
void __ClassD_db_funcmem(Reflex::Class*) {

}
//------Stub functions for class ClassABase -------------------------------
static void operator_2534( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::ClassABase*)o)->operator=)(*(const ::ClassABase*)arg[0]);
else   (((::ClassABase*)o)->operator=)(*(const ::ClassABase*)arg[0]);
}

static void constructor_2535( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::ClassABase(*(const ::ClassABase*)arg[0]);
  else ::new(mem) ::ClassABase(*(const ::ClassABase*)arg[0]);
}

static void constructor_2536( void* retaddr, void* mem, const std::vector<void*>&, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::ClassABase();
  else ::new(mem) ::ClassABase();
}

static void destructor_2537(void*, void * o, const std::vector<void*>&, void *) {
  (((::ClassABase*)o)->::ClassABase::~ClassABase)();
}
static void method_newdel_776( void* retaddr, void*, const std::vector<void*>&, void*)
{
  static ::Reflex::NewDelFunctions s_funcs;
  s_funcs.fNew         = ::Reflex::NewDelFunctionsT< ::ClassABase >::new_T;
  s_funcs.fNewArray    = ::Reflex::NewDelFunctionsT< ::ClassABase >::newArray_T;
  s_funcs.fDelete      = ::Reflex::NewDelFunctionsT< ::ClassABase >::delete_T;
  s_funcs.fDeleteArray = ::Reflex::NewDelFunctionsT< ::ClassABase >::deleteArray_T;
  s_funcs.fDestructor  = ::Reflex::NewDelFunctionsT< ::ClassABase >::destruct_T;
  if (retaddr) *(::Reflex::NewDelFunctions**)retaddr = &s_funcs;
}

//------Dictionary for class ClassABase -------------------------------
void __ClassABase_db_datamem(Reflex::Class*);
void __ClassABase_db_funcmem(Reflex::Class*);
Reflex::GenreflexMemberBuilder __ClassABase_datamem_bld(&__ClassABase_db_datamem);
Reflex::GenreflexMemberBuilder __ClassABase_funcmem_bld(&__ClassABase_db_funcmem);
void __ClassABase_dict() {
  ::Reflex::ClassBuilder("ClassABase", typeid(::ClassABase), sizeof(::ClassABase), ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::VIRTUAL, ::Reflex::CLASS)
  .AddProperty("ClassVersion", "2")
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_7084, type_7085), "operator=", operator_2534, 0, "", ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7085), "ClassABase", constructor_2535, 0, "", ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "ClassABase", constructor_2536, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "~ClassABase", destructor_2537, 0, 0, ::Reflex::PUBLIC | ::Reflex::VIRTUAL | ::Reflex::DESTRUCTOR )
  .AddFunctionMember<void*(void)>("__getNewDelFunctions", method_newdel_776, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddOnDemandDataMemberBuilder(&__ClassABase_datamem_bld);
}

//------Delayed data member builder for class ClassABase -------------------
void __ClassABase_db_datamem(Reflex::Class* cl) {
  ::Reflex::ClassBuilder(cl)
  .AddDataMember(type_36, "m_a", OffsetOf(__shadow__::__ClassABase, m_a), ::Reflex::PRIVATE)
  .AddDataMember(type_1818, "m_b", OffsetOf(__shadow__::__ClassABase, m_b), ::Reflex::PRIVATE);
}
//------Delayed function member builder for class ClassABase -------------------
void __ClassABase_db_funcmem(Reflex::Class*) {

}
//------Stub functions for class ClassA -------------------------------
static void operator_2557( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::ClassA*)o)->operator=)(*(const ::ClassA*)arg[0]);
else   (((::ClassA*)o)->operator=)(*(const ::ClassA*)arg[0]);
}

static void constructor_2558( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::ClassA(*(const ::ClassA*)arg[0]);
  else ::new(mem) ::ClassA(*(const ::ClassA*)arg[0]);
}

static void constructor_2559( void* retaddr, void* mem, const std::vector<void*>&, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::ClassA();
  else ::new(mem) ::ClassA();
}

static void destructor_2560(void*, void * o, const std::vector<void*>&, void *) {
  (((::ClassA*)o)->::ClassA::~ClassA)();
}
static void method_2561( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::ClassA*)o)->SetMdUnit)(*(int*)arg[0]);
}

static void method_2562( void*, void* o, const std::vector<void*>&, void*)
{
  (((::ClassA*)o)->Print)();
}

static void method_newdel_994( void* retaddr, void*, const std::vector<void*>&, void*)
{
  static ::Reflex::NewDelFunctions s_funcs;
  s_funcs.fNew         = ::Reflex::NewDelFunctionsT< ::ClassA >::new_T;
  s_funcs.fNewArray    = ::Reflex::NewDelFunctionsT< ::ClassA >::newArray_T;
  s_funcs.fDelete      = ::Reflex::NewDelFunctionsT< ::ClassA >::delete_T;
  s_funcs.fDeleteArray = ::Reflex::NewDelFunctionsT< ::ClassA >::deleteArray_T;
  s_funcs.fDestructor  = ::Reflex::NewDelFunctionsT< ::ClassA >::destruct_T;
  if (retaddr) *(::Reflex::NewDelFunctions**)retaddr = &s_funcs;
}

static void method_x3( void* retaddr, void*, const std::vector<void*>&, void*)
{
  typedef std::vector<std::pair< ::Reflex::Base, int> > Bases_t;
  static Bases_t s_bases;
  if ( !s_bases.size() ) {
    s_bases.push_back(std::make_pair(::Reflex::Base( ::Reflex::TypeBuilder("ClassABase"), ::Reflex::BaseOffset< ::ClassA,::ClassABase >::Get(),::Reflex::PUBLIC), 0));
  }
  if (retaddr) *(Bases_t**)retaddr = &s_bases;
}

void read___ClassA_0( char *target, TVirtualObject *oldObj )
{
  //--- Variables added by the code generator ---
#if 0
  static int id_m_unit = oldObj->GetId("m_unit");
#endif
  struct __ClassA_Onfile {
    int &m_unit;
    __ClassA_Onfile( int &onfile_m_unit ): m_unit(onfile_m_unit){}
  };
  static Long_t offset_Onfile___ClassA_m_unit = oldObj->GetClass()->GetDataMemberOffset("m_unit");
  char *onfile_add = (char*)oldObj->GetObject();
  __ClassA_Onfile onfile(
         *(int*)(onfile_add+offset_Onfile___ClassA_m_unit) );

  int &m_unit = *(int*)(target + OffsetOf(__shadow__::__ClassA, m_unit));

  ClassA* newObj = (ClassA*)target;
  //--- User's code ---
   { m_unit = 10*onfile.m_unit; } 
  
}

void read___ClassA_1( char *target, TVirtualObject *oldObj )
{
  //--- Variables added by the code generator ---
#if 0
  static int id_m_unit = oldObj->GetId("m_unit");
#endif
  struct __ClassA_Onfile {
    int &m_unit;
    __ClassA_Onfile( int &onfile_m_unit ): m_unit(onfile_m_unit){}
  };
  static Long_t offset_Onfile___ClassA_m_unit = oldObj->GetClass()->GetDataMemberOffset("m_unit");
  char *onfile_add = (char*)oldObj->GetObject();
  __ClassA_Onfile onfile(
         *(int*)(onfile_add+offset_Onfile___ClassA_m_unit) );

  bool &m_md_set = *(bool*)(target + OffsetOf(__shadow__::__ClassA, m_md_set));

  ClassA* newObj = (ClassA*)target;
  //--- User's code ---
   { newObj->SetMdUnit( 10*onfile.m_unit ); m_md_set = true; newObj->Print(); } 
  
}

//------Dictionary for class ClassA -------------------------------
void __ClassA_db_datamem(Reflex::Class*);
void __ClassA_db_funcmem(Reflex::Class*);
Reflex::GenreflexMemberBuilder __ClassA_datamem_bld(&__ClassA_db_datamem);
Reflex::GenreflexMemberBuilder __ClassA_funcmem_bld(&__ClassA_db_funcmem);
void __ClassA_dict() {
  ROOT::TSchemaHelper* rule;
  // the io read rules
  std::vector<ROOT::TSchemaHelper> readrules(2);
  rule = &readrules[0];
  rule->fSourceClass = "ClassA";
  rule->fTarget      = "m_unit";
  rule->fSource      = "int m_unit";
  rule->fFunctionPtr = (void *)read___ClassA_0;
  rule->fCode        = "\n   { m_unit = 10*onfile.m_unit; } \n  ";
  rule->fVersion     = "[2]";
  rule = &readrules[1];
  rule->fSourceClass = "ClassA";
  rule->fTarget      = "m_md_set";
  rule->fSource      = "int m_unit";
  rule->fFunctionPtr = (void *)read___ClassA_1;
  rule->fCode        = "\n   { newObj->SetMdUnit( 10*onfile.m_unit ); m_md_set = true; newObj->Print(); } \n  ";
  rule->fVersion     = "[2]";


  ::Reflex::ClassBuilder("ClassA", typeid(::ClassA), sizeof(::ClassA), ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::VIRTUAL, ::Reflex::CLASS)
  .AddProperty("ClassVersion", "3")
  .AddProperty("ioread", readrules )
  .AddBase(type_776, ::Reflex::BaseOffset< ::ClassA, ::ClassABase >::Get(), ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4204, type_4206), "operator=", operator_2557, 0, "", ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_4206), "ClassA", constructor_2558, 0, "", ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "ClassA", constructor_2559, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "~ClassA", destructor_2560, 0, 0, ::Reflex::PUBLIC | ::Reflex::VIRTUAL | ::Reflex::DESTRUCTOR )
  .AddFunctionMember<void*(void)>("__getNewDelFunctions", method_newdel_994, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("__getBasesTable", method_x3, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddOnDemandDataMemberBuilder(&__ClassA_datamem_bld)
  .AddOnDemandFunctionMemberBuilder(&__ClassA_funcmem_bld);
}

//------Delayed data member builder for class ClassA -------------------
void __ClassA_db_datamem(Reflex::Class* cl) {
  ::Reflex::ClassBuilder(cl)
  .AddDataMember(type_1823, "m_c", OffsetOf(__shadow__::__ClassA, m_c), ::Reflex::PRIVATE)
  .AddDataMember(type_1031, "m_d", OffsetOf(__shadow__::__ClassA, m_d), ::Reflex::PRIVATE)
  .AddDataMember(type_36, "m_e", OffsetOf(__shadow__::__ClassA, m_e), ::Reflex::PRIVATE)
  .AddDataMember(type_36, "m_unit", OffsetOf(__shadow__::__ClassA, m_unit), ::Reflex::PRIVATE)
  .AddDataMember(type_2653, "m_md_set", OffsetOf(__shadow__::__ClassA, m_md_set), ::Reflex::PRIVATE | ::Reflex::TRANSIENT);
}
//------Delayed function member builder for class ClassA -------------------
void __ClassA_db_funcmem(Reflex::Class* cl) {
  ::Reflex::ClassBuilder(cl)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_36), "SetMdUnit", method_2561, 0, "unit", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "Print", method_2562, 0, 0, ::Reflex::PUBLIC);
}
//------Stub functions for class ClassB -------------------------------
static void operator_2565( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::ClassB*)o)->operator=)(*(const ::ClassB*)arg[0]);
else   (((::ClassB*)o)->operator=)(*(const ::ClassB*)arg[0]);
}

static void constructor_2566( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::ClassB(*(const ::ClassB*)arg[0]);
  else ::new(mem) ::ClassB(*(const ::ClassB*)arg[0]);
}

static void constructor_2567( void* retaddr, void* mem, const std::vector<void*>&, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::ClassB();
  else ::new(mem) ::ClassB();
}

static void destructor_2568(void*, void * o, const std::vector<void*>&, void *) {
  (((::ClassB*)o)->::ClassB::~ClassB)();
}
static void method_newdel_995( void* retaddr, void*, const std::vector<void*>&, void*)
{
  static ::Reflex::NewDelFunctions s_funcs;
  s_funcs.fNew         = ::Reflex::NewDelFunctionsT< ::ClassB >::new_T;
  s_funcs.fNewArray    = ::Reflex::NewDelFunctionsT< ::ClassB >::newArray_T;
  s_funcs.fDelete      = ::Reflex::NewDelFunctionsT< ::ClassB >::delete_T;
  s_funcs.fDeleteArray = ::Reflex::NewDelFunctionsT< ::ClassB >::deleteArray_T;
  s_funcs.fDestructor  = ::Reflex::NewDelFunctionsT< ::ClassB >::destruct_T;
  if (retaddr) *(::Reflex::NewDelFunctions**)retaddr = &s_funcs;
}

static void method_x5( void* retaddr, void*, const std::vector<void*>&, void*)
{
  typedef std::vector<std::pair< ::Reflex::Base, int> > Bases_t;
  static Bases_t s_bases;
  if ( !s_bases.size() ) {
    s_bases.push_back(std::make_pair(::Reflex::Base( ::Reflex::TypeBuilder("ClassA"), ::Reflex::BaseOffset< ::ClassB,::ClassA >::Get(),::Reflex::PUBLIC), 0));
    s_bases.push_back(std::make_pair(::Reflex::Base( ::Reflex::TypeBuilder("ClassABase"), ::Reflex::BaseOffset< ::ClassB,::ClassABase >::Get(),::Reflex::PUBLIC), 1));
  }
  if (retaddr) *(Bases_t**)retaddr = &s_bases;
}

//------Dictionary for class ClassB -------------------------------
void __ClassB_db_datamem(Reflex::Class*);
void __ClassB_db_funcmem(Reflex::Class*);
Reflex::GenreflexMemberBuilder __ClassB_datamem_bld(&__ClassB_db_datamem);
Reflex::GenreflexMemberBuilder __ClassB_funcmem_bld(&__ClassB_db_funcmem);
void __ClassB_dict() {
  ::Reflex::ClassBuilder("ClassB", typeid(::ClassB), sizeof(::ClassB), ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::VIRTUAL, ::Reflex::CLASS)
  .AddProperty("ClassVersion", "3")
  .AddBase(type_994, ::Reflex::BaseOffset< ::ClassB, ::ClassA >::Get(), ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4086, type_4088), "operator=", operator_2565, 0, "", ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_4088), "ClassB", constructor_2566, 0, "", ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "ClassB", constructor_2567, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "~ClassB", destructor_2568, 0, 0, ::Reflex::PUBLIC | ::Reflex::VIRTUAL | ::Reflex::DESTRUCTOR )
  .AddFunctionMember<void*(void)>("__getNewDelFunctions", method_newdel_995, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("__getBasesTable", method_x5, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddOnDemandDataMemberBuilder(&__ClassB_datamem_bld);
}

//------Delayed data member builder for class ClassB -------------------
void __ClassB_db_datamem(Reflex::Class* cl) {
  ::Reflex::ClassBuilder(cl)
  .AddDataMember(type_174, "m_f", OffsetOf(__shadow__::__ClassB, m_f), ::Reflex::PRIVATE)
  .AddDataMember(type_1823, "m_g", OffsetOf(__shadow__::__ClassB, m_g), ::Reflex::PRIVATE);
}
//------Delayed function member builder for class ClassB -------------------
void __ClassB_db_funcmem(Reflex::Class*) {

}
//------Stub functions for class ClassC -------------------------------
static void operator_2571( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::ClassC*)o)->operator=)(*(const ::ClassC*)arg[0]);
else   (((::ClassC*)o)->operator=)(*(const ::ClassC*)arg[0]);
}

static void constructor_2572( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::ClassC(*(const ::ClassC*)arg[0]);
  else ::new(mem) ::ClassC(*(const ::ClassC*)arg[0]);
}

static void constructor_2573( void* retaddr, void* mem, const std::vector<void*>&, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::ClassC();
  else ::new(mem) ::ClassC();
}

static void destructor_2574(void*, void * o, const std::vector<void*>&, void *) {
  (((::ClassC*)o)->::ClassC::~ClassC)();
}
static void method_newdel_996( void* retaddr, void*, const std::vector<void*>&, void*)
{
  static ::Reflex::NewDelFunctions s_funcs;
  s_funcs.fNew         = ::Reflex::NewDelFunctionsT< ::ClassC >::new_T;
  s_funcs.fNewArray    = ::Reflex::NewDelFunctionsT< ::ClassC >::newArray_T;
  s_funcs.fDelete      = ::Reflex::NewDelFunctionsT< ::ClassC >::delete_T;
  s_funcs.fDeleteArray = ::Reflex::NewDelFunctionsT< ::ClassC >::deleteArray_T;
  s_funcs.fDestructor  = ::Reflex::NewDelFunctionsT< ::ClassC >::destruct_T;
  if (retaddr) *(::Reflex::NewDelFunctions**)retaddr = &s_funcs;
}

static void method_x7( void* retaddr, void*, const std::vector<void*>&, void*)
{
  typedef std::vector<std::pair< ::Reflex::Base, int> > Bases_t;
  static Bases_t s_bases;
  if ( !s_bases.size() ) {
    s_bases.push_back(std::make_pair(::Reflex::Base( ::Reflex::TypeBuilder("ClassABase"), ::Reflex::BaseOffset< ::ClassC,::ClassABase >::Get(),::Reflex::PUBLIC), 0));
  }
  if (retaddr) *(Bases_t**)retaddr = &s_bases;
}

//------Dictionary for class ClassC -------------------------------
void __ClassC_db_datamem(Reflex::Class*);
void __ClassC_db_funcmem(Reflex::Class*);
Reflex::GenreflexMemberBuilder __ClassC_datamem_bld(&__ClassC_db_datamem);
Reflex::GenreflexMemberBuilder __ClassC_funcmem_bld(&__ClassC_db_funcmem);
void __ClassC_dict() {
  ::Reflex::ClassBuilder("ClassC", typeid(::ClassC), sizeof(::ClassC), ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::VIRTUAL, ::Reflex::CLASS)
  .AddProperty("ClassVersion", "3")
  .AddBase(type_776, ::Reflex::BaseOffset< ::ClassC, ::ClassABase >::Get(), ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3968, type_3970), "operator=", operator_2571, 0, "", ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_3970), "ClassC", constructor_2572, 0, "", ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "ClassC", constructor_2573, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "~ClassC", destructor_2574, 0, 0, ::Reflex::PUBLIC | ::Reflex::VIRTUAL | ::Reflex::DESTRUCTOR )
  .AddFunctionMember<void*(void)>("__getNewDelFunctions", method_newdel_996, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("__getBasesTable", method_x7, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddOnDemandDataMemberBuilder(&__ClassC_datamem_bld);
}

//------Delayed data member builder for class ClassC -------------------
void __ClassC_db_datamem(Reflex::Class* cl) {
  ::Reflex::ClassBuilder(cl)
  .AddDataMember(type_1818, "m_f", OffsetOf(__shadow__::__ClassC, m_f), ::Reflex::PRIVATE)
  .AddDataMember(type_1823, "m_g", OffsetOf(__shadow__::__ClassC, m_g), ::Reflex::PRIVATE);
}
//------Delayed function member builder for class ClassC -------------------
void __ClassC_db_funcmem(Reflex::Class*) {

}
//------Stub functions for class ClassAIns -------------------------------
static void destructor_2580(void*, void * o, const std::vector<void*>&, void *) {
  (((::ClassAIns*)o)->::ClassAIns::~ClassAIns)();
}
static void operator_2581( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::ClassAIns*)o)->operator=)(*(const ::ClassAIns*)arg[0]);
else   (((::ClassAIns*)o)->operator=)(*(const ::ClassAIns*)arg[0]);
}

static void constructor_2582( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::ClassAIns(*(const ::ClassAIns*)arg[0]);
  else ::new(mem) ::ClassAIns(*(const ::ClassAIns*)arg[0]);
}

static void constructor_2583( void* retaddr, void* mem, const std::vector<void*>&, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::ClassAIns();
  else ::new(mem) ::ClassAIns();
}

static void method_2584( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::ClassAIns*)o)->SetUnit)(*(int*)arg[0]);
}

static void method_newdel_1031( void* retaddr, void*, const std::vector<void*>&, void*)
{
  static ::Reflex::NewDelFunctions s_funcs;
  s_funcs.fNew         = ::Reflex::NewDelFunctionsT< ::ClassAIns >::new_T;
  s_funcs.fNewArray    = ::Reflex::NewDelFunctionsT< ::ClassAIns >::newArray_T;
  s_funcs.fDelete      = ::Reflex::NewDelFunctionsT< ::ClassAIns >::delete_T;
  s_funcs.fDeleteArray = ::Reflex::NewDelFunctionsT< ::ClassAIns >::deleteArray_T;
  s_funcs.fDestructor  = ::Reflex::NewDelFunctionsT< ::ClassAIns >::destruct_T;
  if (retaddr) *(::Reflex::NewDelFunctions**)retaddr = &s_funcs;
}

//------Dictionary for class ClassAIns -------------------------------
void __ClassAIns_db_datamem(Reflex::Class*);
void __ClassAIns_db_funcmem(Reflex::Class*);
Reflex::GenreflexMemberBuilder __ClassAIns_datamem_bld(&__ClassAIns_db_datamem);
Reflex::GenreflexMemberBuilder __ClassAIns_funcmem_bld(&__ClassAIns_db_funcmem);
void __ClassAIns_dict() {
  ::Reflex::ClassBuilder("ClassAIns", typeid(::ClassAIns), sizeof(::ClassAIns), ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL, ::Reflex::CLASS)
  .AddProperty("ClassVersion", "3")
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "~ClassAIns", destructor_2580, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::DESTRUCTOR )
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_7089, type_7090), "operator=", operator_2581, 0, "", ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7090), "ClassAIns", constructor_2582, 0, "", ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "ClassAIns", constructor_2583, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember<void*(void)>("__getNewDelFunctions", method_newdel_1031, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddOnDemandDataMemberBuilder(&__ClassAIns_datamem_bld)
  .AddOnDemandFunctionMemberBuilder(&__ClassAIns_funcmem_bld);
}

//------Delayed data member builder for class ClassAIns -------------------
void __ClassAIns_db_datamem(Reflex::Class* cl) {
  ::Reflex::ClassBuilder(cl)
  .AddDataMember(type_36, "m_a", OffsetOf(__shadow__::__ClassAIns, m_a), ::Reflex::PRIVATE)
  .AddDataMember(type_1818, "m_b", OffsetOf(__shadow__::__ClassAIns, m_b), ::Reflex::PRIVATE)
  .AddDataMember(type_36, "m_unit", OffsetOf(__shadow__::__ClassAIns, m_unit), ::Reflex::PRIVATE);
}
//------Delayed function member builder for class ClassAIns -------------------
void __ClassAIns_db_funcmem(Reflex::Class* cl) {
  ::Reflex::ClassBuilder(cl)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_36), "SetUnit", method_2584, 0, "unit", ::Reflex::PUBLIC);
}
//------Stub functions for class vector<ClassD*,std::allocator<ClassD*> > -------------------------------
static void constructor_3803( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 0 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassD*>();
  else ::new(mem) ::std::vector<ClassD*>();
  }
  else if ( arg.size() == 1 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassD*>(*(const ::std::allocator<ClassD*>*)arg[0]);
  else ::new(mem) ::std::vector<ClassD*>(*(const ::std::allocator<ClassD*>*)arg[0]);
  }
}

static void constructor_3804( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 2 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassD*>(*(::size_t*)arg[0],
      *(::ClassD* const*)arg[1]);
  else ::new(mem) ::std::vector<ClassD*>(*(::size_t*)arg[0],
      *(::ClassD* const*)arg[1]);
  }
  else if ( arg.size() == 3 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassD*>(*(::size_t*)arg[0],
      *(::ClassD* const*)arg[1],
      *(const ::std::allocator<ClassD*>*)arg[2]);
  else ::new(mem) ::std::vector<ClassD*>(*(::size_t*)arg[0],
      *(::ClassD* const*)arg[1],
      *(const ::std::allocator<ClassD*>*)arg[2]);
  }
}

static void constructor_3805( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassD*>(*(::size_t*)arg[0]);
  else ::new(mem) ::std::vector<ClassD*>(*(::size_t*)arg[0]);
}

static void constructor_3806( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassD*>(*(const ::std::vector<ClassD*>*)arg[0]);
  else ::new(mem) ::std::vector<ClassD*>(*(const ::std::vector<ClassD*>*)arg[0]);
}

static void destructor_3807(void*, void * o, const std::vector<void*>&, void *) {
  (((::std::vector<ClassD*>*)o)->::std::vector<ClassD*>::~vector)();
}
static void operator_3808( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassD*>*)o)->operator=)(*(const ::std::vector<ClassD*>*)arg[0]);
else   (((::std::vector<ClassD*>*)o)->operator=)(*(const ::std::vector<ClassD*>*)arg[0]);
}

static void method_3809( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassD*>*)o)->assign)(*(::size_t*)arg[0],
    *(::ClassD* const*)arg[1]);
}

static void method_3810( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassD**,std::vector<ClassD*> >)((((::std::vector<ClassD*>*)o)->begin)());
else   (((::std::vector<ClassD*>*)o)->begin)();
}

static void method_3811( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassD* const*,std::vector<ClassD*> >)((((const ::std::vector<ClassD*>*)o)->begin)());
else   (((const ::std::vector<ClassD*>*)o)->begin)();
}

static void method_3812( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassD**,std::vector<ClassD*> >)((((::std::vector<ClassD*>*)o)->end)());
else   (((::std::vector<ClassD*>*)o)->end)();
}

static void method_3813( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassD* const*,std::vector<ClassD*> >)((((const ::std::vector<ClassD*>*)o)->end)());
else   (((const ::std::vector<ClassD*>*)o)->end)();
}

static void method_3818( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassD*>*)o)->size)());
else   (((const ::std::vector<ClassD*>*)o)->size)();
}

static void method_3819( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassD*>*)o)->max_size)());
else   (((const ::std::vector<ClassD*>*)o)->max_size)();
}

static void method_3820( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassD*>*)o)->resize)(*(::size_t*)arg[0],
    *(::ClassD* const*)arg[1]);
}

static void method_3821( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassD*>*)o)->resize)(*(::size_t*)arg[0]);
}

static void method_3822( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassD*>*)o)->capacity)());
else   (((const ::std::vector<ClassD*>*)o)->capacity)();
}

static void method_3823( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (bool)((((const ::std::vector<ClassD*>*)o)->empty)());
else   (((const ::std::vector<ClassD*>*)o)->empty)();
}

static void method_3824( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassD*>*)o)->reserve)(*(::size_t*)arg[0]);
}

static void operator_3825( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassD*>*)o)->operator[])(*(::size_t*)arg[0]);
else   (((::std::vector<ClassD*>*)o)->operator[])(*(::size_t*)arg[0]);
}

static void operator_3826( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassD*>*)o)->operator[])(*(::size_t*)arg[0]);
else   (((const ::std::vector<ClassD*>*)o)->operator[])(*(::size_t*)arg[0]);
}

static void method_3828( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassD*>*)o)->at)(*(::size_t*)arg[0]);
else   (((::std::vector<ClassD*>*)o)->at)(*(::size_t*)arg[0]);
}

static void method_3829( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassD*>*)o)->at)(*(::size_t*)arg[0]);
else   (((const ::std::vector<ClassD*>*)o)->at)(*(::size_t*)arg[0]);
}

static void method_3830( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassD*>*)o)->front)();
else   (((::std::vector<ClassD*>*)o)->front)();
}

static void method_3831( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassD*>*)o)->front)();
else   (((const ::std::vector<ClassD*>*)o)->front)();
}

static void method_3832( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassD*>*)o)->back)();
else   (((::std::vector<ClassD*>*)o)->back)();
}

static void method_3833( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassD*>*)o)->back)();
else   (((const ::std::vector<ClassD*>*)o)->back)();
}

static void method_3834( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassD*>*)o)->push_back)(*(::ClassD* const*)arg[0]);
}

static void method_3835( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<ClassD*>*)o)->pop_back)();
}

static void method_3836( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassD**,std::vector<ClassD*> >)((((::std::vector<ClassD*>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassD**,std::vector<ClassD*> >*)arg[0],
    *(::ClassD* const*)arg[1]));
else   (((::std::vector<ClassD*>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassD**,std::vector<ClassD*> >*)arg[0],
    *(::ClassD* const*)arg[1]);
}

static void method_3837( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassD*>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassD**,std::vector<ClassD*> >*)arg[0],
    *(::size_t*)arg[1],
    *(::ClassD* const*)arg[2]);
}

static void method_3838( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassD**,std::vector<ClassD*> >)((((::std::vector<ClassD*>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassD**,std::vector<ClassD*> >*)arg[0]));
else   (((::std::vector<ClassD*>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassD**,std::vector<ClassD*> >*)arg[0]);
}

static void method_3839( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassD**,std::vector<ClassD*> >)((((::std::vector<ClassD*>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassD**,std::vector<ClassD*> >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<ClassD**,std::vector<ClassD*> >*)arg[1]));
else   (((::std::vector<ClassD*>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassD**,std::vector<ClassD*> >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<ClassD**,std::vector<ClassD*> >*)arg[1]);
}

static void method_3840( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassD*>*)o)->swap)(*(::std::vector<ClassD*>*)arg[0]);
}

static void method_3841( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<ClassD*>*)o)->clear)();
}

static void constructor_x9( void* retaddr, void* mem, const std::vector<void*>&, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassD*>();
  else ::new(mem) ::std::vector<ClassD*>();
}

static void method_newdel_1382( void* retaddr, void*, const std::vector<void*>&, void*)
{
  static ::Reflex::NewDelFunctions s_funcs;
  s_funcs.fNew         = ::Reflex::NewDelFunctionsT< ::std::vector<ClassD*> >::new_T;
  s_funcs.fNewArray    = ::Reflex::NewDelFunctionsT< ::std::vector<ClassD*> >::newArray_T;
  s_funcs.fDelete      = ::Reflex::NewDelFunctionsT< ::std::vector<ClassD*> >::delete_T;
  s_funcs.fDeleteArray = ::Reflex::NewDelFunctionsT< ::std::vector<ClassD*> >::deleteArray_T;
  s_funcs.fDestructor  = ::Reflex::NewDelFunctionsT< ::std::vector<ClassD*> >::destruct_T;
  if (retaddr) *(::Reflex::NewDelFunctions**)retaddr = &s_funcs;
}

static void method_x11( void* retaddr, void*, const std::vector<void*>&, void*)
{
  typedef std::vector<std::pair< ::Reflex::Base, int> > Bases_t;
  static Bases_t s_bases;
  if ( !s_bases.size() ) {
    s_bases.push_back(std::make_pair(::Reflex::Base( ::Reflex::TypeBuilder("std::_Vector_base<ClassD*,std::allocator<ClassD*> >"), ::Reflex::BaseOffset< ::std::vector<ClassD*>,::std::_Vector_base<ClassD*,std::allocator<ClassD*> > >::Get(),::Reflex::PROTECTED), 0));
  }
  if (retaddr) *(Bases_t**)retaddr = &s_bases;
}

static void method_x12( void* retaddr, void*, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**) retaddr = ::Reflex::Proxy< ::std::vector<ClassD*> >::Generate();
  else ::Reflex::Proxy< ::std::vector<ClassD*> >::Generate();
}

//------Dictionary for class vector<ClassD*,std::allocator<ClassD*> > -------------------------------
void __std__vector_ClassDp__db_datamem(Reflex::Class*);
void __std__vector_ClassDp__db_funcmem(Reflex::Class*);
Reflex::GenreflexMemberBuilder __std__vector_ClassDp__datamem_bld(&__std__vector_ClassDp__db_datamem);
Reflex::GenreflexMemberBuilder __std__vector_ClassDp__funcmem_bld(&__std__vector_ClassDp__db_funcmem);
void __std__vector_ClassDp__dict() {
  ::Reflex::ClassBuilder("std::vector<ClassD*>", typeid(::std::vector<ClassD*>), sizeof(::std::vector<ClassD*>), ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL, ::Reflex::CLASS)
  .AddBase(type_1638, ::Reflex::BaseOffset< ::std::vector<ClassD*>, ::std::_Vector_base<ClassD*,std::allocator<ClassD*> > >::Get(), ::Reflex::PROTECTED)
  .AddTypedef(type_1638, "std::vector<ClassD*>::_Base")
  .AddTypedef(type_3786, "std::vector<ClassD*>::value_type")
  .AddTypedef(type_3788, "std::vector<ClassD*>::pointer")
  .AddTypedef(type_3790, "std::vector<ClassD*>::const_pointer")
  .AddTypedef(type_3792, "std::vector<ClassD*>::reference")
  .AddTypedef(type_3794, "std::vector<ClassD*>::const_reference")
  .AddTypedef(type_2311, "std::vector<ClassD*>::iterator")
  .AddTypedef(type_2312, "std::vector<ClassD*>::const_iterator")
  .AddTypedef(type_1608, "std::vector<ClassD*>::const_reverse_iterator")
  .AddTypedef(type_1609, "std::vector<ClassD*>::reverse_iterator")
  .AddTypedef(type_854, "std::vector<ClassD*>::size_type")
  .AddTypedef(type_684, "std::vector<ClassD*>::difference_type")
  .AddTypedef(type_1532, "std::vector<ClassD*>::allocator_type")
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7257), "vector", constructor_3803, 0, "__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854, type_3794, type_7257), "vector", constructor_3804, 0, "__n;__value;__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854), "vector", constructor_3805, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7258), "vector", constructor_3806, 0, "__x", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "~vector", destructor_3807, 0, 0, ::Reflex::PUBLIC | ::Reflex::DESTRUCTOR )
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "vector", constructor_x9, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember<void*(void)>("__getNewDelFunctions", method_newdel_1382, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("__getBasesTable", method_x11, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("createCollFuncTable", method_x12, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddOnDemandFunctionMemberBuilder(&__std__vector_ClassDp__funcmem_bld);
}

//------Delayed data member builder for class vector<ClassD*,std::allocator<ClassD*> > -------------------
void __std__vector_ClassDp__db_datamem(Reflex::Class*) {

}
//------Delayed function member builder for class vector<ClassD*,std::allocator<ClassD*> > -------------------
void __std__vector_ClassDp__db_funcmem(Reflex::Class* cl) {
  ::Reflex::ClassBuilder(cl)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_7259, type_7258), "operator=", operator_3808, 0, "__x", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_3794), "assign", method_3809, 0, "__n;__val", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2311), "begin", method_3810, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2312), "begin", method_3811, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2311), "end", method_3812, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2312), "end", method_3813, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "size", method_3818, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "max_size", method_3819, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_3794), "resize", method_3820, 0, "__new_size;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "resize", method_3821, 0, "__new_size", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "capacity", method_3822, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2653), "empty", method_3823, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "reserve", method_3824, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3792, type_854), "operator[]", operator_3825, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3794, type_854), "operator[]", operator_3826, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3792, type_854), "at", method_3828, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3794, type_854), "at", method_3829, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3792), "front", method_3830, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3794), "front", method_3831, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3792), "back", method_3832, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3794), "back", method_3833, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_3794), "push_back", method_3834, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "pop_back", method_3835, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2311, type_2311, type_3794), "insert", method_3836, 0, "__position;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_2311, type_854, type_3794), "insert", method_3837, 0, "__position;__n;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2311, type_2311), "erase", method_3838, 0, "__position", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2311, type_2311, type_2311), "erase", method_3839, 0, "__first;__last", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_7259), "swap", method_3840, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "clear", method_3841, 0, 0, ::Reflex::PUBLIC);
}
//------Stub functions for class vector<ClassD,std::allocator<ClassD> > -------------------------------
static void constructor_3861( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 0 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassD>();
  else ::new(mem) ::std::vector<ClassD>();
  }
  else if ( arg.size() == 1 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassD>(*(const ::std::allocator<ClassD>*)arg[0]);
  else ::new(mem) ::std::vector<ClassD>(*(const ::std::allocator<ClassD>*)arg[0]);
  }
}

static void constructor_3862( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 2 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassD>(*(::size_t*)arg[0],
      *(const ::ClassD*)arg[1]);
  else ::new(mem) ::std::vector<ClassD>(*(::size_t*)arg[0],
      *(const ::ClassD*)arg[1]);
  }
  else if ( arg.size() == 3 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassD>(*(::size_t*)arg[0],
      *(const ::ClassD*)arg[1],
      *(const ::std::allocator<ClassD>*)arg[2]);
  else ::new(mem) ::std::vector<ClassD>(*(::size_t*)arg[0],
      *(const ::ClassD*)arg[1],
      *(const ::std::allocator<ClassD>*)arg[2]);
  }
}

static void constructor_3863( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassD>(*(::size_t*)arg[0]);
  else ::new(mem) ::std::vector<ClassD>(*(::size_t*)arg[0]);
}

static void constructor_3864( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassD>(*(const ::std::vector<ClassD>*)arg[0]);
  else ::new(mem) ::std::vector<ClassD>(*(const ::std::vector<ClassD>*)arg[0]);
}

static void destructor_3865(void*, void * o, const std::vector<void*>&, void *) {
  (((::std::vector<ClassD>*)o)->::std::vector<ClassD>::~vector)();
}
static void operator_3866( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassD>*)o)->operator=)(*(const ::std::vector<ClassD>*)arg[0]);
else   (((::std::vector<ClassD>*)o)->operator=)(*(const ::std::vector<ClassD>*)arg[0]);
}

static void method_3867( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassD>*)o)->assign)(*(::size_t*)arg[0],
    *(const ::ClassD*)arg[1]);
}

static void method_3868( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassD*,std::vector<ClassD> >)((((::std::vector<ClassD>*)o)->begin)());
else   (((::std::vector<ClassD>*)o)->begin)();
}

static void method_3869( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<const ClassD*,std::vector<ClassD> >)((((const ::std::vector<ClassD>*)o)->begin)());
else   (((const ::std::vector<ClassD>*)o)->begin)();
}

static void method_3870( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassD*,std::vector<ClassD> >)((((::std::vector<ClassD>*)o)->end)());
else   (((::std::vector<ClassD>*)o)->end)();
}

static void method_3871( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<const ClassD*,std::vector<ClassD> >)((((const ::std::vector<ClassD>*)o)->end)());
else   (((const ::std::vector<ClassD>*)o)->end)();
}

static void method_3876( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassD>*)o)->size)());
else   (((const ::std::vector<ClassD>*)o)->size)();
}

static void method_3877( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassD>*)o)->max_size)());
else   (((const ::std::vector<ClassD>*)o)->max_size)();
}

static void method_3878( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassD>*)o)->resize)(*(::size_t*)arg[0],
    *(const ::ClassD*)arg[1]);
}

static void method_3879( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassD>*)o)->resize)(*(::size_t*)arg[0]);
}

static void method_3880( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassD>*)o)->capacity)());
else   (((const ::std::vector<ClassD>*)o)->capacity)();
}

static void method_3881( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (bool)((((const ::std::vector<ClassD>*)o)->empty)());
else   (((const ::std::vector<ClassD>*)o)->empty)();
}

static void method_3882( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassD>*)o)->reserve)(*(::size_t*)arg[0]);
}

static void operator_3883( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassD>*)o)->operator[])(*(::size_t*)arg[0]);
else   (((::std::vector<ClassD>*)o)->operator[])(*(::size_t*)arg[0]);
}

static void operator_3884( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassD>*)o)->operator[])(*(::size_t*)arg[0]);
else   (((const ::std::vector<ClassD>*)o)->operator[])(*(::size_t*)arg[0]);
}

static void method_3886( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassD>*)o)->at)(*(::size_t*)arg[0]);
else   (((::std::vector<ClassD>*)o)->at)(*(::size_t*)arg[0]);
}

static void method_3887( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassD>*)o)->at)(*(::size_t*)arg[0]);
else   (((const ::std::vector<ClassD>*)o)->at)(*(::size_t*)arg[0]);
}

static void method_3888( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassD>*)o)->front)();
else   (((::std::vector<ClassD>*)o)->front)();
}

static void method_3889( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassD>*)o)->front)();
else   (((const ::std::vector<ClassD>*)o)->front)();
}

static void method_3890( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassD>*)o)->back)();
else   (((::std::vector<ClassD>*)o)->back)();
}

static void method_3891( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassD>*)o)->back)();
else   (((const ::std::vector<ClassD>*)o)->back)();
}

static void method_3892( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassD>*)o)->push_back)(*(const ::ClassD*)arg[0]);
}

static void method_3893( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<ClassD>*)o)->pop_back)();
}

static void method_3894( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassD*,std::vector<ClassD> >)((((::std::vector<ClassD>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassD*,std::vector<ClassD> >*)arg[0],
    *(const ::ClassD*)arg[1]));
else   (((::std::vector<ClassD>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassD*,std::vector<ClassD> >*)arg[0],
    *(const ::ClassD*)arg[1]);
}

static void method_3895( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassD>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassD*,std::vector<ClassD> >*)arg[0],
    *(::size_t*)arg[1],
    *(const ::ClassD*)arg[2]);
}

static void method_3896( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassD*,std::vector<ClassD> >)((((::std::vector<ClassD>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassD*,std::vector<ClassD> >*)arg[0]));
else   (((::std::vector<ClassD>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassD*,std::vector<ClassD> >*)arg[0]);
}

static void method_3897( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassD*,std::vector<ClassD> >)((((::std::vector<ClassD>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassD*,std::vector<ClassD> >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<ClassD*,std::vector<ClassD> >*)arg[1]));
else   (((::std::vector<ClassD>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassD*,std::vector<ClassD> >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<ClassD*,std::vector<ClassD> >*)arg[1]);
}

static void method_3898( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassD>*)o)->swap)(*(::std::vector<ClassD>*)arg[0]);
}

static void method_3899( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<ClassD>*)o)->clear)();
}

static void constructor_x13( void* retaddr, void* mem, const std::vector<void*>&, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassD>();
  else ::new(mem) ::std::vector<ClassD>();
}

static void method_newdel_1383( void* retaddr, void*, const std::vector<void*>&, void*)
{
  static ::Reflex::NewDelFunctions s_funcs;
  s_funcs.fNew         = ::Reflex::NewDelFunctionsT< ::std::vector<ClassD> >::new_T;
  s_funcs.fNewArray    = ::Reflex::NewDelFunctionsT< ::std::vector<ClassD> >::newArray_T;
  s_funcs.fDelete      = ::Reflex::NewDelFunctionsT< ::std::vector<ClassD> >::delete_T;
  s_funcs.fDeleteArray = ::Reflex::NewDelFunctionsT< ::std::vector<ClassD> >::deleteArray_T;
  s_funcs.fDestructor  = ::Reflex::NewDelFunctionsT< ::std::vector<ClassD> >::destruct_T;
  if (retaddr) *(::Reflex::NewDelFunctions**)retaddr = &s_funcs;
}

static void method_x15( void* retaddr, void*, const std::vector<void*>&, void*)
{
  typedef std::vector<std::pair< ::Reflex::Base, int> > Bases_t;
  static Bases_t s_bases;
  if ( !s_bases.size() ) {
    s_bases.push_back(std::make_pair(::Reflex::Base( ::Reflex::TypeBuilder("std::_Vector_base<ClassD,std::allocator<ClassD> >"), ::Reflex::BaseOffset< ::std::vector<ClassD>,::std::_Vector_base<ClassD,std::allocator<ClassD> > >::Get(),::Reflex::PROTECTED), 0));
  }
  if (retaddr) *(Bases_t**)retaddr = &s_bases;
}

static void method_x16( void* retaddr, void*, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**) retaddr = ::Reflex::Proxy< ::std::vector<ClassD> >::Generate();
  else ::Reflex::Proxy< ::std::vector<ClassD> >::Generate();
}

//------Dictionary for class vector<ClassD,std::allocator<ClassD> > -------------------------------
void __std__vector_ClassD__db_datamem(Reflex::Class*);
void __std__vector_ClassD__db_funcmem(Reflex::Class*);
Reflex::GenreflexMemberBuilder __std__vector_ClassD__datamem_bld(&__std__vector_ClassD__db_datamem);
Reflex::GenreflexMemberBuilder __std__vector_ClassD__funcmem_bld(&__std__vector_ClassD__db_funcmem);
void __std__vector_ClassD__dict() {
  ::Reflex::ClassBuilder("std::vector<ClassD>", typeid(::std::vector<ClassD>), sizeof(::std::vector<ClassD>), ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL, ::Reflex::CLASS)
  .AddBase(type_1639, ::Reflex::BaseOffset< ::std::vector<ClassD>, ::std::_Vector_base<ClassD,std::allocator<ClassD> > >::Get(), ::Reflex::PROTECTED)
  .AddTypedef(type_1639, "std::vector<ClassD>::_Base")
  .AddTypedef(type_29, "std::vector<ClassD>::value_type")
  .AddTypedef(type_3786, "std::vector<ClassD>::pointer")
  .AddTypedef(type_3848, "std::vector<ClassD>::const_pointer")
  .AddTypedef(type_3850, "std::vector<ClassD>::reference")
  .AddTypedef(type_3852, "std::vector<ClassD>::const_reference")
  .AddTypedef(type_2313, "std::vector<ClassD>::iterator")
  .AddTypedef(type_2314, "std::vector<ClassD>::const_iterator")
  .AddTypedef(type_1610, "std::vector<ClassD>::const_reverse_iterator")
  .AddTypedef(type_1611, "std::vector<ClassD>::reverse_iterator")
  .AddTypedef(type_854, "std::vector<ClassD>::size_type")
  .AddTypedef(type_684, "std::vector<ClassD>::difference_type")
  .AddTypedef(type_1533, "std::vector<ClassD>::allocator_type")
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7261), "vector", constructor_3861, 0, "__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854, type_3852, type_7261), "vector", constructor_3862, 0, "__n;__value;__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854), "vector", constructor_3863, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7262), "vector", constructor_3864, 0, "__x", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "~vector", destructor_3865, 0, 0, ::Reflex::PUBLIC | ::Reflex::DESTRUCTOR )
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "vector", constructor_x13, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember<void*(void)>("__getNewDelFunctions", method_newdel_1383, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("__getBasesTable", method_x15, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("createCollFuncTable", method_x16, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddOnDemandFunctionMemberBuilder(&__std__vector_ClassD__funcmem_bld);
}

//------Delayed data member builder for class vector<ClassD,std::allocator<ClassD> > -------------------
void __std__vector_ClassD__db_datamem(Reflex::Class*) {

}
//------Delayed function member builder for class vector<ClassD,std::allocator<ClassD> > -------------------
void __std__vector_ClassD__db_funcmem(Reflex::Class* cl) {
  ::Reflex::ClassBuilder(cl)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_7263, type_7262), "operator=", operator_3866, 0, "__x", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_3852), "assign", method_3867, 0, "__n;__val", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2313), "begin", method_3868, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2314), "begin", method_3869, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2313), "end", method_3870, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2314), "end", method_3871, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "size", method_3876, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "max_size", method_3877, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_3852), "resize", method_3878, 0, "__new_size;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "resize", method_3879, 0, "__new_size", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "capacity", method_3880, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2653), "empty", method_3881, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "reserve", method_3882, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3850, type_854), "operator[]", operator_3883, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3852, type_854), "operator[]", operator_3884, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3850, type_854), "at", method_3886, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3852, type_854), "at", method_3887, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3850), "front", method_3888, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3852), "front", method_3889, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3850), "back", method_3890, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3852), "back", method_3891, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_3852), "push_back", method_3892, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "pop_back", method_3893, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2313, type_2313, type_3852), "insert", method_3894, 0, "__position;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_2313, type_854, type_3852), "insert", method_3895, 0, "__position;__n;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2313, type_2313), "erase", method_3896, 0, "__position", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2313, type_2313, type_2313), "erase", method_3897, 0, "__first;__last", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_7263), "swap", method_3898, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "clear", method_3899, 0, 0, ::Reflex::PUBLIC);
}
//------Stub functions for class vector<ClassC*,std::allocator<ClassC*> > -------------------------------
static void constructor_3921( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 0 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassC*>();
  else ::new(mem) ::std::vector<ClassC*>();
  }
  else if ( arg.size() == 1 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassC*>(*(const ::std::allocator<ClassC*>*)arg[0]);
  else ::new(mem) ::std::vector<ClassC*>(*(const ::std::allocator<ClassC*>*)arg[0]);
  }
}

static void constructor_3922( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 2 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassC*>(*(::size_t*)arg[0],
      *(::ClassC* const*)arg[1]);
  else ::new(mem) ::std::vector<ClassC*>(*(::size_t*)arg[0],
      *(::ClassC* const*)arg[1]);
  }
  else if ( arg.size() == 3 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassC*>(*(::size_t*)arg[0],
      *(::ClassC* const*)arg[1],
      *(const ::std::allocator<ClassC*>*)arg[2]);
  else ::new(mem) ::std::vector<ClassC*>(*(::size_t*)arg[0],
      *(::ClassC* const*)arg[1],
      *(const ::std::allocator<ClassC*>*)arg[2]);
  }
}

static void constructor_3923( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassC*>(*(::size_t*)arg[0]);
  else ::new(mem) ::std::vector<ClassC*>(*(::size_t*)arg[0]);
}

static void constructor_3924( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassC*>(*(const ::std::vector<ClassC*>*)arg[0]);
  else ::new(mem) ::std::vector<ClassC*>(*(const ::std::vector<ClassC*>*)arg[0]);
}

static void destructor_3925(void*, void * o, const std::vector<void*>&, void *) {
  (((::std::vector<ClassC*>*)o)->::std::vector<ClassC*>::~vector)();
}
static void operator_3926( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassC*>*)o)->operator=)(*(const ::std::vector<ClassC*>*)arg[0]);
else   (((::std::vector<ClassC*>*)o)->operator=)(*(const ::std::vector<ClassC*>*)arg[0]);
}

static void method_3927( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassC*>*)o)->assign)(*(::size_t*)arg[0],
    *(::ClassC* const*)arg[1]);
}

static void method_3928( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassC**,std::vector<ClassC*> >)((((::std::vector<ClassC*>*)o)->begin)());
else   (((::std::vector<ClassC*>*)o)->begin)();
}

static void method_3929( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassC* const*,std::vector<ClassC*> >)((((const ::std::vector<ClassC*>*)o)->begin)());
else   (((const ::std::vector<ClassC*>*)o)->begin)();
}

static void method_3930( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassC**,std::vector<ClassC*> >)((((::std::vector<ClassC*>*)o)->end)());
else   (((::std::vector<ClassC*>*)o)->end)();
}

static void method_3931( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassC* const*,std::vector<ClassC*> >)((((const ::std::vector<ClassC*>*)o)->end)());
else   (((const ::std::vector<ClassC*>*)o)->end)();
}

static void method_3936( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassC*>*)o)->size)());
else   (((const ::std::vector<ClassC*>*)o)->size)();
}

static void method_3937( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassC*>*)o)->max_size)());
else   (((const ::std::vector<ClassC*>*)o)->max_size)();
}

static void method_3938( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassC*>*)o)->resize)(*(::size_t*)arg[0],
    *(::ClassC* const*)arg[1]);
}

static void method_3939( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassC*>*)o)->resize)(*(::size_t*)arg[0]);
}

static void method_3940( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassC*>*)o)->capacity)());
else   (((const ::std::vector<ClassC*>*)o)->capacity)();
}

static void method_3941( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (bool)((((const ::std::vector<ClassC*>*)o)->empty)());
else   (((const ::std::vector<ClassC*>*)o)->empty)();
}

static void method_3942( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassC*>*)o)->reserve)(*(::size_t*)arg[0]);
}

static void operator_3943( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassC*>*)o)->operator[])(*(::size_t*)arg[0]);
else   (((::std::vector<ClassC*>*)o)->operator[])(*(::size_t*)arg[0]);
}

static void operator_3944( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassC*>*)o)->operator[])(*(::size_t*)arg[0]);
else   (((const ::std::vector<ClassC*>*)o)->operator[])(*(::size_t*)arg[0]);
}

static void method_3946( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassC*>*)o)->at)(*(::size_t*)arg[0]);
else   (((::std::vector<ClassC*>*)o)->at)(*(::size_t*)arg[0]);
}

static void method_3947( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassC*>*)o)->at)(*(::size_t*)arg[0]);
else   (((const ::std::vector<ClassC*>*)o)->at)(*(::size_t*)arg[0]);
}

static void method_3948( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassC*>*)o)->front)();
else   (((::std::vector<ClassC*>*)o)->front)();
}

static void method_3949( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassC*>*)o)->front)();
else   (((const ::std::vector<ClassC*>*)o)->front)();
}

static void method_3950( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassC*>*)o)->back)();
else   (((::std::vector<ClassC*>*)o)->back)();
}

static void method_3951( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassC*>*)o)->back)();
else   (((const ::std::vector<ClassC*>*)o)->back)();
}

static void method_3952( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassC*>*)o)->push_back)(*(::ClassC* const*)arg[0]);
}

static void method_3953( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<ClassC*>*)o)->pop_back)();
}

static void method_3954( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassC**,std::vector<ClassC*> >)((((::std::vector<ClassC*>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassC**,std::vector<ClassC*> >*)arg[0],
    *(::ClassC* const*)arg[1]));
else   (((::std::vector<ClassC*>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassC**,std::vector<ClassC*> >*)arg[0],
    *(::ClassC* const*)arg[1]);
}

static void method_3955( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassC*>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassC**,std::vector<ClassC*> >*)arg[0],
    *(::size_t*)arg[1],
    *(::ClassC* const*)arg[2]);
}

static void method_3956( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassC**,std::vector<ClassC*> >)((((::std::vector<ClassC*>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassC**,std::vector<ClassC*> >*)arg[0]));
else   (((::std::vector<ClassC*>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassC**,std::vector<ClassC*> >*)arg[0]);
}

static void method_3957( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassC**,std::vector<ClassC*> >)((((::std::vector<ClassC*>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassC**,std::vector<ClassC*> >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<ClassC**,std::vector<ClassC*> >*)arg[1]));
else   (((::std::vector<ClassC*>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassC**,std::vector<ClassC*> >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<ClassC**,std::vector<ClassC*> >*)arg[1]);
}

static void method_3958( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassC*>*)o)->swap)(*(::std::vector<ClassC*>*)arg[0]);
}

static void method_3959( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<ClassC*>*)o)->clear)();
}

static void constructor_x17( void* retaddr, void* mem, const std::vector<void*>&, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassC*>();
  else ::new(mem) ::std::vector<ClassC*>();
}

static void method_newdel_1384( void* retaddr, void*, const std::vector<void*>&, void*)
{
  static ::Reflex::NewDelFunctions s_funcs;
  s_funcs.fNew         = ::Reflex::NewDelFunctionsT< ::std::vector<ClassC*> >::new_T;
  s_funcs.fNewArray    = ::Reflex::NewDelFunctionsT< ::std::vector<ClassC*> >::newArray_T;
  s_funcs.fDelete      = ::Reflex::NewDelFunctionsT< ::std::vector<ClassC*> >::delete_T;
  s_funcs.fDeleteArray = ::Reflex::NewDelFunctionsT< ::std::vector<ClassC*> >::deleteArray_T;
  s_funcs.fDestructor  = ::Reflex::NewDelFunctionsT< ::std::vector<ClassC*> >::destruct_T;
  if (retaddr) *(::Reflex::NewDelFunctions**)retaddr = &s_funcs;
}

static void method_x19( void* retaddr, void*, const std::vector<void*>&, void*)
{
  typedef std::vector<std::pair< ::Reflex::Base, int> > Bases_t;
  static Bases_t s_bases;
  if ( !s_bases.size() ) {
    s_bases.push_back(std::make_pair(::Reflex::Base( ::Reflex::TypeBuilder("std::_Vector_base<ClassC*,std::allocator<ClassC*> >"), ::Reflex::BaseOffset< ::std::vector<ClassC*>,::std::_Vector_base<ClassC*,std::allocator<ClassC*> > >::Get(),::Reflex::PROTECTED), 0));
  }
  if (retaddr) *(Bases_t**)retaddr = &s_bases;
}

static void method_x20( void* retaddr, void*, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**) retaddr = ::Reflex::Proxy< ::std::vector<ClassC*> >::Generate();
  else ::Reflex::Proxy< ::std::vector<ClassC*> >::Generate();
}

//------Dictionary for class vector<ClassC*,std::allocator<ClassC*> > -------------------------------
void __std__vector_ClassCp__db_datamem(Reflex::Class*);
void __std__vector_ClassCp__db_funcmem(Reflex::Class*);
Reflex::GenreflexMemberBuilder __std__vector_ClassCp__datamem_bld(&__std__vector_ClassCp__db_datamem);
Reflex::GenreflexMemberBuilder __std__vector_ClassCp__funcmem_bld(&__std__vector_ClassCp__db_funcmem);
void __std__vector_ClassCp__dict() {
  ::Reflex::ClassBuilder("std::vector<ClassC*>", typeid(::std::vector<ClassC*>), sizeof(::std::vector<ClassC*>), ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL, ::Reflex::CLASS)
  .AddBase(type_1640, ::Reflex::BaseOffset< ::std::vector<ClassC*>, ::std::_Vector_base<ClassC*,std::allocator<ClassC*> > >::Get(), ::Reflex::PROTECTED)
  .AddTypedef(type_1640, "std::vector<ClassC*>::_Base")
  .AddTypedef(type_3904, "std::vector<ClassC*>::value_type")
  .AddTypedef(type_3906, "std::vector<ClassC*>::pointer")
  .AddTypedef(type_3908, "std::vector<ClassC*>::const_pointer")
  .AddTypedef(type_3910, "std::vector<ClassC*>::reference")
  .AddTypedef(type_3912, "std::vector<ClassC*>::const_reference")
  .AddTypedef(type_2315, "std::vector<ClassC*>::iterator")
  .AddTypedef(type_2316, "std::vector<ClassC*>::const_iterator")
  .AddTypedef(type_1612, "std::vector<ClassC*>::const_reverse_iterator")
  .AddTypedef(type_1613, "std::vector<ClassC*>::reverse_iterator")
  .AddTypedef(type_854, "std::vector<ClassC*>::size_type")
  .AddTypedef(type_684, "std::vector<ClassC*>::difference_type")
  .AddTypedef(type_1534, "std::vector<ClassC*>::allocator_type")
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7265), "vector", constructor_3921, 0, "__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854, type_3912, type_7265), "vector", constructor_3922, 0, "__n;__value;__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854), "vector", constructor_3923, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7266), "vector", constructor_3924, 0, "__x", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "~vector", destructor_3925, 0, 0, ::Reflex::PUBLIC | ::Reflex::DESTRUCTOR )
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "vector", constructor_x17, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember<void*(void)>("__getNewDelFunctions", method_newdel_1384, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("__getBasesTable", method_x19, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("createCollFuncTable", method_x20, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddOnDemandFunctionMemberBuilder(&__std__vector_ClassCp__funcmem_bld);
}

//------Delayed data member builder for class vector<ClassC*,std::allocator<ClassC*> > -------------------
void __std__vector_ClassCp__db_datamem(Reflex::Class*) {

}
//------Delayed function member builder for class vector<ClassC*,std::allocator<ClassC*> > -------------------
void __std__vector_ClassCp__db_funcmem(Reflex::Class* cl) {
  ::Reflex::ClassBuilder(cl)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_7267, type_7266), "operator=", operator_3926, 0, "__x", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_3912), "assign", method_3927, 0, "__n;__val", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2315), "begin", method_3928, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2316), "begin", method_3929, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2315), "end", method_3930, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2316), "end", method_3931, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "size", method_3936, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "max_size", method_3937, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_3912), "resize", method_3938, 0, "__new_size;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "resize", method_3939, 0, "__new_size", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "capacity", method_3940, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2653), "empty", method_3941, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "reserve", method_3942, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3910, type_854), "operator[]", operator_3943, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3912, type_854), "operator[]", operator_3944, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3910, type_854), "at", method_3946, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3912, type_854), "at", method_3947, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3910), "front", method_3948, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3912), "front", method_3949, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3910), "back", method_3950, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3912), "back", method_3951, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_3912), "push_back", method_3952, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "pop_back", method_3953, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2315, type_2315, type_3912), "insert", method_3954, 0, "__position;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_2315, type_854, type_3912), "insert", method_3955, 0, "__position;__n;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2315, type_2315), "erase", method_3956, 0, "__position", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2315, type_2315, type_2315), "erase", method_3957, 0, "__first;__last", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_7267), "swap", method_3958, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "clear", method_3959, 0, 0, ::Reflex::PUBLIC);
}
//------Stub functions for class vector<ClassC,std::allocator<ClassC> > -------------------------------
static void constructor_3979( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 0 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassC>();
  else ::new(mem) ::std::vector<ClassC>();
  }
  else if ( arg.size() == 1 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassC>(*(const ::std::allocator<ClassC>*)arg[0]);
  else ::new(mem) ::std::vector<ClassC>(*(const ::std::allocator<ClassC>*)arg[0]);
  }
}

static void constructor_3980( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 2 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassC>(*(::size_t*)arg[0],
      *(const ::ClassC*)arg[1]);
  else ::new(mem) ::std::vector<ClassC>(*(::size_t*)arg[0],
      *(const ::ClassC*)arg[1]);
  }
  else if ( arg.size() == 3 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassC>(*(::size_t*)arg[0],
      *(const ::ClassC*)arg[1],
      *(const ::std::allocator<ClassC>*)arg[2]);
  else ::new(mem) ::std::vector<ClassC>(*(::size_t*)arg[0],
      *(const ::ClassC*)arg[1],
      *(const ::std::allocator<ClassC>*)arg[2]);
  }
}

static void constructor_3981( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassC>(*(::size_t*)arg[0]);
  else ::new(mem) ::std::vector<ClassC>(*(::size_t*)arg[0]);
}

static void constructor_3982( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassC>(*(const ::std::vector<ClassC>*)arg[0]);
  else ::new(mem) ::std::vector<ClassC>(*(const ::std::vector<ClassC>*)arg[0]);
}

static void destructor_3983(void*, void * o, const std::vector<void*>&, void *) {
  (((::std::vector<ClassC>*)o)->::std::vector<ClassC>::~vector)();
}
static void operator_3984( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassC>*)o)->operator=)(*(const ::std::vector<ClassC>*)arg[0]);
else   (((::std::vector<ClassC>*)o)->operator=)(*(const ::std::vector<ClassC>*)arg[0]);
}

static void method_3985( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassC>*)o)->assign)(*(::size_t*)arg[0],
    *(const ::ClassC*)arg[1]);
}

static void method_3986( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassC*,std::vector<ClassC> >)((((::std::vector<ClassC>*)o)->begin)());
else   (((::std::vector<ClassC>*)o)->begin)();
}

static void method_3987( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<const ClassC*,std::vector<ClassC> >)((((const ::std::vector<ClassC>*)o)->begin)());
else   (((const ::std::vector<ClassC>*)o)->begin)();
}

static void method_3988( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassC*,std::vector<ClassC> >)((((::std::vector<ClassC>*)o)->end)());
else   (((::std::vector<ClassC>*)o)->end)();
}

static void method_3989( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<const ClassC*,std::vector<ClassC> >)((((const ::std::vector<ClassC>*)o)->end)());
else   (((const ::std::vector<ClassC>*)o)->end)();
}

static void method_3994( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassC>*)o)->size)());
else   (((const ::std::vector<ClassC>*)o)->size)();
}

static void method_3995( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassC>*)o)->max_size)());
else   (((const ::std::vector<ClassC>*)o)->max_size)();
}

static void method_3996( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassC>*)o)->resize)(*(::size_t*)arg[0],
    *(const ::ClassC*)arg[1]);
}

static void method_3997( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassC>*)o)->resize)(*(::size_t*)arg[0]);
}

static void method_3998( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassC>*)o)->capacity)());
else   (((const ::std::vector<ClassC>*)o)->capacity)();
}

static void method_3999( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (bool)((((const ::std::vector<ClassC>*)o)->empty)());
else   (((const ::std::vector<ClassC>*)o)->empty)();
}

static void method_4000( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassC>*)o)->reserve)(*(::size_t*)arg[0]);
}

static void operator_4001( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassC>*)o)->operator[])(*(::size_t*)arg[0]);
else   (((::std::vector<ClassC>*)o)->operator[])(*(::size_t*)arg[0]);
}

static void operator_4002( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassC>*)o)->operator[])(*(::size_t*)arg[0]);
else   (((const ::std::vector<ClassC>*)o)->operator[])(*(::size_t*)arg[0]);
}

static void method_4004( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassC>*)o)->at)(*(::size_t*)arg[0]);
else   (((::std::vector<ClassC>*)o)->at)(*(::size_t*)arg[0]);
}

static void method_4005( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassC>*)o)->at)(*(::size_t*)arg[0]);
else   (((const ::std::vector<ClassC>*)o)->at)(*(::size_t*)arg[0]);
}

static void method_4006( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassC>*)o)->front)();
else   (((::std::vector<ClassC>*)o)->front)();
}

static void method_4007( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassC>*)o)->front)();
else   (((const ::std::vector<ClassC>*)o)->front)();
}

static void method_4008( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassC>*)o)->back)();
else   (((::std::vector<ClassC>*)o)->back)();
}

static void method_4009( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassC>*)o)->back)();
else   (((const ::std::vector<ClassC>*)o)->back)();
}

static void method_4010( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassC>*)o)->push_back)(*(const ::ClassC*)arg[0]);
}

static void method_4011( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<ClassC>*)o)->pop_back)();
}

static void method_4012( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassC*,std::vector<ClassC> >)((((::std::vector<ClassC>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassC*,std::vector<ClassC> >*)arg[0],
    *(const ::ClassC*)arg[1]));
else   (((::std::vector<ClassC>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassC*,std::vector<ClassC> >*)arg[0],
    *(const ::ClassC*)arg[1]);
}

static void method_4013( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassC>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassC*,std::vector<ClassC> >*)arg[0],
    *(::size_t*)arg[1],
    *(const ::ClassC*)arg[2]);
}

static void method_4014( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassC*,std::vector<ClassC> >)((((::std::vector<ClassC>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassC*,std::vector<ClassC> >*)arg[0]));
else   (((::std::vector<ClassC>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassC*,std::vector<ClassC> >*)arg[0]);
}

static void method_4015( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassC*,std::vector<ClassC> >)((((::std::vector<ClassC>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassC*,std::vector<ClassC> >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<ClassC*,std::vector<ClassC> >*)arg[1]));
else   (((::std::vector<ClassC>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassC*,std::vector<ClassC> >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<ClassC*,std::vector<ClassC> >*)arg[1]);
}

static void method_4016( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassC>*)o)->swap)(*(::std::vector<ClassC>*)arg[0]);
}

static void method_4017( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<ClassC>*)o)->clear)();
}

static void constructor_x21( void* retaddr, void* mem, const std::vector<void*>&, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassC>();
  else ::new(mem) ::std::vector<ClassC>();
}

static void method_newdel_1385( void* retaddr, void*, const std::vector<void*>&, void*)
{
  static ::Reflex::NewDelFunctions s_funcs;
  s_funcs.fNew         = ::Reflex::NewDelFunctionsT< ::std::vector<ClassC> >::new_T;
  s_funcs.fNewArray    = ::Reflex::NewDelFunctionsT< ::std::vector<ClassC> >::newArray_T;
  s_funcs.fDelete      = ::Reflex::NewDelFunctionsT< ::std::vector<ClassC> >::delete_T;
  s_funcs.fDeleteArray = ::Reflex::NewDelFunctionsT< ::std::vector<ClassC> >::deleteArray_T;
  s_funcs.fDestructor  = ::Reflex::NewDelFunctionsT< ::std::vector<ClassC> >::destruct_T;
  if (retaddr) *(::Reflex::NewDelFunctions**)retaddr = &s_funcs;
}

static void method_x23( void* retaddr, void*, const std::vector<void*>&, void*)
{
  typedef std::vector<std::pair< ::Reflex::Base, int> > Bases_t;
  static Bases_t s_bases;
  if ( !s_bases.size() ) {
    s_bases.push_back(std::make_pair(::Reflex::Base( ::Reflex::TypeBuilder("std::_Vector_base<ClassC,std::allocator<ClassC> >"), ::Reflex::BaseOffset< ::std::vector<ClassC>,::std::_Vector_base<ClassC,std::allocator<ClassC> > >::Get(),::Reflex::PROTECTED), 0));
  }
  if (retaddr) *(Bases_t**)retaddr = &s_bases;
}

static void method_x24( void* retaddr, void*, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**) retaddr = ::Reflex::Proxy< ::std::vector<ClassC> >::Generate();
  else ::Reflex::Proxy< ::std::vector<ClassC> >::Generate();
}

//------Dictionary for class vector<ClassC,std::allocator<ClassC> > -------------------------------
void __std__vector_ClassC__db_datamem(Reflex::Class*);
void __std__vector_ClassC__db_funcmem(Reflex::Class*);
Reflex::GenreflexMemberBuilder __std__vector_ClassC__datamem_bld(&__std__vector_ClassC__db_datamem);
Reflex::GenreflexMemberBuilder __std__vector_ClassC__funcmem_bld(&__std__vector_ClassC__db_funcmem);
void __std__vector_ClassC__dict() {
  ::Reflex::ClassBuilder("std::vector<ClassC>", typeid(::std::vector<ClassC>), sizeof(::std::vector<ClassC>), ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL, ::Reflex::CLASS)
  .AddBase(type_1641, ::Reflex::BaseOffset< ::std::vector<ClassC>, ::std::_Vector_base<ClassC,std::allocator<ClassC> > >::Get(), ::Reflex::PROTECTED)
  .AddTypedef(type_1641, "std::vector<ClassC>::_Base")
  .AddTypedef(type_996, "std::vector<ClassC>::value_type")
  .AddTypedef(type_3904, "std::vector<ClassC>::pointer")
  .AddTypedef(type_3966, "std::vector<ClassC>::const_pointer")
  .AddTypedef(type_3968, "std::vector<ClassC>::reference")
  .AddTypedef(type_3970, "std::vector<ClassC>::const_reference")
  .AddTypedef(type_2317, "std::vector<ClassC>::iterator")
  .AddTypedef(type_2318, "std::vector<ClassC>::const_iterator")
  .AddTypedef(type_1614, "std::vector<ClassC>::const_reverse_iterator")
  .AddTypedef(type_1615, "std::vector<ClassC>::reverse_iterator")
  .AddTypedef(type_854, "std::vector<ClassC>::size_type")
  .AddTypedef(type_684, "std::vector<ClassC>::difference_type")
  .AddTypedef(type_1535, "std::vector<ClassC>::allocator_type")
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7269), "vector", constructor_3979, 0, "__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854, type_3970, type_7269), "vector", constructor_3980, 0, "__n;__value;__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854), "vector", constructor_3981, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7270), "vector", constructor_3982, 0, "__x", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "~vector", destructor_3983, 0, 0, ::Reflex::PUBLIC | ::Reflex::DESTRUCTOR )
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "vector", constructor_x21, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember<void*(void)>("__getNewDelFunctions", method_newdel_1385, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("__getBasesTable", method_x23, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("createCollFuncTable", method_x24, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddOnDemandFunctionMemberBuilder(&__std__vector_ClassC__funcmem_bld);
}

//------Delayed data member builder for class vector<ClassC,std::allocator<ClassC> > -------------------
void __std__vector_ClassC__db_datamem(Reflex::Class*) {

}
//------Delayed function member builder for class vector<ClassC,std::allocator<ClassC> > -------------------
void __std__vector_ClassC__db_funcmem(Reflex::Class* cl) {
  ::Reflex::ClassBuilder(cl)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_7271, type_7270), "operator=", operator_3984, 0, "__x", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_3970), "assign", method_3985, 0, "__n;__val", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2317), "begin", method_3986, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2318), "begin", method_3987, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2317), "end", method_3988, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2318), "end", method_3989, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "size", method_3994, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "max_size", method_3995, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_3970), "resize", method_3996, 0, "__new_size;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "resize", method_3997, 0, "__new_size", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "capacity", method_3998, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2653), "empty", method_3999, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "reserve", method_4000, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3968, type_854), "operator[]", operator_4001, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3970, type_854), "operator[]", operator_4002, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3968, type_854), "at", method_4004, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3970, type_854), "at", method_4005, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3968), "front", method_4006, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3970), "front", method_4007, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3968), "back", method_4008, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_3970), "back", method_4009, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_3970), "push_back", method_4010, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "pop_back", method_4011, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2317, type_2317, type_3970), "insert", method_4012, 0, "__position;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_2317, type_854, type_3970), "insert", method_4013, 0, "__position;__n;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2317, type_2317), "erase", method_4014, 0, "__position", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2317, type_2317, type_2317), "erase", method_4015, 0, "__first;__last", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_7271), "swap", method_4016, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "clear", method_4017, 0, 0, ::Reflex::PUBLIC);
}
//------Stub functions for class vector<ClassB*,std::allocator<ClassB*> > -------------------------------
static void constructor_4039( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 0 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassB*>();
  else ::new(mem) ::std::vector<ClassB*>();
  }
  else if ( arg.size() == 1 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassB*>(*(const ::std::allocator<ClassB*>*)arg[0]);
  else ::new(mem) ::std::vector<ClassB*>(*(const ::std::allocator<ClassB*>*)arg[0]);
  }
}

static void constructor_4040( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 2 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassB*>(*(::size_t*)arg[0],
      *(::ClassB* const*)arg[1]);
  else ::new(mem) ::std::vector<ClassB*>(*(::size_t*)arg[0],
      *(::ClassB* const*)arg[1]);
  }
  else if ( arg.size() == 3 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassB*>(*(::size_t*)arg[0],
      *(::ClassB* const*)arg[1],
      *(const ::std::allocator<ClassB*>*)arg[2]);
  else ::new(mem) ::std::vector<ClassB*>(*(::size_t*)arg[0],
      *(::ClassB* const*)arg[1],
      *(const ::std::allocator<ClassB*>*)arg[2]);
  }
}

static void constructor_4041( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassB*>(*(::size_t*)arg[0]);
  else ::new(mem) ::std::vector<ClassB*>(*(::size_t*)arg[0]);
}

static void constructor_4042( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassB*>(*(const ::std::vector<ClassB*>*)arg[0]);
  else ::new(mem) ::std::vector<ClassB*>(*(const ::std::vector<ClassB*>*)arg[0]);
}

static void destructor_4043(void*, void * o, const std::vector<void*>&, void *) {
  (((::std::vector<ClassB*>*)o)->::std::vector<ClassB*>::~vector)();
}
static void operator_4044( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassB*>*)o)->operator=)(*(const ::std::vector<ClassB*>*)arg[0]);
else   (((::std::vector<ClassB*>*)o)->operator=)(*(const ::std::vector<ClassB*>*)arg[0]);
}

static void method_4045( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassB*>*)o)->assign)(*(::size_t*)arg[0],
    *(::ClassB* const*)arg[1]);
}

static void method_4046( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassB**,std::vector<ClassB*> >)((((::std::vector<ClassB*>*)o)->begin)());
else   (((::std::vector<ClassB*>*)o)->begin)();
}

static void method_4047( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassB* const*,std::vector<ClassB*> >)((((const ::std::vector<ClassB*>*)o)->begin)());
else   (((const ::std::vector<ClassB*>*)o)->begin)();
}

static void method_4048( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassB**,std::vector<ClassB*> >)((((::std::vector<ClassB*>*)o)->end)());
else   (((::std::vector<ClassB*>*)o)->end)();
}

static void method_4049( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassB* const*,std::vector<ClassB*> >)((((const ::std::vector<ClassB*>*)o)->end)());
else   (((const ::std::vector<ClassB*>*)o)->end)();
}

static void method_4054( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassB*>*)o)->size)());
else   (((const ::std::vector<ClassB*>*)o)->size)();
}

static void method_4055( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassB*>*)o)->max_size)());
else   (((const ::std::vector<ClassB*>*)o)->max_size)();
}

static void method_4056( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassB*>*)o)->resize)(*(::size_t*)arg[0],
    *(::ClassB* const*)arg[1]);
}

static void method_4057( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassB*>*)o)->resize)(*(::size_t*)arg[0]);
}

static void method_4058( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassB*>*)o)->capacity)());
else   (((const ::std::vector<ClassB*>*)o)->capacity)();
}

static void method_4059( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (bool)((((const ::std::vector<ClassB*>*)o)->empty)());
else   (((const ::std::vector<ClassB*>*)o)->empty)();
}

static void method_4060( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassB*>*)o)->reserve)(*(::size_t*)arg[0]);
}

static void operator_4061( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassB*>*)o)->operator[])(*(::size_t*)arg[0]);
else   (((::std::vector<ClassB*>*)o)->operator[])(*(::size_t*)arg[0]);
}

static void operator_4062( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassB*>*)o)->operator[])(*(::size_t*)arg[0]);
else   (((const ::std::vector<ClassB*>*)o)->operator[])(*(::size_t*)arg[0]);
}

static void method_4064( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassB*>*)o)->at)(*(::size_t*)arg[0]);
else   (((::std::vector<ClassB*>*)o)->at)(*(::size_t*)arg[0]);
}

static void method_4065( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassB*>*)o)->at)(*(::size_t*)arg[0]);
else   (((const ::std::vector<ClassB*>*)o)->at)(*(::size_t*)arg[0]);
}

static void method_4066( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassB*>*)o)->front)();
else   (((::std::vector<ClassB*>*)o)->front)();
}

static void method_4067( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassB*>*)o)->front)();
else   (((const ::std::vector<ClassB*>*)o)->front)();
}

static void method_4068( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassB*>*)o)->back)();
else   (((::std::vector<ClassB*>*)o)->back)();
}

static void method_4069( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassB*>*)o)->back)();
else   (((const ::std::vector<ClassB*>*)o)->back)();
}

static void method_4070( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassB*>*)o)->push_back)(*(::ClassB* const*)arg[0]);
}

static void method_4071( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<ClassB*>*)o)->pop_back)();
}

static void method_4072( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassB**,std::vector<ClassB*> >)((((::std::vector<ClassB*>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassB**,std::vector<ClassB*> >*)arg[0],
    *(::ClassB* const*)arg[1]));
else   (((::std::vector<ClassB*>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassB**,std::vector<ClassB*> >*)arg[0],
    *(::ClassB* const*)arg[1]);
}

static void method_4073( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassB*>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassB**,std::vector<ClassB*> >*)arg[0],
    *(::size_t*)arg[1],
    *(::ClassB* const*)arg[2]);
}

static void method_4074( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassB**,std::vector<ClassB*> >)((((::std::vector<ClassB*>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassB**,std::vector<ClassB*> >*)arg[0]));
else   (((::std::vector<ClassB*>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassB**,std::vector<ClassB*> >*)arg[0]);
}

static void method_4075( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassB**,std::vector<ClassB*> >)((((::std::vector<ClassB*>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassB**,std::vector<ClassB*> >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<ClassB**,std::vector<ClassB*> >*)arg[1]));
else   (((::std::vector<ClassB*>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassB**,std::vector<ClassB*> >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<ClassB**,std::vector<ClassB*> >*)arg[1]);
}

static void method_4076( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassB*>*)o)->swap)(*(::std::vector<ClassB*>*)arg[0]);
}

static void method_4077( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<ClassB*>*)o)->clear)();
}

static void constructor_x25( void* retaddr, void* mem, const std::vector<void*>&, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassB*>();
  else ::new(mem) ::std::vector<ClassB*>();
}

static void method_newdel_1386( void* retaddr, void*, const std::vector<void*>&, void*)
{
  static ::Reflex::NewDelFunctions s_funcs;
  s_funcs.fNew         = ::Reflex::NewDelFunctionsT< ::std::vector<ClassB*> >::new_T;
  s_funcs.fNewArray    = ::Reflex::NewDelFunctionsT< ::std::vector<ClassB*> >::newArray_T;
  s_funcs.fDelete      = ::Reflex::NewDelFunctionsT< ::std::vector<ClassB*> >::delete_T;
  s_funcs.fDeleteArray = ::Reflex::NewDelFunctionsT< ::std::vector<ClassB*> >::deleteArray_T;
  s_funcs.fDestructor  = ::Reflex::NewDelFunctionsT< ::std::vector<ClassB*> >::destruct_T;
  if (retaddr) *(::Reflex::NewDelFunctions**)retaddr = &s_funcs;
}

static void method_x27( void* retaddr, void*, const std::vector<void*>&, void*)
{
  typedef std::vector<std::pair< ::Reflex::Base, int> > Bases_t;
  static Bases_t s_bases;
  if ( !s_bases.size() ) {
    s_bases.push_back(std::make_pair(::Reflex::Base( ::Reflex::TypeBuilder("std::_Vector_base<ClassB*,std::allocator<ClassB*> >"), ::Reflex::BaseOffset< ::std::vector<ClassB*>,::std::_Vector_base<ClassB*,std::allocator<ClassB*> > >::Get(),::Reflex::PROTECTED), 0));
  }
  if (retaddr) *(Bases_t**)retaddr = &s_bases;
}

static void method_x28( void* retaddr, void*, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**) retaddr = ::Reflex::Proxy< ::std::vector<ClassB*> >::Generate();
  else ::Reflex::Proxy< ::std::vector<ClassB*> >::Generate();
}

//------Dictionary for class vector<ClassB*,std::allocator<ClassB*> > -------------------------------
void __std__vector_ClassBp__db_datamem(Reflex::Class*);
void __std__vector_ClassBp__db_funcmem(Reflex::Class*);
Reflex::GenreflexMemberBuilder __std__vector_ClassBp__datamem_bld(&__std__vector_ClassBp__db_datamem);
Reflex::GenreflexMemberBuilder __std__vector_ClassBp__funcmem_bld(&__std__vector_ClassBp__db_funcmem);
void __std__vector_ClassBp__dict() {
  ::Reflex::ClassBuilder("std::vector<ClassB*>", typeid(::std::vector<ClassB*>), sizeof(::std::vector<ClassB*>), ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL, ::Reflex::CLASS)
  .AddBase(type_1642, ::Reflex::BaseOffset< ::std::vector<ClassB*>, ::std::_Vector_base<ClassB*,std::allocator<ClassB*> > >::Get(), ::Reflex::PROTECTED)
  .AddTypedef(type_1642, "std::vector<ClassB*>::_Base")
  .AddTypedef(type_4022, "std::vector<ClassB*>::value_type")
  .AddTypedef(type_4024, "std::vector<ClassB*>::pointer")
  .AddTypedef(type_4026, "std::vector<ClassB*>::const_pointer")
  .AddTypedef(type_4028, "std::vector<ClassB*>::reference")
  .AddTypedef(type_4030, "std::vector<ClassB*>::const_reference")
  .AddTypedef(type_2319, "std::vector<ClassB*>::iterator")
  .AddTypedef(type_2320, "std::vector<ClassB*>::const_iterator")
  .AddTypedef(type_1616, "std::vector<ClassB*>::const_reverse_iterator")
  .AddTypedef(type_1617, "std::vector<ClassB*>::reverse_iterator")
  .AddTypedef(type_854, "std::vector<ClassB*>::size_type")
  .AddTypedef(type_684, "std::vector<ClassB*>::difference_type")
  .AddTypedef(type_1536, "std::vector<ClassB*>::allocator_type")
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7273), "vector", constructor_4039, 0, "__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854, type_4030, type_7273), "vector", constructor_4040, 0, "__n;__value;__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854), "vector", constructor_4041, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7274), "vector", constructor_4042, 0, "__x", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "~vector", destructor_4043, 0, 0, ::Reflex::PUBLIC | ::Reflex::DESTRUCTOR )
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "vector", constructor_x25, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember<void*(void)>("__getNewDelFunctions", method_newdel_1386, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("__getBasesTable", method_x27, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("createCollFuncTable", method_x28, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddOnDemandFunctionMemberBuilder(&__std__vector_ClassBp__funcmem_bld);
}

//------Delayed data member builder for class vector<ClassB*,std::allocator<ClassB*> > -------------------
void __std__vector_ClassBp__db_datamem(Reflex::Class*) {

}
//------Delayed function member builder for class vector<ClassB*,std::allocator<ClassB*> > -------------------
void __std__vector_ClassBp__db_funcmem(Reflex::Class* cl) {
  ::Reflex::ClassBuilder(cl)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_7275, type_7274), "operator=", operator_4044, 0, "__x", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_4030), "assign", method_4045, 0, "__n;__val", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2319), "begin", method_4046, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2320), "begin", method_4047, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2319), "end", method_4048, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2320), "end", method_4049, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "size", method_4054, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "max_size", method_4055, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_4030), "resize", method_4056, 0, "__new_size;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "resize", method_4057, 0, "__new_size", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "capacity", method_4058, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2653), "empty", method_4059, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "reserve", method_4060, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4028, type_854), "operator[]", operator_4061, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4030, type_854), "operator[]", operator_4062, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4028, type_854), "at", method_4064, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4030, type_854), "at", method_4065, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4028), "front", method_4066, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4030), "front", method_4067, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4028), "back", method_4068, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4030), "back", method_4069, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_4030), "push_back", method_4070, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "pop_back", method_4071, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2319, type_2319, type_4030), "insert", method_4072, 0, "__position;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_2319, type_854, type_4030), "insert", method_4073, 0, "__position;__n;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2319, type_2319), "erase", method_4074, 0, "__position", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2319, type_2319, type_2319), "erase", method_4075, 0, "__first;__last", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_7275), "swap", method_4076, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "clear", method_4077, 0, 0, ::Reflex::PUBLIC);
}
//------Stub functions for class vector<ClassB,std::allocator<ClassB> > -------------------------------
static void constructor_4097( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 0 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassB>();
  else ::new(mem) ::std::vector<ClassB>();
  }
  else if ( arg.size() == 1 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassB>(*(const ::std::allocator<ClassB>*)arg[0]);
  else ::new(mem) ::std::vector<ClassB>(*(const ::std::allocator<ClassB>*)arg[0]);
  }
}

static void constructor_4098( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 2 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassB>(*(::size_t*)arg[0],
      *(const ::ClassB*)arg[1]);
  else ::new(mem) ::std::vector<ClassB>(*(::size_t*)arg[0],
      *(const ::ClassB*)arg[1]);
  }
  else if ( arg.size() == 3 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassB>(*(::size_t*)arg[0],
      *(const ::ClassB*)arg[1],
      *(const ::std::allocator<ClassB>*)arg[2]);
  else ::new(mem) ::std::vector<ClassB>(*(::size_t*)arg[0],
      *(const ::ClassB*)arg[1],
      *(const ::std::allocator<ClassB>*)arg[2]);
  }
}

static void constructor_4099( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassB>(*(::size_t*)arg[0]);
  else ::new(mem) ::std::vector<ClassB>(*(::size_t*)arg[0]);
}

static void constructor_4100( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassB>(*(const ::std::vector<ClassB>*)arg[0]);
  else ::new(mem) ::std::vector<ClassB>(*(const ::std::vector<ClassB>*)arg[0]);
}

static void destructor_4101(void*, void * o, const std::vector<void*>&, void *) {
  (((::std::vector<ClassB>*)o)->::std::vector<ClassB>::~vector)();
}
static void operator_4102( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassB>*)o)->operator=)(*(const ::std::vector<ClassB>*)arg[0]);
else   (((::std::vector<ClassB>*)o)->operator=)(*(const ::std::vector<ClassB>*)arg[0]);
}

static void method_4103( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassB>*)o)->assign)(*(::size_t*)arg[0],
    *(const ::ClassB*)arg[1]);
}

static void method_4104( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassB*,std::vector<ClassB> >)((((::std::vector<ClassB>*)o)->begin)());
else   (((::std::vector<ClassB>*)o)->begin)();
}

static void method_4105( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<const ClassB*,std::vector<ClassB> >)((((const ::std::vector<ClassB>*)o)->begin)());
else   (((const ::std::vector<ClassB>*)o)->begin)();
}

static void method_4106( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassB*,std::vector<ClassB> >)((((::std::vector<ClassB>*)o)->end)());
else   (((::std::vector<ClassB>*)o)->end)();
}

static void method_4107( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<const ClassB*,std::vector<ClassB> >)((((const ::std::vector<ClassB>*)o)->end)());
else   (((const ::std::vector<ClassB>*)o)->end)();
}

static void method_4112( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassB>*)o)->size)());
else   (((const ::std::vector<ClassB>*)o)->size)();
}

static void method_4113( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassB>*)o)->max_size)());
else   (((const ::std::vector<ClassB>*)o)->max_size)();
}

static void method_4114( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassB>*)o)->resize)(*(::size_t*)arg[0],
    *(const ::ClassB*)arg[1]);
}

static void method_4115( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassB>*)o)->resize)(*(::size_t*)arg[0]);
}

static void method_4116( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassB>*)o)->capacity)());
else   (((const ::std::vector<ClassB>*)o)->capacity)();
}

static void method_4117( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (bool)((((const ::std::vector<ClassB>*)o)->empty)());
else   (((const ::std::vector<ClassB>*)o)->empty)();
}

static void method_4118( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassB>*)o)->reserve)(*(::size_t*)arg[0]);
}

static void operator_4119( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassB>*)o)->operator[])(*(::size_t*)arg[0]);
else   (((::std::vector<ClassB>*)o)->operator[])(*(::size_t*)arg[0]);
}

static void operator_4120( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassB>*)o)->operator[])(*(::size_t*)arg[0]);
else   (((const ::std::vector<ClassB>*)o)->operator[])(*(::size_t*)arg[0]);
}

static void method_4122( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassB>*)o)->at)(*(::size_t*)arg[0]);
else   (((::std::vector<ClassB>*)o)->at)(*(::size_t*)arg[0]);
}

static void method_4123( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassB>*)o)->at)(*(::size_t*)arg[0]);
else   (((const ::std::vector<ClassB>*)o)->at)(*(::size_t*)arg[0]);
}

static void method_4124( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassB>*)o)->front)();
else   (((::std::vector<ClassB>*)o)->front)();
}

static void method_4125( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassB>*)o)->front)();
else   (((const ::std::vector<ClassB>*)o)->front)();
}

static void method_4126( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassB>*)o)->back)();
else   (((::std::vector<ClassB>*)o)->back)();
}

static void method_4127( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassB>*)o)->back)();
else   (((const ::std::vector<ClassB>*)o)->back)();
}

static void method_4128( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassB>*)o)->push_back)(*(const ::ClassB*)arg[0]);
}

static void method_4129( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<ClassB>*)o)->pop_back)();
}

static void method_4130( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassB*,std::vector<ClassB> >)((((::std::vector<ClassB>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassB*,std::vector<ClassB> >*)arg[0],
    *(const ::ClassB*)arg[1]));
else   (((::std::vector<ClassB>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassB*,std::vector<ClassB> >*)arg[0],
    *(const ::ClassB*)arg[1]);
}

static void method_4131( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassB>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassB*,std::vector<ClassB> >*)arg[0],
    *(::size_t*)arg[1],
    *(const ::ClassB*)arg[2]);
}

static void method_4132( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassB*,std::vector<ClassB> >)((((::std::vector<ClassB>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassB*,std::vector<ClassB> >*)arg[0]));
else   (((::std::vector<ClassB>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassB*,std::vector<ClassB> >*)arg[0]);
}

static void method_4133( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassB*,std::vector<ClassB> >)((((::std::vector<ClassB>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassB*,std::vector<ClassB> >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<ClassB*,std::vector<ClassB> >*)arg[1]));
else   (((::std::vector<ClassB>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassB*,std::vector<ClassB> >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<ClassB*,std::vector<ClassB> >*)arg[1]);
}

static void method_4134( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassB>*)o)->swap)(*(::std::vector<ClassB>*)arg[0]);
}

static void method_4135( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<ClassB>*)o)->clear)();
}

static void constructor_x29( void* retaddr, void* mem, const std::vector<void*>&, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassB>();
  else ::new(mem) ::std::vector<ClassB>();
}

static void method_newdel_1387( void* retaddr, void*, const std::vector<void*>&, void*)
{
  static ::Reflex::NewDelFunctions s_funcs;
  s_funcs.fNew         = ::Reflex::NewDelFunctionsT< ::std::vector<ClassB> >::new_T;
  s_funcs.fNewArray    = ::Reflex::NewDelFunctionsT< ::std::vector<ClassB> >::newArray_T;
  s_funcs.fDelete      = ::Reflex::NewDelFunctionsT< ::std::vector<ClassB> >::delete_T;
  s_funcs.fDeleteArray = ::Reflex::NewDelFunctionsT< ::std::vector<ClassB> >::deleteArray_T;
  s_funcs.fDestructor  = ::Reflex::NewDelFunctionsT< ::std::vector<ClassB> >::destruct_T;
  if (retaddr) *(::Reflex::NewDelFunctions**)retaddr = &s_funcs;
}

static void method_x31( void* retaddr, void*, const std::vector<void*>&, void*)
{
  typedef std::vector<std::pair< ::Reflex::Base, int> > Bases_t;
  static Bases_t s_bases;
  if ( !s_bases.size() ) {
    s_bases.push_back(std::make_pair(::Reflex::Base( ::Reflex::TypeBuilder("std::_Vector_base<ClassB,std::allocator<ClassB> >"), ::Reflex::BaseOffset< ::std::vector<ClassB>,::std::_Vector_base<ClassB,std::allocator<ClassB> > >::Get(),::Reflex::PROTECTED), 0));
  }
  if (retaddr) *(Bases_t**)retaddr = &s_bases;
}

static void method_x32( void* retaddr, void*, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**) retaddr = ::Reflex::Proxy< ::std::vector<ClassB> >::Generate();
  else ::Reflex::Proxy< ::std::vector<ClassB> >::Generate();
}

//------Dictionary for class vector<ClassB,std::allocator<ClassB> > -------------------------------
void __std__vector_ClassB__db_datamem(Reflex::Class*);
void __std__vector_ClassB__db_funcmem(Reflex::Class*);
Reflex::GenreflexMemberBuilder __std__vector_ClassB__datamem_bld(&__std__vector_ClassB__db_datamem);
Reflex::GenreflexMemberBuilder __std__vector_ClassB__funcmem_bld(&__std__vector_ClassB__db_funcmem);
void __std__vector_ClassB__dict() {
  ::Reflex::ClassBuilder("std::vector<ClassB>", typeid(::std::vector<ClassB>), sizeof(::std::vector<ClassB>), ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL, ::Reflex::CLASS)
  .AddBase(type_1643, ::Reflex::BaseOffset< ::std::vector<ClassB>, ::std::_Vector_base<ClassB,std::allocator<ClassB> > >::Get(), ::Reflex::PROTECTED)
  .AddTypedef(type_1643, "std::vector<ClassB>::_Base")
  .AddTypedef(type_995, "std::vector<ClassB>::value_type")
  .AddTypedef(type_4022, "std::vector<ClassB>::pointer")
  .AddTypedef(type_4084, "std::vector<ClassB>::const_pointer")
  .AddTypedef(type_4086, "std::vector<ClassB>::reference")
  .AddTypedef(type_4088, "std::vector<ClassB>::const_reference")
  .AddTypedef(type_2321, "std::vector<ClassB>::iterator")
  .AddTypedef(type_2322, "std::vector<ClassB>::const_iterator")
  .AddTypedef(type_1618, "std::vector<ClassB>::const_reverse_iterator")
  .AddTypedef(type_1619, "std::vector<ClassB>::reverse_iterator")
  .AddTypedef(type_854, "std::vector<ClassB>::size_type")
  .AddTypedef(type_684, "std::vector<ClassB>::difference_type")
  .AddTypedef(type_1537, "std::vector<ClassB>::allocator_type")
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7277), "vector", constructor_4097, 0, "__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854, type_4088, type_7277), "vector", constructor_4098, 0, "__n;__value;__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854), "vector", constructor_4099, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7278), "vector", constructor_4100, 0, "__x", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "~vector", destructor_4101, 0, 0, ::Reflex::PUBLIC | ::Reflex::DESTRUCTOR )
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "vector", constructor_x29, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember<void*(void)>("__getNewDelFunctions", method_newdel_1387, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("__getBasesTable", method_x31, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("createCollFuncTable", method_x32, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddOnDemandFunctionMemberBuilder(&__std__vector_ClassB__funcmem_bld);
}

//------Delayed data member builder for class vector<ClassB,std::allocator<ClassB> > -------------------
void __std__vector_ClassB__db_datamem(Reflex::Class*) {

}
//------Delayed function member builder for class vector<ClassB,std::allocator<ClassB> > -------------------
void __std__vector_ClassB__db_funcmem(Reflex::Class* cl) {
  ::Reflex::ClassBuilder(cl)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_7279, type_7278), "operator=", operator_4102, 0, "__x", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_4088), "assign", method_4103, 0, "__n;__val", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2321), "begin", method_4104, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2322), "begin", method_4105, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2321), "end", method_4106, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2322), "end", method_4107, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "size", method_4112, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "max_size", method_4113, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_4088), "resize", method_4114, 0, "__new_size;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "resize", method_4115, 0, "__new_size", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "capacity", method_4116, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2653), "empty", method_4117, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "reserve", method_4118, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4086, type_854), "operator[]", operator_4119, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4088, type_854), "operator[]", operator_4120, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4086, type_854), "at", method_4122, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4088, type_854), "at", method_4123, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4086), "front", method_4124, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4088), "front", method_4125, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4086), "back", method_4126, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4088), "back", method_4127, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_4088), "push_back", method_4128, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "pop_back", method_4129, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2321, type_2321, type_4088), "insert", method_4130, 0, "__position;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_2321, type_854, type_4088), "insert", method_4131, 0, "__position;__n;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2321, type_2321), "erase", method_4132, 0, "__position", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2321, type_2321, type_2321), "erase", method_4133, 0, "__first;__last", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_7279), "swap", method_4134, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "clear", method_4135, 0, 0, ::Reflex::PUBLIC);
}
//------Stub functions for class vector<ClassA*,std::allocator<ClassA*> > -------------------------------
static void constructor_4157( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 0 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassA*>();
  else ::new(mem) ::std::vector<ClassA*>();
  }
  else if ( arg.size() == 1 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassA*>(*(const ::std::allocator<ClassA*>*)arg[0]);
  else ::new(mem) ::std::vector<ClassA*>(*(const ::std::allocator<ClassA*>*)arg[0]);
  }
}

static void constructor_4158( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 2 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassA*>(*(::size_t*)arg[0],
      *(::ClassA* const*)arg[1]);
  else ::new(mem) ::std::vector<ClassA*>(*(::size_t*)arg[0],
      *(::ClassA* const*)arg[1]);
  }
  else if ( arg.size() == 3 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassA*>(*(::size_t*)arg[0],
      *(::ClassA* const*)arg[1],
      *(const ::std::allocator<ClassA*>*)arg[2]);
  else ::new(mem) ::std::vector<ClassA*>(*(::size_t*)arg[0],
      *(::ClassA* const*)arg[1],
      *(const ::std::allocator<ClassA*>*)arg[2]);
  }
}

static void constructor_4159( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassA*>(*(::size_t*)arg[0]);
  else ::new(mem) ::std::vector<ClassA*>(*(::size_t*)arg[0]);
}

static void constructor_4160( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassA*>(*(const ::std::vector<ClassA*>*)arg[0]);
  else ::new(mem) ::std::vector<ClassA*>(*(const ::std::vector<ClassA*>*)arg[0]);
}

static void destructor_4161(void*, void * o, const std::vector<void*>&, void *) {
  (((::std::vector<ClassA*>*)o)->::std::vector<ClassA*>::~vector)();
}
static void operator_4162( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassA*>*)o)->operator=)(*(const ::std::vector<ClassA*>*)arg[0]);
else   (((::std::vector<ClassA*>*)o)->operator=)(*(const ::std::vector<ClassA*>*)arg[0]);
}

static void method_4163( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassA*>*)o)->assign)(*(::size_t*)arg[0],
    *(::ClassA* const*)arg[1]);
}

static void method_4164( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassA**,std::vector<ClassA*> >)((((::std::vector<ClassA*>*)o)->begin)());
else   (((::std::vector<ClassA*>*)o)->begin)();
}

static void method_4165( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassA* const*,std::vector<ClassA*> >)((((const ::std::vector<ClassA*>*)o)->begin)());
else   (((const ::std::vector<ClassA*>*)o)->begin)();
}

static void method_4166( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassA**,std::vector<ClassA*> >)((((::std::vector<ClassA*>*)o)->end)());
else   (((::std::vector<ClassA*>*)o)->end)();
}

static void method_4167( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassA* const*,std::vector<ClassA*> >)((((const ::std::vector<ClassA*>*)o)->end)());
else   (((const ::std::vector<ClassA*>*)o)->end)();
}

static void method_4172( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassA*>*)o)->size)());
else   (((const ::std::vector<ClassA*>*)o)->size)();
}

static void method_4173( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassA*>*)o)->max_size)());
else   (((const ::std::vector<ClassA*>*)o)->max_size)();
}

static void method_4174( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassA*>*)o)->resize)(*(::size_t*)arg[0],
    *(::ClassA* const*)arg[1]);
}

static void method_4175( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassA*>*)o)->resize)(*(::size_t*)arg[0]);
}

static void method_4176( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassA*>*)o)->capacity)());
else   (((const ::std::vector<ClassA*>*)o)->capacity)();
}

static void method_4177( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (bool)((((const ::std::vector<ClassA*>*)o)->empty)());
else   (((const ::std::vector<ClassA*>*)o)->empty)();
}

static void method_4178( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassA*>*)o)->reserve)(*(::size_t*)arg[0]);
}

static void operator_4179( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassA*>*)o)->operator[])(*(::size_t*)arg[0]);
else   (((::std::vector<ClassA*>*)o)->operator[])(*(::size_t*)arg[0]);
}

static void operator_4180( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassA*>*)o)->operator[])(*(::size_t*)arg[0]);
else   (((const ::std::vector<ClassA*>*)o)->operator[])(*(::size_t*)arg[0]);
}

static void method_4182( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassA*>*)o)->at)(*(::size_t*)arg[0]);
else   (((::std::vector<ClassA*>*)o)->at)(*(::size_t*)arg[0]);
}

static void method_4183( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassA*>*)o)->at)(*(::size_t*)arg[0]);
else   (((const ::std::vector<ClassA*>*)o)->at)(*(::size_t*)arg[0]);
}

static void method_4184( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassA*>*)o)->front)();
else   (((::std::vector<ClassA*>*)o)->front)();
}

static void method_4185( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassA*>*)o)->front)();
else   (((const ::std::vector<ClassA*>*)o)->front)();
}

static void method_4186( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassA*>*)o)->back)();
else   (((::std::vector<ClassA*>*)o)->back)();
}

static void method_4187( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassA*>*)o)->back)();
else   (((const ::std::vector<ClassA*>*)o)->back)();
}

static void method_4188( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassA*>*)o)->push_back)(*(::ClassA* const*)arg[0]);
}

static void method_4189( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<ClassA*>*)o)->pop_back)();
}

static void method_4190( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassA**,std::vector<ClassA*> >)((((::std::vector<ClassA*>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassA**,std::vector<ClassA*> >*)arg[0],
    *(::ClassA* const*)arg[1]));
else   (((::std::vector<ClassA*>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassA**,std::vector<ClassA*> >*)arg[0],
    *(::ClassA* const*)arg[1]);
}

static void method_4191( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassA*>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassA**,std::vector<ClassA*> >*)arg[0],
    *(::size_t*)arg[1],
    *(::ClassA* const*)arg[2]);
}

static void method_4192( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassA**,std::vector<ClassA*> >)((((::std::vector<ClassA*>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassA**,std::vector<ClassA*> >*)arg[0]));
else   (((::std::vector<ClassA*>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassA**,std::vector<ClassA*> >*)arg[0]);
}

static void method_4193( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassA**,std::vector<ClassA*> >)((((::std::vector<ClassA*>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassA**,std::vector<ClassA*> >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<ClassA**,std::vector<ClassA*> >*)arg[1]));
else   (((::std::vector<ClassA*>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassA**,std::vector<ClassA*> >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<ClassA**,std::vector<ClassA*> >*)arg[1]);
}

static void method_4194( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassA*>*)o)->swap)(*(::std::vector<ClassA*>*)arg[0]);
}

static void method_4195( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<ClassA*>*)o)->clear)();
}

static void constructor_x33( void* retaddr, void* mem, const std::vector<void*>&, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassA*>();
  else ::new(mem) ::std::vector<ClassA*>();
}

static void method_newdel_1388( void* retaddr, void*, const std::vector<void*>&, void*)
{
  static ::Reflex::NewDelFunctions s_funcs;
  s_funcs.fNew         = ::Reflex::NewDelFunctionsT< ::std::vector<ClassA*> >::new_T;
  s_funcs.fNewArray    = ::Reflex::NewDelFunctionsT< ::std::vector<ClassA*> >::newArray_T;
  s_funcs.fDelete      = ::Reflex::NewDelFunctionsT< ::std::vector<ClassA*> >::delete_T;
  s_funcs.fDeleteArray = ::Reflex::NewDelFunctionsT< ::std::vector<ClassA*> >::deleteArray_T;
  s_funcs.fDestructor  = ::Reflex::NewDelFunctionsT< ::std::vector<ClassA*> >::destruct_T;
  if (retaddr) *(::Reflex::NewDelFunctions**)retaddr = &s_funcs;
}

static void method_x35( void* retaddr, void*, const std::vector<void*>&, void*)
{
  typedef std::vector<std::pair< ::Reflex::Base, int> > Bases_t;
  static Bases_t s_bases;
  if ( !s_bases.size() ) {
    s_bases.push_back(std::make_pair(::Reflex::Base( ::Reflex::TypeBuilder("std::_Vector_base<ClassA*,std::allocator<ClassA*> >"), ::Reflex::BaseOffset< ::std::vector<ClassA*>,::std::_Vector_base<ClassA*,std::allocator<ClassA*> > >::Get(),::Reflex::PROTECTED), 0));
  }
  if (retaddr) *(Bases_t**)retaddr = &s_bases;
}

static void method_x36( void* retaddr, void*, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**) retaddr = ::Reflex::Proxy< ::std::vector<ClassA*> >::Generate();
  else ::Reflex::Proxy< ::std::vector<ClassA*> >::Generate();
}

//------Dictionary for class vector<ClassA*,std::allocator<ClassA*> > -------------------------------
void __std__vector_ClassAp__db_datamem(Reflex::Class*);
void __std__vector_ClassAp__db_funcmem(Reflex::Class*);
Reflex::GenreflexMemberBuilder __std__vector_ClassAp__datamem_bld(&__std__vector_ClassAp__db_datamem);
Reflex::GenreflexMemberBuilder __std__vector_ClassAp__funcmem_bld(&__std__vector_ClassAp__db_funcmem);
void __std__vector_ClassAp__dict() {
  ::Reflex::ClassBuilder("std::vector<ClassA*>", typeid(::std::vector<ClassA*>), sizeof(::std::vector<ClassA*>), ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL, ::Reflex::CLASS)
  .AddBase(type_1646, ::Reflex::BaseOffset< ::std::vector<ClassA*>, ::std::_Vector_base<ClassA*,std::allocator<ClassA*> > >::Get(), ::Reflex::PROTECTED)
  .AddTypedef(type_1646, "std::vector<ClassA*>::_Base")
  .AddTypedef(type_4140, "std::vector<ClassA*>::value_type")
  .AddTypedef(type_4142, "std::vector<ClassA*>::pointer")
  .AddTypedef(type_4144, "std::vector<ClassA*>::const_pointer")
  .AddTypedef(type_4146, "std::vector<ClassA*>::reference")
  .AddTypedef(type_4148, "std::vector<ClassA*>::const_reference")
  .AddTypedef(type_2327, "std::vector<ClassA*>::iterator")
  .AddTypedef(type_2328, "std::vector<ClassA*>::const_iterator")
  .AddTypedef(type_1624, "std::vector<ClassA*>::const_reverse_iterator")
  .AddTypedef(type_1625, "std::vector<ClassA*>::reverse_iterator")
  .AddTypedef(type_854, "std::vector<ClassA*>::size_type")
  .AddTypedef(type_684, "std::vector<ClassA*>::difference_type")
  .AddTypedef(type_1538, "std::vector<ClassA*>::allocator_type")
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7281), "vector", constructor_4157, 0, "__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854, type_4148, type_7281), "vector", constructor_4158, 0, "__n;__value;__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854), "vector", constructor_4159, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7282), "vector", constructor_4160, 0, "__x", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "~vector", destructor_4161, 0, 0, ::Reflex::PUBLIC | ::Reflex::DESTRUCTOR )
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "vector", constructor_x33, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember<void*(void)>("__getNewDelFunctions", method_newdel_1388, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("__getBasesTable", method_x35, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("createCollFuncTable", method_x36, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddOnDemandFunctionMemberBuilder(&__std__vector_ClassAp__funcmem_bld);
}

//------Delayed data member builder for class vector<ClassA*,std::allocator<ClassA*> > -------------------
void __std__vector_ClassAp__db_datamem(Reflex::Class*) {

}
//------Delayed function member builder for class vector<ClassA*,std::allocator<ClassA*> > -------------------
void __std__vector_ClassAp__db_funcmem(Reflex::Class* cl) {
  ::Reflex::ClassBuilder(cl)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_7283, type_7282), "operator=", operator_4162, 0, "__x", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_4148), "assign", method_4163, 0, "__n;__val", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2327), "begin", method_4164, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2328), "begin", method_4165, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2327), "end", method_4166, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2328), "end", method_4167, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "size", method_4172, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "max_size", method_4173, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_4148), "resize", method_4174, 0, "__new_size;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "resize", method_4175, 0, "__new_size", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "capacity", method_4176, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2653), "empty", method_4177, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "reserve", method_4178, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4146, type_854), "operator[]", operator_4179, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4148, type_854), "operator[]", operator_4180, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4146, type_854), "at", method_4182, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4148, type_854), "at", method_4183, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4146), "front", method_4184, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4148), "front", method_4185, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4146), "back", method_4186, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4148), "back", method_4187, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_4148), "push_back", method_4188, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "pop_back", method_4189, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2327, type_2327, type_4148), "insert", method_4190, 0, "__position;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_2327, type_854, type_4148), "insert", method_4191, 0, "__position;__n;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2327, type_2327), "erase", method_4192, 0, "__position", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2327, type_2327, type_2327), "erase", method_4193, 0, "__first;__last", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_7283), "swap", method_4194, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "clear", method_4195, 0, 0, ::Reflex::PUBLIC);
}
//------Stub functions for class vector<ClassA,std::allocator<ClassA> > -------------------------------
static void constructor_4215( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 0 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassA>();
  else ::new(mem) ::std::vector<ClassA>();
  }
  else if ( arg.size() == 1 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassA>(*(const ::std::allocator<ClassA>*)arg[0]);
  else ::new(mem) ::std::vector<ClassA>(*(const ::std::allocator<ClassA>*)arg[0]);
  }
}

static void constructor_4216( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 2 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassA>(*(::size_t*)arg[0],
      *(const ::ClassA*)arg[1]);
  else ::new(mem) ::std::vector<ClassA>(*(::size_t*)arg[0],
      *(const ::ClassA*)arg[1]);
  }
  else if ( arg.size() == 3 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassA>(*(::size_t*)arg[0],
      *(const ::ClassA*)arg[1],
      *(const ::std::allocator<ClassA>*)arg[2]);
  else ::new(mem) ::std::vector<ClassA>(*(::size_t*)arg[0],
      *(const ::ClassA*)arg[1],
      *(const ::std::allocator<ClassA>*)arg[2]);
  }
}

static void constructor_4217( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassA>(*(::size_t*)arg[0]);
  else ::new(mem) ::std::vector<ClassA>(*(::size_t*)arg[0]);
}

static void constructor_4218( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassA>(*(const ::std::vector<ClassA>*)arg[0]);
  else ::new(mem) ::std::vector<ClassA>(*(const ::std::vector<ClassA>*)arg[0]);
}

static void destructor_4219(void*, void * o, const std::vector<void*>&, void *) {
  (((::std::vector<ClassA>*)o)->::std::vector<ClassA>::~vector)();
}
static void operator_4220( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassA>*)o)->operator=)(*(const ::std::vector<ClassA>*)arg[0]);
else   (((::std::vector<ClassA>*)o)->operator=)(*(const ::std::vector<ClassA>*)arg[0]);
}

static void method_4221( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassA>*)o)->assign)(*(::size_t*)arg[0],
    *(const ::ClassA*)arg[1]);
}

static void method_4222( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassA*,std::vector<ClassA> >)((((::std::vector<ClassA>*)o)->begin)());
else   (((::std::vector<ClassA>*)o)->begin)();
}

static void method_4223( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<const ClassA*,std::vector<ClassA> >)((((const ::std::vector<ClassA>*)o)->begin)());
else   (((const ::std::vector<ClassA>*)o)->begin)();
}

static void method_4224( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassA*,std::vector<ClassA> >)((((::std::vector<ClassA>*)o)->end)());
else   (((::std::vector<ClassA>*)o)->end)();
}

static void method_4225( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<const ClassA*,std::vector<ClassA> >)((((const ::std::vector<ClassA>*)o)->end)());
else   (((const ::std::vector<ClassA>*)o)->end)();
}

static void method_4230( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassA>*)o)->size)());
else   (((const ::std::vector<ClassA>*)o)->size)();
}

static void method_4231( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassA>*)o)->max_size)());
else   (((const ::std::vector<ClassA>*)o)->max_size)();
}

static void method_4232( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassA>*)o)->resize)(*(::size_t*)arg[0],
    *(const ::ClassA*)arg[1]);
}

static void method_4233( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassA>*)o)->resize)(*(::size_t*)arg[0]);
}

static void method_4234( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<ClassA>*)o)->capacity)());
else   (((const ::std::vector<ClassA>*)o)->capacity)();
}

static void method_4235( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (bool)((((const ::std::vector<ClassA>*)o)->empty)());
else   (((const ::std::vector<ClassA>*)o)->empty)();
}

static void method_4236( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassA>*)o)->reserve)(*(::size_t*)arg[0]);
}

static void operator_4237( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassA>*)o)->operator[])(*(::size_t*)arg[0]);
else   (((::std::vector<ClassA>*)o)->operator[])(*(::size_t*)arg[0]);
}

static void operator_4238( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassA>*)o)->operator[])(*(::size_t*)arg[0]);
else   (((const ::std::vector<ClassA>*)o)->operator[])(*(::size_t*)arg[0]);
}

static void method_4240( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassA>*)o)->at)(*(::size_t*)arg[0]);
else   (((::std::vector<ClassA>*)o)->at)(*(::size_t*)arg[0]);
}

static void method_4241( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassA>*)o)->at)(*(::size_t*)arg[0]);
else   (((const ::std::vector<ClassA>*)o)->at)(*(::size_t*)arg[0]);
}

static void method_4242( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassA>*)o)->front)();
else   (((::std::vector<ClassA>*)o)->front)();
}

static void method_4243( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassA>*)o)->front)();
else   (((const ::std::vector<ClassA>*)o)->front)();
}

static void method_4244( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<ClassA>*)o)->back)();
else   (((::std::vector<ClassA>*)o)->back)();
}

static void method_4245( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<ClassA>*)o)->back)();
else   (((const ::std::vector<ClassA>*)o)->back)();
}

static void method_4246( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassA>*)o)->push_back)(*(const ::ClassA*)arg[0]);
}

static void method_4247( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<ClassA>*)o)->pop_back)();
}

static void method_4248( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassA*,std::vector<ClassA> >)((((::std::vector<ClassA>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassA*,std::vector<ClassA> >*)arg[0],
    *(const ::ClassA*)arg[1]));
else   (((::std::vector<ClassA>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassA*,std::vector<ClassA> >*)arg[0],
    *(const ::ClassA*)arg[1]);
}

static void method_4249( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassA>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<ClassA*,std::vector<ClassA> >*)arg[0],
    *(::size_t*)arg[1],
    *(const ::ClassA*)arg[2]);
}

static void method_4250( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassA*,std::vector<ClassA> >)((((::std::vector<ClassA>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassA*,std::vector<ClassA> >*)arg[0]));
else   (((::std::vector<ClassA>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassA*,std::vector<ClassA> >*)arg[0]);
}

static void method_4251( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<ClassA*,std::vector<ClassA> >)((((::std::vector<ClassA>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassA*,std::vector<ClassA> >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<ClassA*,std::vector<ClassA> >*)arg[1]));
else   (((::std::vector<ClassA>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<ClassA*,std::vector<ClassA> >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<ClassA*,std::vector<ClassA> >*)arg[1]);
}

static void method_4252( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<ClassA>*)o)->swap)(*(::std::vector<ClassA>*)arg[0]);
}

static void method_4253( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<ClassA>*)o)->clear)();
}

static void constructor_x37( void* retaddr, void* mem, const std::vector<void*>&, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<ClassA>();
  else ::new(mem) ::std::vector<ClassA>();
}

static void method_newdel_1389( void* retaddr, void*, const std::vector<void*>&, void*)
{
  static ::Reflex::NewDelFunctions s_funcs;
  s_funcs.fNew         = ::Reflex::NewDelFunctionsT< ::std::vector<ClassA> >::new_T;
  s_funcs.fNewArray    = ::Reflex::NewDelFunctionsT< ::std::vector<ClassA> >::newArray_T;
  s_funcs.fDelete      = ::Reflex::NewDelFunctionsT< ::std::vector<ClassA> >::delete_T;
  s_funcs.fDeleteArray = ::Reflex::NewDelFunctionsT< ::std::vector<ClassA> >::deleteArray_T;
  s_funcs.fDestructor  = ::Reflex::NewDelFunctionsT< ::std::vector<ClassA> >::destruct_T;
  if (retaddr) *(::Reflex::NewDelFunctions**)retaddr = &s_funcs;
}

static void method_x39( void* retaddr, void*, const std::vector<void*>&, void*)
{
  typedef std::vector<std::pair< ::Reflex::Base, int> > Bases_t;
  static Bases_t s_bases;
  if ( !s_bases.size() ) {
    s_bases.push_back(std::make_pair(::Reflex::Base( ::Reflex::TypeBuilder("std::_Vector_base<ClassA,std::allocator<ClassA> >"), ::Reflex::BaseOffset< ::std::vector<ClassA>,::std::_Vector_base<ClassA,std::allocator<ClassA> > >::Get(),::Reflex::PROTECTED), 0));
  }
  if (retaddr) *(Bases_t**)retaddr = &s_bases;
}

static void method_x40( void* retaddr, void*, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**) retaddr = ::Reflex::Proxy< ::std::vector<ClassA> >::Generate();
  else ::Reflex::Proxy< ::std::vector<ClassA> >::Generate();
}

//------Dictionary for class vector<ClassA,std::allocator<ClassA> > -------------------------------
void __std__vector_ClassA__db_datamem(Reflex::Class*);
void __std__vector_ClassA__db_funcmem(Reflex::Class*);
Reflex::GenreflexMemberBuilder __std__vector_ClassA__datamem_bld(&__std__vector_ClassA__db_datamem);
Reflex::GenreflexMemberBuilder __std__vector_ClassA__funcmem_bld(&__std__vector_ClassA__db_funcmem);
void __std__vector_ClassA__dict() {
  ::Reflex::ClassBuilder("std::vector<ClassA>", typeid(::std::vector<ClassA>), sizeof(::std::vector<ClassA>), ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL, ::Reflex::CLASS)
  .AddBase(type_1648, ::Reflex::BaseOffset< ::std::vector<ClassA>, ::std::_Vector_base<ClassA,std::allocator<ClassA> > >::Get(), ::Reflex::PROTECTED)
  .AddTypedef(type_1648, "std::vector<ClassA>::_Base")
  .AddTypedef(type_994, "std::vector<ClassA>::value_type")
  .AddTypedef(type_4140, "std::vector<ClassA>::pointer")
  .AddTypedef(type_4202, "std::vector<ClassA>::const_pointer")
  .AddTypedef(type_4204, "std::vector<ClassA>::reference")
  .AddTypedef(type_4206, "std::vector<ClassA>::const_reference")
  .AddTypedef(type_2331, "std::vector<ClassA>::iterator")
  .AddTypedef(type_2332, "std::vector<ClassA>::const_iterator")
  .AddTypedef(type_1628, "std::vector<ClassA>::const_reverse_iterator")
  .AddTypedef(type_1629, "std::vector<ClassA>::reverse_iterator")
  .AddTypedef(type_854, "std::vector<ClassA>::size_type")
  .AddTypedef(type_684, "std::vector<ClassA>::difference_type")
  .AddTypedef(type_1539, "std::vector<ClassA>::allocator_type")
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7285), "vector", constructor_4215, 0, "__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854, type_4206, type_7285), "vector", constructor_4216, 0, "__n;__value;__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854), "vector", constructor_4217, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7286), "vector", constructor_4218, 0, "__x", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "~vector", destructor_4219, 0, 0, ::Reflex::PUBLIC | ::Reflex::DESTRUCTOR )
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "vector", constructor_x37, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember<void*(void)>("__getNewDelFunctions", method_newdel_1389, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("__getBasesTable", method_x39, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("createCollFuncTable", method_x40, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddOnDemandFunctionMemberBuilder(&__std__vector_ClassA__funcmem_bld);
}

//------Delayed data member builder for class vector<ClassA,std::allocator<ClassA> > -------------------
void __std__vector_ClassA__db_datamem(Reflex::Class*) {

}
//------Delayed function member builder for class vector<ClassA,std::allocator<ClassA> > -------------------
void __std__vector_ClassA__db_funcmem(Reflex::Class* cl) {
  ::Reflex::ClassBuilder(cl)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_7287, type_7286), "operator=", operator_4220, 0, "__x", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_4206), "assign", method_4221, 0, "__n;__val", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2331), "begin", method_4222, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2332), "begin", method_4223, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2331), "end", method_4224, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2332), "end", method_4225, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "size", method_4230, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "max_size", method_4231, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_4206), "resize", method_4232, 0, "__new_size;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "resize", method_4233, 0, "__new_size", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "capacity", method_4234, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2653), "empty", method_4235, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "reserve", method_4236, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4204, type_854), "operator[]", operator_4237, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4206, type_854), "operator[]", operator_4238, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4204, type_854), "at", method_4240, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4206, type_854), "at", method_4241, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4204), "front", method_4242, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4206), "front", method_4243, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4204), "back", method_4244, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4206), "back", method_4245, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_4206), "push_back", method_4246, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "pop_back", method_4247, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2331, type_2331, type_4206), "insert", method_4248, 0, "__position;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_2331, type_854, type_4206), "insert", method_4249, 0, "__position;__n;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2331, type_2331), "erase", method_4250, 0, "__position", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2331, type_2331, type_2331), "erase", method_4251, 0, "__first;__last", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_7287), "swap", method_4252, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "clear", method_4253, 0, 0, ::Reflex::PUBLIC);
}
//------Stub functions for class pair<int,double> -------------------------------
static void destructor_4578(void*, void * o, const std::vector<void*>&, void *) {
  (((::std::pair<int,double>*)o)->::std::pair<int,double>::~pair)();
}
static void operator_4579( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::pair<int,double>*)o)->operator=)(*(const ::std::pair<int,double>*)arg[0]);
else   (((::std::pair<int,double>*)o)->operator=)(*(const ::std::pair<int,double>*)arg[0]);
}

static void constructor_4580( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::pair<int,double>(*(const ::std::pair<int,double>*)arg[0]);
  else ::new(mem) ::std::pair<int,double>(*(const ::std::pair<int,double>*)arg[0]);
}

static void constructor_4581( void* retaddr, void* mem, const std::vector<void*>&, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::pair<int,double>();
  else ::new(mem) ::std::pair<int,double>();
}

static void constructor_4582( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::pair<int,double>(*(const int*)arg[0],
      *(const double*)arg[1]);
  else ::new(mem) ::std::pair<int,double>(*(const int*)arg[0],
      *(const double*)arg[1]);
}

static void method_newdel_1430( void* retaddr, void*, const std::vector<void*>&, void*)
{
  static ::Reflex::NewDelFunctions s_funcs;
  s_funcs.fNew         = ::Reflex::NewDelFunctionsT< ::std::pair<int,double> >::new_T;
  s_funcs.fNewArray    = ::Reflex::NewDelFunctionsT< ::std::pair<int,double> >::newArray_T;
  s_funcs.fDelete      = ::Reflex::NewDelFunctionsT< ::std::pair<int,double> >::delete_T;
  s_funcs.fDeleteArray = ::Reflex::NewDelFunctionsT< ::std::pair<int,double> >::deleteArray_T;
  s_funcs.fDestructor  = ::Reflex::NewDelFunctionsT< ::std::pair<int,double> >::destruct_T;
  if (retaddr) *(::Reflex::NewDelFunctions**)retaddr = &s_funcs;
}

//------Dictionary for class pair<int,double> -------------------------------
void __std__pair_int_double__db_datamem(Reflex::Class*);
void __std__pair_int_double__db_funcmem(Reflex::Class*);
Reflex::GenreflexMemberBuilder __std__pair_int_double__datamem_bld(&__std__pair_int_double__db_datamem);
Reflex::GenreflexMemberBuilder __std__pair_int_double__funcmem_bld(&__std__pair_int_double__db_funcmem);
void __std__pair_int_double__dict() {
  ::Reflex::ClassBuilder("std::pair<int,double>", typeid(::std::pair<int,double>), sizeof(::std::pair<int,double>), ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL, ::Reflex::STRUCT)
  .AddTypedef(type_36, "std::pair<int,double>::first_type")
  .AddTypedef(type_1818, "std::pair<int,double>::second_type")
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "~pair", destructor_4578, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::DESTRUCTOR )
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4263, type_4265), "operator=", operator_4579, 0, "", ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_4265), "pair", constructor_4580, 0, "", ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "pair", constructor_4581, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7343, type_4323), "pair", constructor_4582, 0, "__a;__b", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember<void*(void)>("__getNewDelFunctions", method_newdel_1430, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddOnDemandDataMemberBuilder(&__std__pair_int_double__datamem_bld);
}

//------Delayed data member builder for class pair<int,double> -------------------
void __std__pair_int_double__db_datamem(Reflex::Class* cl) {
  ::Reflex::ClassBuilder(cl)
  .AddDataMember(type_36, "first", OffsetOf(__shadow__::__std__pair_int_double_, first), ::Reflex::PUBLIC)
  .AddDataMember(type_1818, "second", OffsetOf(__shadow__::__std__pair_int_double_, second), ::Reflex::PUBLIC);
}
//------Delayed function member builder for class pair<int,double> -------------------
void __std__pair_int_double__db_funcmem(Reflex::Class*) {

}
//------Stub functions for class vector<std::pair<int, double>,std::allocator<std::pair<int, double> > > -------------------------------
static void constructor_4274( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 0 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<std::pair<int,double> >();
  else ::new(mem) ::std::vector<std::pair<int,double> >();
  }
  else if ( arg.size() == 1 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<std::pair<int,double> >(*(const ::std::allocator<std::pair<int,double> >*)arg[0]);
  else ::new(mem) ::std::vector<std::pair<int,double> >(*(const ::std::allocator<std::pair<int,double> >*)arg[0]);
  }
}

static void constructor_4275( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 2 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<std::pair<int,double> >(*(::size_t*)arg[0],
      *(const ::std::pair<int,double>*)arg[1]);
  else ::new(mem) ::std::vector<std::pair<int,double> >(*(::size_t*)arg[0],
      *(const ::std::pair<int,double>*)arg[1]);
  }
  else if ( arg.size() == 3 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<std::pair<int,double> >(*(::size_t*)arg[0],
      *(const ::std::pair<int,double>*)arg[1],
      *(const ::std::allocator<std::pair<int,double> >*)arg[2]);
  else ::new(mem) ::std::vector<std::pair<int,double> >(*(::size_t*)arg[0],
      *(const ::std::pair<int,double>*)arg[1],
      *(const ::std::allocator<std::pair<int,double> >*)arg[2]);
  }
}

static void constructor_4276( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<std::pair<int,double> >(*(::size_t*)arg[0]);
  else ::new(mem) ::std::vector<std::pair<int,double> >(*(::size_t*)arg[0]);
}

static void constructor_4277( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<std::pair<int,double> >(*(const ::std::vector<std::pair<int,double> >*)arg[0]);
  else ::new(mem) ::std::vector<std::pair<int,double> >(*(const ::std::vector<std::pair<int,double> >*)arg[0]);
}

static void destructor_4278(void*, void * o, const std::vector<void*>&, void *) {
  (((::std::vector<std::pair<int,double> >*)o)->::std::vector<std::pair<int,double> >::~vector)();
}
static void operator_4279( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<std::pair<int,double> >*)o)->operator=)(*(const ::std::vector<std::pair<int,double> >*)arg[0]);
else   (((::std::vector<std::pair<int,double> >*)o)->operator=)(*(const ::std::vector<std::pair<int,double> >*)arg[0]);
}

static void method_4280( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<std::pair<int,double> >*)o)->assign)(*(::size_t*)arg[0],
    *(const ::std::pair<int,double>*)arg[1]);
}

static void method_4281( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<std::pair<int,double>*,std::vector<std::pair<int,double> > >)((((::std::vector<std::pair<int,double> >*)o)->begin)());
else   (((::std::vector<std::pair<int,double> >*)o)->begin)();
}

static void method_4282( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<const std::pair<int,double>*,std::vector<std::pair<int,double> > >)((((const ::std::vector<std::pair<int,double> >*)o)->begin)());
else   (((const ::std::vector<std::pair<int,double> >*)o)->begin)();
}

static void method_4283( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<std::pair<int,double>*,std::vector<std::pair<int,double> > >)((((::std::vector<std::pair<int,double> >*)o)->end)());
else   (((::std::vector<std::pair<int,double> >*)o)->end)();
}

static void method_4284( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<const std::pair<int,double>*,std::vector<std::pair<int,double> > >)((((const ::std::vector<std::pair<int,double> >*)o)->end)());
else   (((const ::std::vector<std::pair<int,double> >*)o)->end)();
}

static void method_4289( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<std::pair<int,double> >*)o)->size)());
else   (((const ::std::vector<std::pair<int,double> >*)o)->size)();
}

static void method_4290( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<std::pair<int,double> >*)o)->max_size)());
else   (((const ::std::vector<std::pair<int,double> >*)o)->max_size)();
}

static void method_4291( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<std::pair<int,double> >*)o)->resize)(*(::size_t*)arg[0],
    *(const ::std::pair<int,double>*)arg[1]);
}

static void method_4292( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<std::pair<int,double> >*)o)->resize)(*(::size_t*)arg[0]);
}

static void method_4293( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<std::pair<int,double> >*)o)->capacity)());
else   (((const ::std::vector<std::pair<int,double> >*)o)->capacity)();
}

static void method_4294( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (bool)((((const ::std::vector<std::pair<int,double> >*)o)->empty)());
else   (((const ::std::vector<std::pair<int,double> >*)o)->empty)();
}

static void method_4295( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<std::pair<int,double> >*)o)->reserve)(*(::size_t*)arg[0]);
}

static void operator_4296( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<std::pair<int,double> >*)o)->operator[])(*(::size_t*)arg[0]);
else   (((::std::vector<std::pair<int,double> >*)o)->operator[])(*(::size_t*)arg[0]);
}

static void operator_4297( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<std::pair<int,double> >*)o)->operator[])(*(::size_t*)arg[0]);
else   (((const ::std::vector<std::pair<int,double> >*)o)->operator[])(*(::size_t*)arg[0]);
}

static void method_4299( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<std::pair<int,double> >*)o)->at)(*(::size_t*)arg[0]);
else   (((::std::vector<std::pair<int,double> >*)o)->at)(*(::size_t*)arg[0]);
}

static void method_4300( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<std::pair<int,double> >*)o)->at)(*(::size_t*)arg[0]);
else   (((const ::std::vector<std::pair<int,double> >*)o)->at)(*(::size_t*)arg[0]);
}

static void method_4301( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<std::pair<int,double> >*)o)->front)();
else   (((::std::vector<std::pair<int,double> >*)o)->front)();
}

static void method_4302( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<std::pair<int,double> >*)o)->front)();
else   (((const ::std::vector<std::pair<int,double> >*)o)->front)();
}

static void method_4303( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<std::pair<int,double> >*)o)->back)();
else   (((::std::vector<std::pair<int,double> >*)o)->back)();
}

static void method_4304( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<std::pair<int,double> >*)o)->back)();
else   (((const ::std::vector<std::pair<int,double> >*)o)->back)();
}

static void method_4305( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<std::pair<int,double> >*)o)->push_back)(*(const ::std::pair<int,double>*)arg[0]);
}

static void method_4306( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<std::pair<int,double> >*)o)->pop_back)();
}

static void method_4307( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<std::pair<int,double>*,std::vector<std::pair<int,double> > >)((((::std::vector<std::pair<int,double> >*)o)->insert)(*(::__gnu_cxx::__normal_iterator<std::pair<int,double>*,std::vector<std::pair<int,double> > >*)arg[0],
    *(const ::std::pair<int,double>*)arg[1]));
else   (((::std::vector<std::pair<int,double> >*)o)->insert)(*(::__gnu_cxx::__normal_iterator<std::pair<int,double>*,std::vector<std::pair<int,double> > >*)arg[0],
    *(const ::std::pair<int,double>*)arg[1]);
}

static void method_4308( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<std::pair<int,double> >*)o)->insert)(*(::__gnu_cxx::__normal_iterator<std::pair<int,double>*,std::vector<std::pair<int,double> > >*)arg[0],
    *(::size_t*)arg[1],
    *(const ::std::pair<int,double>*)arg[2]);
}

static void method_4309( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<std::pair<int,double>*,std::vector<std::pair<int,double> > >)((((::std::vector<std::pair<int,double> >*)o)->erase)(*(::__gnu_cxx::__normal_iterator<std::pair<int,double>*,std::vector<std::pair<int,double> > >*)arg[0]));
else   (((::std::vector<std::pair<int,double> >*)o)->erase)(*(::__gnu_cxx::__normal_iterator<std::pair<int,double>*,std::vector<std::pair<int,double> > >*)arg[0]);
}

static void method_4310( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<std::pair<int,double>*,std::vector<std::pair<int,double> > >)((((::std::vector<std::pair<int,double> >*)o)->erase)(*(::__gnu_cxx::__normal_iterator<std::pair<int,double>*,std::vector<std::pair<int,double> > >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<std::pair<int,double>*,std::vector<std::pair<int,double> > >*)arg[1]));
else   (((::std::vector<std::pair<int,double> >*)o)->erase)(*(::__gnu_cxx::__normal_iterator<std::pair<int,double>*,std::vector<std::pair<int,double> > >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<std::pair<int,double>*,std::vector<std::pair<int,double> > >*)arg[1]);
}

static void method_4311( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<std::pair<int,double> >*)o)->swap)(*(::std::vector<std::pair<int,double> >*)arg[0]);
}

static void method_4312( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<std::pair<int,double> >*)o)->clear)();
}

static void constructor_x42( void* retaddr, void* mem, const std::vector<void*>&, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<std::pair<int,double> >();
  else ::new(mem) ::std::vector<std::pair<int,double> >();
}

static void method_newdel_1390( void* retaddr, void*, const std::vector<void*>&, void*)
{
  static ::Reflex::NewDelFunctions s_funcs;
  s_funcs.fNew         = ::Reflex::NewDelFunctionsT< ::std::vector<std::pair<int,double> > >::new_T;
  s_funcs.fNewArray    = ::Reflex::NewDelFunctionsT< ::std::vector<std::pair<int,double> > >::newArray_T;
  s_funcs.fDelete      = ::Reflex::NewDelFunctionsT< ::std::vector<std::pair<int,double> > >::delete_T;
  s_funcs.fDeleteArray = ::Reflex::NewDelFunctionsT< ::std::vector<std::pair<int,double> > >::deleteArray_T;
  s_funcs.fDestructor  = ::Reflex::NewDelFunctionsT< ::std::vector<std::pair<int,double> > >::destruct_T;
  if (retaddr) *(::Reflex::NewDelFunctions**)retaddr = &s_funcs;
}

static void method_x44( void* retaddr, void*, const std::vector<void*>&, void*)
{
  typedef std::vector<std::pair< ::Reflex::Base, int> > Bases_t;
  static Bases_t s_bases;
  if ( !s_bases.size() ) {
    s_bases.push_back(std::make_pair(::Reflex::Base( ::Reflex::TypeBuilder("std::_Vector_base<std::pair<int,double>,std::allocator<std::pair<int,double> > >"), ::Reflex::BaseOffset< ::std::vector<std::pair<int,double> >,::std::_Vector_base<std::pair<int,double>,std::allocator<std::pair<int,double> > > >::Get(),::Reflex::PROTECTED), 0));
  }
  if (retaddr) *(Bases_t**)retaddr = &s_bases;
}

static void method_x45( void* retaddr, void*, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**) retaddr = ::Reflex::Proxy< ::std::vector<std::pair<int,double> > >::Generate();
  else ::Reflex::Proxy< ::std::vector<std::pair<int,double> > >::Generate();
}

//------Dictionary for class vector<std::pair<int, double>,std::allocator<std::pair<int, double> > > -------------------------------
void __std__vector_std__pair_int_double_s__db_datamem(Reflex::Class*);
void __std__vector_std__pair_int_double_s__db_funcmem(Reflex::Class*);
Reflex::GenreflexMemberBuilder __std__vector_std__pair_int_double_s__datamem_bld(&__std__vector_std__pair_int_double_s__db_datamem);
Reflex::GenreflexMemberBuilder __std__vector_std__pair_int_double_s__funcmem_bld(&__std__vector_std__pair_int_double_s__db_funcmem);
void __std__vector_std__pair_int_double_s__dict() {
  ::Reflex::ClassBuilder("std::vector<std::pair<int,double> >", typeid(::std::vector<std::pair<int,double> >), sizeof(::std::vector<std::pair<int,double> >), ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL, ::Reflex::CLASS)
  .AddBase(type_1644, ::Reflex::BaseOffset< ::std::vector<std::pair<int,double> >, ::std::_Vector_base<std::pair<int,double>,std::allocator<std::pair<int,double> > > >::Get(), ::Reflex::PROTECTED)
  .AddTypedef(type_1644, "std::vector<std::pair<int,double> >::_Base")
  .AddTypedef(type_1430, "std::vector<std::pair<int,double> >::value_type")
  .AddTypedef(type_4259, "std::vector<std::pair<int,double> >::pointer")
  .AddTypedef(type_4261, "std::vector<std::pair<int,double> >::const_pointer")
  .AddTypedef(type_4263, "std::vector<std::pair<int,double> >::reference")
  .AddTypedef(type_4265, "std::vector<std::pair<int,double> >::const_reference")
  .AddTypedef(type_2323, "std::vector<std::pair<int,double> >::iterator")
  .AddTypedef(type_2324, "std::vector<std::pair<int,double> >::const_iterator")
  .AddTypedef(type_1620, "std::vector<std::pair<int,double> >::const_reverse_iterator")
  .AddTypedef(type_1621, "std::vector<std::pair<int,double> >::reverse_iterator")
  .AddTypedef(type_854, "std::vector<std::pair<int,double> >::size_type")
  .AddTypedef(type_684, "std::vector<std::pair<int,double> >::difference_type")
  .AddTypedef(type_1540, "std::vector<std::pair<int,double> >::allocator_type")
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7289), "vector", constructor_4274, 0, "__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854, type_4265, type_7289), "vector", constructor_4275, 0, "__n;__value;__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854), "vector", constructor_4276, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7290), "vector", constructor_4277, 0, "__x", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "~vector", destructor_4278, 0, 0, ::Reflex::PUBLIC | ::Reflex::DESTRUCTOR )
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "vector", constructor_x42, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember<void*(void)>("__getNewDelFunctions", method_newdel_1390, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("__getBasesTable", method_x44, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("createCollFuncTable", method_x45, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddOnDemandFunctionMemberBuilder(&__std__vector_std__pair_int_double_s__funcmem_bld);
}

//------Delayed data member builder for class vector<std::pair<int, double>,std::allocator<std::pair<int, double> > > -------------------
void __std__vector_std__pair_int_double_s__db_datamem(Reflex::Class*) {

}
//------Delayed function member builder for class vector<std::pair<int, double>,std::allocator<std::pair<int, double> > > -------------------
void __std__vector_std__pair_int_double_s__db_funcmem(Reflex::Class* cl) {
  ::Reflex::ClassBuilder(cl)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_7291, type_7290), "operator=", operator_4279, 0, "__x", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_4265), "assign", method_4280, 0, "__n;__val", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2323), "begin", method_4281, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2324), "begin", method_4282, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2323), "end", method_4283, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2324), "end", method_4284, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "size", method_4289, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "max_size", method_4290, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_4265), "resize", method_4291, 0, "__new_size;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "resize", method_4292, 0, "__new_size", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "capacity", method_4293, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2653), "empty", method_4294, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "reserve", method_4295, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4263, type_854), "operator[]", operator_4296, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4265, type_854), "operator[]", operator_4297, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4263, type_854), "at", method_4299, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4265, type_854), "at", method_4300, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4263), "front", method_4301, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4265), "front", method_4302, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4263), "back", method_4303, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4265), "back", method_4304, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_4265), "push_back", method_4305, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "pop_back", method_4306, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2323, type_2323, type_4265), "insert", method_4307, 0, "__position;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_2323, type_854, type_4265), "insert", method_4308, 0, "__position;__n;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2323, type_2323), "erase", method_4309, 0, "__position", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2323, type_2323, type_2323), "erase", method_4310, 0, "__first;__last", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_7291), "swap", method_4311, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "clear", method_4312, 0, 0, ::Reflex::PUBLIC);
}
//------Stub functions for class vector<double,std::allocator<double> > -------------------------------
static void constructor_4332( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 0 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<double>();
  else ::new(mem) ::std::vector<double>();
  }
  else if ( arg.size() == 1 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<double>(*(const ::std::allocator<double>*)arg[0]);
  else ::new(mem) ::std::vector<double>(*(const ::std::allocator<double>*)arg[0]);
  }
}

static void constructor_4333( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 2 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<double>(*(::size_t*)arg[0],
      *(const double*)arg[1]);
  else ::new(mem) ::std::vector<double>(*(::size_t*)arg[0],
      *(const double*)arg[1]);
  }
  else if ( arg.size() == 3 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<double>(*(::size_t*)arg[0],
      *(const double*)arg[1],
      *(const ::std::allocator<double>*)arg[2]);
  else ::new(mem) ::std::vector<double>(*(::size_t*)arg[0],
      *(const double*)arg[1],
      *(const ::std::allocator<double>*)arg[2]);
  }
}

static void constructor_4334( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<double>(*(::size_t*)arg[0]);
  else ::new(mem) ::std::vector<double>(*(::size_t*)arg[0]);
}

static void constructor_4335( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<double>(*(const ::std::vector<double>*)arg[0]);
  else ::new(mem) ::std::vector<double>(*(const ::std::vector<double>*)arg[0]);
}

static void destructor_4336(void*, void * o, const std::vector<void*>&, void *) {
  (((::std::vector<double>*)o)->::std::vector<double>::~vector)();
}
static void operator_4337( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<double>*)o)->operator=)(*(const ::std::vector<double>*)arg[0]);
else   (((::std::vector<double>*)o)->operator=)(*(const ::std::vector<double>*)arg[0]);
}

static void method_4338( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<double>*)o)->assign)(*(::size_t*)arg[0],
    *(const double*)arg[1]);
}

static void method_4339( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<double*,std::vector<double> >)((((::std::vector<double>*)o)->begin)());
else   (((::std::vector<double>*)o)->begin)();
}

static void method_4340( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<const double*,std::vector<double> >)((((const ::std::vector<double>*)o)->begin)());
else   (((const ::std::vector<double>*)o)->begin)();
}

static void method_4341( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<double*,std::vector<double> >)((((::std::vector<double>*)o)->end)());
else   (((::std::vector<double>*)o)->end)();
}

static void method_4342( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<const double*,std::vector<double> >)((((const ::std::vector<double>*)o)->end)());
else   (((const ::std::vector<double>*)o)->end)();
}

static void method_4347( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<double>*)o)->size)());
else   (((const ::std::vector<double>*)o)->size)();
}

static void method_4348( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<double>*)o)->max_size)());
else   (((const ::std::vector<double>*)o)->max_size)();
}

static void method_4349( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<double>*)o)->resize)(*(::size_t*)arg[0],
    *(const double*)arg[1]);
}

static void method_4350( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<double>*)o)->resize)(*(::size_t*)arg[0]);
}

static void method_4351( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<double>*)o)->capacity)());
else   (((const ::std::vector<double>*)o)->capacity)();
}

static void method_4352( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (bool)((((const ::std::vector<double>*)o)->empty)());
else   (((const ::std::vector<double>*)o)->empty)();
}

static void method_4353( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<double>*)o)->reserve)(*(::size_t*)arg[0]);
}

static void operator_4354( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<double>*)o)->operator[])(*(::size_t*)arg[0]);
else   (((::std::vector<double>*)o)->operator[])(*(::size_t*)arg[0]);
}

static void operator_4355( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<double>*)o)->operator[])(*(::size_t*)arg[0]);
else   (((const ::std::vector<double>*)o)->operator[])(*(::size_t*)arg[0]);
}

static void method_4357( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<double>*)o)->at)(*(::size_t*)arg[0]);
else   (((::std::vector<double>*)o)->at)(*(::size_t*)arg[0]);
}

static void method_4358( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<double>*)o)->at)(*(::size_t*)arg[0]);
else   (((const ::std::vector<double>*)o)->at)(*(::size_t*)arg[0]);
}

static void method_4359( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<double>*)o)->front)();
else   (((::std::vector<double>*)o)->front)();
}

static void method_4360( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<double>*)o)->front)();
else   (((const ::std::vector<double>*)o)->front)();
}

static void method_4361( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<double>*)o)->back)();
else   (((::std::vector<double>*)o)->back)();
}

static void method_4362( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<double>*)o)->back)();
else   (((const ::std::vector<double>*)o)->back)();
}

static void method_4363( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<double>*)o)->push_back)(*(const double*)arg[0]);
}

static void method_4364( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<double>*)o)->pop_back)();
}

static void method_4365( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<double*,std::vector<double> >)((((::std::vector<double>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<double*,std::vector<double> >*)arg[0],
    *(const double*)arg[1]));
else   (((::std::vector<double>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<double*,std::vector<double> >*)arg[0],
    *(const double*)arg[1]);
}

static void method_4366( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<double>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<double*,std::vector<double> >*)arg[0],
    *(::size_t*)arg[1],
    *(const double*)arg[2]);
}

static void method_4367( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<double*,std::vector<double> >)((((::std::vector<double>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<double*,std::vector<double> >*)arg[0]));
else   (((::std::vector<double>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<double*,std::vector<double> >*)arg[0]);
}

static void method_4368( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<double*,std::vector<double> >)((((::std::vector<double>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<double*,std::vector<double> >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<double*,std::vector<double> >*)arg[1]));
else   (((::std::vector<double>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<double*,std::vector<double> >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<double*,std::vector<double> >*)arg[1]);
}

static void method_4369( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<double>*)o)->swap)(*(::std::vector<double>*)arg[0]);
}

static void method_4370( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<double>*)o)->clear)();
}

static void constructor_x46( void* retaddr, void* mem, const std::vector<void*>&, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<double>();
  else ::new(mem) ::std::vector<double>();
}

static void method_newdel_1391( void* retaddr, void*, const std::vector<void*>&, void*)
{
  static ::Reflex::NewDelFunctions s_funcs;
  s_funcs.fNew         = ::Reflex::NewDelFunctionsT< ::std::vector<double> >::new_T;
  s_funcs.fNewArray    = ::Reflex::NewDelFunctionsT< ::std::vector<double> >::newArray_T;
  s_funcs.fDelete      = ::Reflex::NewDelFunctionsT< ::std::vector<double> >::delete_T;
  s_funcs.fDeleteArray = ::Reflex::NewDelFunctionsT< ::std::vector<double> >::deleteArray_T;
  s_funcs.fDestructor  = ::Reflex::NewDelFunctionsT< ::std::vector<double> >::destruct_T;
  if (retaddr) *(::Reflex::NewDelFunctions**)retaddr = &s_funcs;
}

static void method_x48( void* retaddr, void*, const std::vector<void*>&, void*)
{
  typedef std::vector<std::pair< ::Reflex::Base, int> > Bases_t;
  static Bases_t s_bases;
  if ( !s_bases.size() ) {
    s_bases.push_back(std::make_pair(::Reflex::Base( ::Reflex::TypeBuilder("std::_Vector_base<double,std::allocator<double> >"), ::Reflex::BaseOffset< ::std::vector<double>,::std::_Vector_base<double,std::allocator<double> > >::Get(),::Reflex::PROTECTED), 0));
  }
  if (retaddr) *(Bases_t**)retaddr = &s_bases;
}

static void method_x49( void* retaddr, void*, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**) retaddr = ::Reflex::Proxy< ::std::vector<double> >::Generate();
  else ::Reflex::Proxy< ::std::vector<double> >::Generate();
}

//------Dictionary for class vector<double,std::allocator<double> > -------------------------------
void __std__vector_double__db_datamem(Reflex::Class*);
void __std__vector_double__db_funcmem(Reflex::Class*);
Reflex::GenreflexMemberBuilder __std__vector_double__datamem_bld(&__std__vector_double__db_datamem);
Reflex::GenreflexMemberBuilder __std__vector_double__funcmem_bld(&__std__vector_double__db_funcmem);
void __std__vector_double__dict() {
  ::Reflex::ClassBuilder("std::vector<double>", typeid(::std::vector<double>), sizeof(::std::vector<double>), ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL, ::Reflex::CLASS)
  .AddBase(type_1645, ::Reflex::BaseOffset< ::std::vector<double>, ::std::_Vector_base<double,std::allocator<double> > >::Get(), ::Reflex::PROTECTED)
  .AddTypedef(type_1645, "std::vector<double>::_Base")
  .AddTypedef(type_1818, "std::vector<double>::value_type")
  .AddTypedef(type_2215, "std::vector<double>::pointer")
  .AddTypedef(type_4319, "std::vector<double>::const_pointer")
  .AddTypedef(type_4321, "std::vector<double>::reference")
  .AddTypedef(type_4323, "std::vector<double>::const_reference")
  .AddTypedef(type_2325, "std::vector<double>::iterator")
  .AddTypedef(type_2326, "std::vector<double>::const_iterator")
  .AddTypedef(type_1622, "std::vector<double>::const_reverse_iterator")
  .AddTypedef(type_1623, "std::vector<double>::reverse_iterator")
  .AddTypedef(type_854, "std::vector<double>::size_type")
  .AddTypedef(type_684, "std::vector<double>::difference_type")
  .AddTypedef(type_1541, "std::vector<double>::allocator_type")
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7293), "vector", constructor_4332, 0, "__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854, type_4323, type_7293), "vector", constructor_4333, 0, "__n;__value;__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854), "vector", constructor_4334, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7294), "vector", constructor_4335, 0, "__x", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "~vector", destructor_4336, 0, 0, ::Reflex::PUBLIC | ::Reflex::DESTRUCTOR )
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "vector", constructor_x46, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember<void*(void)>("__getNewDelFunctions", method_newdel_1391, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("__getBasesTable", method_x48, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("createCollFuncTable", method_x49, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddOnDemandFunctionMemberBuilder(&__std__vector_double__funcmem_bld);
}

//------Delayed data member builder for class vector<double,std::allocator<double> > -------------------
void __std__vector_double__db_datamem(Reflex::Class*) {

}
//------Delayed function member builder for class vector<double,std::allocator<double> > -------------------
void __std__vector_double__db_funcmem(Reflex::Class* cl) {
  ::Reflex::ClassBuilder(cl)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_7295, type_7294), "operator=", operator_4337, 0, "__x", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_4323), "assign", method_4338, 0, "__n;__val", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2325), "begin", method_4339, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2326), "begin", method_4340, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2325), "end", method_4341, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2326), "end", method_4342, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "size", method_4347, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "max_size", method_4348, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_4323), "resize", method_4349, 0, "__new_size;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "resize", method_4350, 0, "__new_size", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "capacity", method_4351, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2653), "empty", method_4352, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "reserve", method_4353, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4321, type_854), "operator[]", operator_4354, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4323, type_854), "operator[]", operator_4355, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4321, type_854), "at", method_4357, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4323, type_854), "at", method_4358, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4321), "front", method_4359, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4323), "front", method_4360, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4321), "back", method_4361, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4323), "back", method_4362, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_4323), "push_back", method_4363, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "pop_back", method_4364, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2325, type_2325, type_4323), "insert", method_4365, 0, "__position;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_2325, type_854, type_4323), "insert", method_4366, 0, "__position;__n;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2325, type_2325), "erase", method_4367, 0, "__position", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2325, type_2325, type_2325), "erase", method_4368, 0, "__first;__last", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_7295), "swap", method_4369, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "clear", method_4370, 0, 0, ::Reflex::PUBLIC);
}
//------Stub functions for class pair<int,float> -------------------------------
static void destructor_4587(void*, void * o, const std::vector<void*>&, void *) {
  (((::std::pair<int,float>*)o)->::std::pair<int,float>::~pair)();
}
static void operator_4588( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::pair<int,float>*)o)->operator=)(*(const ::std::pair<int,float>*)arg[0]);
else   (((::std::pair<int,float>*)o)->operator=)(*(const ::std::pair<int,float>*)arg[0]);
}

static void constructor_4589( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::pair<int,float>(*(const ::std::pair<int,float>*)arg[0]);
  else ::new(mem) ::std::pair<int,float>(*(const ::std::pair<int,float>*)arg[0]);
}

static void constructor_4590( void* retaddr, void* mem, const std::vector<void*>&, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::pair<int,float>();
  else ::new(mem) ::std::pair<int,float>();
}

static void constructor_4591( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::pair<int,float>(*(const int*)arg[0],
      *(const float*)arg[1]);
  else ::new(mem) ::std::pair<int,float>(*(const int*)arg[0],
      *(const float*)arg[1]);
}

static void method_newdel_1431( void* retaddr, void*, const std::vector<void*>&, void*)
{
  static ::Reflex::NewDelFunctions s_funcs;
  s_funcs.fNew         = ::Reflex::NewDelFunctionsT< ::std::pair<int,float> >::new_T;
  s_funcs.fNewArray    = ::Reflex::NewDelFunctionsT< ::std::pair<int,float> >::newArray_T;
  s_funcs.fDelete      = ::Reflex::NewDelFunctionsT< ::std::pair<int,float> >::delete_T;
  s_funcs.fDeleteArray = ::Reflex::NewDelFunctionsT< ::std::pair<int,float> >::deleteArray_T;
  s_funcs.fDestructor  = ::Reflex::NewDelFunctionsT< ::std::pair<int,float> >::destruct_T;
  if (retaddr) *(::Reflex::NewDelFunctions**)retaddr = &s_funcs;
}

//------Dictionary for class pair<int,float> -------------------------------
void __std__pair_int_float__db_datamem(Reflex::Class*);
void __std__pair_int_float__db_funcmem(Reflex::Class*);
Reflex::GenreflexMemberBuilder __std__pair_int_float__datamem_bld(&__std__pair_int_float__db_datamem);
Reflex::GenreflexMemberBuilder __std__pair_int_float__funcmem_bld(&__std__pair_int_float__db_funcmem);
void __std__pair_int_float__dict() {
  ::Reflex::ClassBuilder("std::pair<int,float>", typeid(::std::pair<int,float>), sizeof(::std::pair<int,float>), ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL, ::Reflex::STRUCT)
  .AddTypedef(type_36, "std::pair<int,float>::first_type")
  .AddTypedef(type_1823, "std::pair<int,float>::second_type")
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "~pair", destructor_4587, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::DESTRUCTOR )
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4380, type_4382), "operator=", operator_4588, 0, "", ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_4382), "pair", constructor_4589, 0, "", ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "pair", constructor_4590, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7343, type_4440), "pair", constructor_4591, 0, "__a;__b", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember<void*(void)>("__getNewDelFunctions", method_newdel_1431, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddOnDemandDataMemberBuilder(&__std__pair_int_float__datamem_bld);
}

//------Delayed data member builder for class pair<int,float> -------------------
void __std__pair_int_float__db_datamem(Reflex::Class* cl) {
  ::Reflex::ClassBuilder(cl)
  .AddDataMember(type_36, "first", OffsetOf(__shadow__::__std__pair_int_float_, first), ::Reflex::PUBLIC)
  .AddDataMember(type_1823, "second", OffsetOf(__shadow__::__std__pair_int_float_, second), ::Reflex::PUBLIC);
}
//------Delayed function member builder for class pair<int,float> -------------------
void __std__pair_int_float__db_funcmem(Reflex::Class*) {

}
//------Stub functions for class vector<std::pair<int, float>,std::allocator<std::pair<int, float> > > -------------------------------
static void constructor_4391( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 0 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<std::pair<int,float> >();
  else ::new(mem) ::std::vector<std::pair<int,float> >();
  }
  else if ( arg.size() == 1 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<std::pair<int,float> >(*(const ::std::allocator<std::pair<int,float> >*)arg[0]);
  else ::new(mem) ::std::vector<std::pair<int,float> >(*(const ::std::allocator<std::pair<int,float> >*)arg[0]);
  }
}

static void constructor_4392( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 2 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<std::pair<int,float> >(*(::size_t*)arg[0],
      *(const ::std::pair<int,float>*)arg[1]);
  else ::new(mem) ::std::vector<std::pair<int,float> >(*(::size_t*)arg[0],
      *(const ::std::pair<int,float>*)arg[1]);
  }
  else if ( arg.size() == 3 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<std::pair<int,float> >(*(::size_t*)arg[0],
      *(const ::std::pair<int,float>*)arg[1],
      *(const ::std::allocator<std::pair<int,float> >*)arg[2]);
  else ::new(mem) ::std::vector<std::pair<int,float> >(*(::size_t*)arg[0],
      *(const ::std::pair<int,float>*)arg[1],
      *(const ::std::allocator<std::pair<int,float> >*)arg[2]);
  }
}

static void constructor_4393( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<std::pair<int,float> >(*(::size_t*)arg[0]);
  else ::new(mem) ::std::vector<std::pair<int,float> >(*(::size_t*)arg[0]);
}

static void constructor_4394( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<std::pair<int,float> >(*(const ::std::vector<std::pair<int,float> >*)arg[0]);
  else ::new(mem) ::std::vector<std::pair<int,float> >(*(const ::std::vector<std::pair<int,float> >*)arg[0]);
}

static void destructor_4395(void*, void * o, const std::vector<void*>&, void *) {
  (((::std::vector<std::pair<int,float> >*)o)->::std::vector<std::pair<int,float> >::~vector)();
}
static void operator_4396( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<std::pair<int,float> >*)o)->operator=)(*(const ::std::vector<std::pair<int,float> >*)arg[0]);
else   (((::std::vector<std::pair<int,float> >*)o)->operator=)(*(const ::std::vector<std::pair<int,float> >*)arg[0]);
}

static void method_4397( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<std::pair<int,float> >*)o)->assign)(*(::size_t*)arg[0],
    *(const ::std::pair<int,float>*)arg[1]);
}

static void method_4398( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<std::pair<int,float>*,std::vector<std::pair<int,float> > >)((((::std::vector<std::pair<int,float> >*)o)->begin)());
else   (((::std::vector<std::pair<int,float> >*)o)->begin)();
}

static void method_4399( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<const std::pair<int,float>*,std::vector<std::pair<int,float> > >)((((const ::std::vector<std::pair<int,float> >*)o)->begin)());
else   (((const ::std::vector<std::pair<int,float> >*)o)->begin)();
}

static void method_4400( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<std::pair<int,float>*,std::vector<std::pair<int,float> > >)((((::std::vector<std::pair<int,float> >*)o)->end)());
else   (((::std::vector<std::pair<int,float> >*)o)->end)();
}

static void method_4401( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<const std::pair<int,float>*,std::vector<std::pair<int,float> > >)((((const ::std::vector<std::pair<int,float> >*)o)->end)());
else   (((const ::std::vector<std::pair<int,float> >*)o)->end)();
}

static void method_4406( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<std::pair<int,float> >*)o)->size)());
else   (((const ::std::vector<std::pair<int,float> >*)o)->size)();
}

static void method_4407( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<std::pair<int,float> >*)o)->max_size)());
else   (((const ::std::vector<std::pair<int,float> >*)o)->max_size)();
}

static void method_4408( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<std::pair<int,float> >*)o)->resize)(*(::size_t*)arg[0],
    *(const ::std::pair<int,float>*)arg[1]);
}

static void method_4409( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<std::pair<int,float> >*)o)->resize)(*(::size_t*)arg[0]);
}

static void method_4410( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<std::pair<int,float> >*)o)->capacity)());
else   (((const ::std::vector<std::pair<int,float> >*)o)->capacity)();
}

static void method_4411( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (bool)((((const ::std::vector<std::pair<int,float> >*)o)->empty)());
else   (((const ::std::vector<std::pair<int,float> >*)o)->empty)();
}

static void method_4412( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<std::pair<int,float> >*)o)->reserve)(*(::size_t*)arg[0]);
}

static void operator_4413( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<std::pair<int,float> >*)o)->operator[])(*(::size_t*)arg[0]);
else   (((::std::vector<std::pair<int,float> >*)o)->operator[])(*(::size_t*)arg[0]);
}

static void operator_4414( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<std::pair<int,float> >*)o)->operator[])(*(::size_t*)arg[0]);
else   (((const ::std::vector<std::pair<int,float> >*)o)->operator[])(*(::size_t*)arg[0]);
}

static void method_4416( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<std::pair<int,float> >*)o)->at)(*(::size_t*)arg[0]);
else   (((::std::vector<std::pair<int,float> >*)o)->at)(*(::size_t*)arg[0]);
}

static void method_4417( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<std::pair<int,float> >*)o)->at)(*(::size_t*)arg[0]);
else   (((const ::std::vector<std::pair<int,float> >*)o)->at)(*(::size_t*)arg[0]);
}

static void method_4418( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<std::pair<int,float> >*)o)->front)();
else   (((::std::vector<std::pair<int,float> >*)o)->front)();
}

static void method_4419( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<std::pair<int,float> >*)o)->front)();
else   (((const ::std::vector<std::pair<int,float> >*)o)->front)();
}

static void method_4420( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<std::pair<int,float> >*)o)->back)();
else   (((::std::vector<std::pair<int,float> >*)o)->back)();
}

static void method_4421( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<std::pair<int,float> >*)o)->back)();
else   (((const ::std::vector<std::pair<int,float> >*)o)->back)();
}

static void method_4422( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<std::pair<int,float> >*)o)->push_back)(*(const ::std::pair<int,float>*)arg[0]);
}

static void method_4423( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<std::pair<int,float> >*)o)->pop_back)();
}

static void method_4424( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<std::pair<int,float>*,std::vector<std::pair<int,float> > >)((((::std::vector<std::pair<int,float> >*)o)->insert)(*(::__gnu_cxx::__normal_iterator<std::pair<int,float>*,std::vector<std::pair<int,float> > >*)arg[0],
    *(const ::std::pair<int,float>*)arg[1]));
else   (((::std::vector<std::pair<int,float> >*)o)->insert)(*(::__gnu_cxx::__normal_iterator<std::pair<int,float>*,std::vector<std::pair<int,float> > >*)arg[0],
    *(const ::std::pair<int,float>*)arg[1]);
}

static void method_4425( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<std::pair<int,float> >*)o)->insert)(*(::__gnu_cxx::__normal_iterator<std::pair<int,float>*,std::vector<std::pair<int,float> > >*)arg[0],
    *(::size_t*)arg[1],
    *(const ::std::pair<int,float>*)arg[2]);
}

static void method_4426( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<std::pair<int,float>*,std::vector<std::pair<int,float> > >)((((::std::vector<std::pair<int,float> >*)o)->erase)(*(::__gnu_cxx::__normal_iterator<std::pair<int,float>*,std::vector<std::pair<int,float> > >*)arg[0]));
else   (((::std::vector<std::pair<int,float> >*)o)->erase)(*(::__gnu_cxx::__normal_iterator<std::pair<int,float>*,std::vector<std::pair<int,float> > >*)arg[0]);
}

static void method_4427( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<std::pair<int,float>*,std::vector<std::pair<int,float> > >)((((::std::vector<std::pair<int,float> >*)o)->erase)(*(::__gnu_cxx::__normal_iterator<std::pair<int,float>*,std::vector<std::pair<int,float> > >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<std::pair<int,float>*,std::vector<std::pair<int,float> > >*)arg[1]));
else   (((::std::vector<std::pair<int,float> >*)o)->erase)(*(::__gnu_cxx::__normal_iterator<std::pair<int,float>*,std::vector<std::pair<int,float> > >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<std::pair<int,float>*,std::vector<std::pair<int,float> > >*)arg[1]);
}

static void method_4428( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<std::pair<int,float> >*)o)->swap)(*(::std::vector<std::pair<int,float> >*)arg[0]);
}

static void method_4429( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<std::pair<int,float> >*)o)->clear)();
}

static void constructor_x51( void* retaddr, void* mem, const std::vector<void*>&, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<std::pair<int,float> >();
  else ::new(mem) ::std::vector<std::pair<int,float> >();
}

static void method_newdel_1392( void* retaddr, void*, const std::vector<void*>&, void*)
{
  static ::Reflex::NewDelFunctions s_funcs;
  s_funcs.fNew         = ::Reflex::NewDelFunctionsT< ::std::vector<std::pair<int,float> > >::new_T;
  s_funcs.fNewArray    = ::Reflex::NewDelFunctionsT< ::std::vector<std::pair<int,float> > >::newArray_T;
  s_funcs.fDelete      = ::Reflex::NewDelFunctionsT< ::std::vector<std::pair<int,float> > >::delete_T;
  s_funcs.fDeleteArray = ::Reflex::NewDelFunctionsT< ::std::vector<std::pair<int,float> > >::deleteArray_T;
  s_funcs.fDestructor  = ::Reflex::NewDelFunctionsT< ::std::vector<std::pair<int,float> > >::destruct_T;
  if (retaddr) *(::Reflex::NewDelFunctions**)retaddr = &s_funcs;
}

static void method_x53( void* retaddr, void*, const std::vector<void*>&, void*)
{
  typedef std::vector<std::pair< ::Reflex::Base, int> > Bases_t;
  static Bases_t s_bases;
  if ( !s_bases.size() ) {
    s_bases.push_back(std::make_pair(::Reflex::Base( ::Reflex::TypeBuilder("std::_Vector_base<std::pair<int,float>,std::allocator<std::pair<int,float> > >"), ::Reflex::BaseOffset< ::std::vector<std::pair<int,float> >,::std::_Vector_base<std::pair<int,float>,std::allocator<std::pair<int,float> > > >::Get(),::Reflex::PROTECTED), 0));
  }
  if (retaddr) *(Bases_t**)retaddr = &s_bases;
}

static void method_x54( void* retaddr, void*, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**) retaddr = ::Reflex::Proxy< ::std::vector<std::pair<int,float> > >::Generate();
  else ::Reflex::Proxy< ::std::vector<std::pair<int,float> > >::Generate();
}

//------Dictionary for class vector<std::pair<int, float>,std::allocator<std::pair<int, float> > > -------------------------------
void __std__vector_std__pair_int_float_s__db_datamem(Reflex::Class*);
void __std__vector_std__pair_int_float_s__db_funcmem(Reflex::Class*);
Reflex::GenreflexMemberBuilder __std__vector_std__pair_int_float_s__datamem_bld(&__std__vector_std__pair_int_float_s__db_datamem);
Reflex::GenreflexMemberBuilder __std__vector_std__pair_int_float_s__funcmem_bld(&__std__vector_std__pair_int_float_s__db_funcmem);
void __std__vector_std__pair_int_float_s__dict() {
  ::Reflex::ClassBuilder("std::vector<std::pair<int,float> >", typeid(::std::vector<std::pair<int,float> >), sizeof(::std::vector<std::pair<int,float> >), ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL, ::Reflex::CLASS)
  .AddBase(type_1647, ::Reflex::BaseOffset< ::std::vector<std::pair<int,float> >, ::std::_Vector_base<std::pair<int,float>,std::allocator<std::pair<int,float> > > >::Get(), ::Reflex::PROTECTED)
  .AddTypedef(type_1647, "std::vector<std::pair<int,float> >::_Base")
  .AddTypedef(type_1431, "std::vector<std::pair<int,float> >::value_type")
  .AddTypedef(type_4376, "std::vector<std::pair<int,float> >::pointer")
  .AddTypedef(type_4378, "std::vector<std::pair<int,float> >::const_pointer")
  .AddTypedef(type_4380, "std::vector<std::pair<int,float> >::reference")
  .AddTypedef(type_4382, "std::vector<std::pair<int,float> >::const_reference")
  .AddTypedef(type_2329, "std::vector<std::pair<int,float> >::iterator")
  .AddTypedef(type_2330, "std::vector<std::pair<int,float> >::const_iterator")
  .AddTypedef(type_1626, "std::vector<std::pair<int,float> >::const_reverse_iterator")
  .AddTypedef(type_1627, "std::vector<std::pair<int,float> >::reverse_iterator")
  .AddTypedef(type_854, "std::vector<std::pair<int,float> >::size_type")
  .AddTypedef(type_684, "std::vector<std::pair<int,float> >::difference_type")
  .AddTypedef(type_1542, "std::vector<std::pair<int,float> >::allocator_type")
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7297), "vector", constructor_4391, 0, "__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854, type_4382, type_7297), "vector", constructor_4392, 0, "__n;__value;__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854), "vector", constructor_4393, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7298), "vector", constructor_4394, 0, "__x", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "~vector", destructor_4395, 0, 0, ::Reflex::PUBLIC | ::Reflex::DESTRUCTOR )
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "vector", constructor_x51, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember<void*(void)>("__getNewDelFunctions", method_newdel_1392, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("__getBasesTable", method_x53, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("createCollFuncTable", method_x54, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddOnDemandFunctionMemberBuilder(&__std__vector_std__pair_int_float_s__funcmem_bld);
}

//------Delayed data member builder for class vector<std::pair<int, float>,std::allocator<std::pair<int, float> > > -------------------
void __std__vector_std__pair_int_float_s__db_datamem(Reflex::Class*) {

}
//------Delayed function member builder for class vector<std::pair<int, float>,std::allocator<std::pair<int, float> > > -------------------
void __std__vector_std__pair_int_float_s__db_funcmem(Reflex::Class* cl) {
  ::Reflex::ClassBuilder(cl)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_7299, type_7298), "operator=", operator_4396, 0, "__x", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_4382), "assign", method_4397, 0, "__n;__val", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2329), "begin", method_4398, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2330), "begin", method_4399, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2329), "end", method_4400, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2330), "end", method_4401, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "size", method_4406, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "max_size", method_4407, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_4382), "resize", method_4408, 0, "__new_size;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "resize", method_4409, 0, "__new_size", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "capacity", method_4410, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2653), "empty", method_4411, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "reserve", method_4412, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4380, type_854), "operator[]", operator_4413, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4382, type_854), "operator[]", operator_4414, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4380, type_854), "at", method_4416, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4382, type_854), "at", method_4417, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4380), "front", method_4418, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4382), "front", method_4419, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4380), "back", method_4420, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4382), "back", method_4421, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_4382), "push_back", method_4422, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "pop_back", method_4423, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2329, type_2329, type_4382), "insert", method_4424, 0, "__position;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_2329, type_854, type_4382), "insert", method_4425, 0, "__position;__n;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2329, type_2329), "erase", method_4426, 0, "__position", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2329, type_2329, type_2329), "erase", method_4427, 0, "__first;__last", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_7299), "swap", method_4428, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "clear", method_4429, 0, 0, ::Reflex::PUBLIC);
}
//------Stub functions for class vector<float,std::allocator<float> > -------------------------------
static void constructor_4449( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 0 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<float>();
  else ::new(mem) ::std::vector<float>();
  }
  else if ( arg.size() == 1 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<float>(*(const ::std::allocator<float>*)arg[0]);
  else ::new(mem) ::std::vector<float>(*(const ::std::allocator<float>*)arg[0]);
  }
}

static void constructor_4450( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if ( arg.size() == 2 ) {
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<float>(*(::size_t*)arg[0],
      *(const float*)arg[1]);
  else ::new(mem) ::std::vector<float>(*(::size_t*)arg[0],
      *(const float*)arg[1]);
  }
  else if ( arg.size() == 3 ) { 
    if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<float>(*(::size_t*)arg[0],
      *(const float*)arg[1],
      *(const ::std::allocator<float>*)arg[2]);
  else ::new(mem) ::std::vector<float>(*(::size_t*)arg[0],
      *(const float*)arg[1],
      *(const ::std::allocator<float>*)arg[2]);
  }
}

static void constructor_4451( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<float>(*(::size_t*)arg[0]);
  else ::new(mem) ::std::vector<float>(*(::size_t*)arg[0]);
}

static void constructor_4452( void* retaddr, void* mem, const std::vector<void*>& arg, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<float>(*(const ::std::vector<float>*)arg[0]);
  else ::new(mem) ::std::vector<float>(*(const ::std::vector<float>*)arg[0]);
}

static void destructor_4453(void*, void * o, const std::vector<void*>&, void *) {
  (((::std::vector<float>*)o)->::std::vector<float>::~vector)();
}
static void operator_4454( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<float>*)o)->operator=)(*(const ::std::vector<float>*)arg[0]);
else   (((::std::vector<float>*)o)->operator=)(*(const ::std::vector<float>*)arg[0]);
}

static void method_4455( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<float>*)o)->assign)(*(::size_t*)arg[0],
    *(const float*)arg[1]);
}

static void method_4456( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<float*,std::vector<float> >)((((::std::vector<float>*)o)->begin)());
else   (((::std::vector<float>*)o)->begin)();
}

static void method_4457( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<const float*,std::vector<float> >)((((const ::std::vector<float>*)o)->begin)());
else   (((const ::std::vector<float>*)o)->begin)();
}

static void method_4458( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<float*,std::vector<float> >)((((::std::vector<float>*)o)->end)());
else   (((::std::vector<float>*)o)->end)();
}

static void method_4459( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<const float*,std::vector<float> >)((((const ::std::vector<float>*)o)->end)());
else   (((const ::std::vector<float>*)o)->end)();
}

static void method_4464( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<float>*)o)->size)());
else   (((const ::std::vector<float>*)o)->size)();
}

static void method_4465( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<float>*)o)->max_size)());
else   (((const ::std::vector<float>*)o)->max_size)();
}

static void method_4466( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<float>*)o)->resize)(*(::size_t*)arg[0],
    *(const float*)arg[1]);
}

static void method_4467( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<float>*)o)->resize)(*(::size_t*)arg[0]);
}

static void method_4468( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (size_t)((((const ::std::vector<float>*)o)->capacity)());
else   (((const ::std::vector<float>*)o)->capacity)();
}

static void method_4469( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) new (retaddr) (bool)((((const ::std::vector<float>*)o)->empty)());
else   (((const ::std::vector<float>*)o)->empty)();
}

static void method_4470( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<float>*)o)->reserve)(*(::size_t*)arg[0]);
}

static void operator_4471( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<float>*)o)->operator[])(*(::size_t*)arg[0]);
else   (((::std::vector<float>*)o)->operator[])(*(::size_t*)arg[0]);
}

static void operator_4472( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<float>*)o)->operator[])(*(::size_t*)arg[0]);
else   (((const ::std::vector<float>*)o)->operator[])(*(::size_t*)arg[0]);
}

static void method_4474( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<float>*)o)->at)(*(::size_t*)arg[0]);
else   (((::std::vector<float>*)o)->at)(*(::size_t*)arg[0]);
}

static void method_4475( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<float>*)o)->at)(*(::size_t*)arg[0]);
else   (((const ::std::vector<float>*)o)->at)(*(::size_t*)arg[0]);
}

static void method_4476( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<float>*)o)->front)();
else   (((::std::vector<float>*)o)->front)();
}

static void method_4477( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<float>*)o)->front)();
else   (((const ::std::vector<float>*)o)->front)();
}

static void method_4478( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((::std::vector<float>*)o)->back)();
else   (((::std::vector<float>*)o)->back)();
}

static void method_4479( void* retaddr, void* o, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**)retaddr = (void*)&(((const ::std::vector<float>*)o)->back)();
else   (((const ::std::vector<float>*)o)->back)();
}

static void method_4480( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<float>*)o)->push_back)(*(const float*)arg[0]);
}

static void method_4481( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<float>*)o)->pop_back)();
}

static void method_4482( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<float*,std::vector<float> >)((((::std::vector<float>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<float*,std::vector<float> >*)arg[0],
    *(const float*)arg[1]));
else   (((::std::vector<float>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<float*,std::vector<float> >*)arg[0],
    *(const float*)arg[1]);
}

static void method_4483( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<float>*)o)->insert)(*(::__gnu_cxx::__normal_iterator<float*,std::vector<float> >*)arg[0],
    *(::size_t*)arg[1],
    *(const float*)arg[2]);
}

static void method_4484( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<float*,std::vector<float> >)((((::std::vector<float>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<float*,std::vector<float> >*)arg[0]));
else   (((::std::vector<float>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<float*,std::vector<float> >*)arg[0]);
}

static void method_4485( void* retaddr, void* o, const std::vector<void*>& arg, void*)
{
  if (retaddr) new (retaddr) (__gnu_cxx::__normal_iterator<float*,std::vector<float> >)((((::std::vector<float>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<float*,std::vector<float> >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<float*,std::vector<float> >*)arg[1]));
else   (((::std::vector<float>*)o)->erase)(*(::__gnu_cxx::__normal_iterator<float*,std::vector<float> >*)arg[0],
    *(::__gnu_cxx::__normal_iterator<float*,std::vector<float> >*)arg[1]);
}

static void method_4486( void*, void* o, const std::vector<void*>& arg, void*)
{
  (((::std::vector<float>*)o)->swap)(*(::std::vector<float>*)arg[0]);
}

static void method_4487( void*, void* o, const std::vector<void*>&, void*)
{
  (((::std::vector<float>*)o)->clear)();
}

static void constructor_x55( void* retaddr, void* mem, const std::vector<void*>&, void*) {
  if (retaddr) *(void**)retaddr = ::new(mem) ::std::vector<float>();
  else ::new(mem) ::std::vector<float>();
}

static void method_newdel_1393( void* retaddr, void*, const std::vector<void*>&, void*)
{
  static ::Reflex::NewDelFunctions s_funcs;
  s_funcs.fNew         = ::Reflex::NewDelFunctionsT< ::std::vector<float> >::new_T;
  s_funcs.fNewArray    = ::Reflex::NewDelFunctionsT< ::std::vector<float> >::newArray_T;
  s_funcs.fDelete      = ::Reflex::NewDelFunctionsT< ::std::vector<float> >::delete_T;
  s_funcs.fDeleteArray = ::Reflex::NewDelFunctionsT< ::std::vector<float> >::deleteArray_T;
  s_funcs.fDestructor  = ::Reflex::NewDelFunctionsT< ::std::vector<float> >::destruct_T;
  if (retaddr) *(::Reflex::NewDelFunctions**)retaddr = &s_funcs;
}

static void method_x57( void* retaddr, void*, const std::vector<void*>&, void*)
{
  typedef std::vector<std::pair< ::Reflex::Base, int> > Bases_t;
  static Bases_t s_bases;
  if ( !s_bases.size() ) {
    s_bases.push_back(std::make_pair(::Reflex::Base( ::Reflex::TypeBuilder("std::_Vector_base<float,std::allocator<float> >"), ::Reflex::BaseOffset< ::std::vector<float>,::std::_Vector_base<float,std::allocator<float> > >::Get(),::Reflex::PROTECTED), 0));
  }
  if (retaddr) *(Bases_t**)retaddr = &s_bases;
}

static void method_x58( void* retaddr, void*, const std::vector<void*>&, void*)
{
  if (retaddr) *(void**) retaddr = ::Reflex::Proxy< ::std::vector<float> >::Generate();
  else ::Reflex::Proxy< ::std::vector<float> >::Generate();
}

//------Dictionary for class vector<float,std::allocator<float> > -------------------------------
void __std__vector_float__db_datamem(Reflex::Class*);
void __std__vector_float__db_funcmem(Reflex::Class*);
Reflex::GenreflexMemberBuilder __std__vector_float__datamem_bld(&__std__vector_float__db_datamem);
Reflex::GenreflexMemberBuilder __std__vector_float__funcmem_bld(&__std__vector_float__db_funcmem);
void __std__vector_float__dict() {
  ::Reflex::ClassBuilder("std::vector<float>", typeid(::std::vector<float>), sizeof(::std::vector<float>), ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL, ::Reflex::CLASS)
  .AddBase(type_1649, ::Reflex::BaseOffset< ::std::vector<float>, ::std::_Vector_base<float,std::allocator<float> > >::Get(), ::Reflex::PROTECTED)
  .AddTypedef(type_1649, "std::vector<float>::_Base")
  .AddTypedef(type_1823, "std::vector<float>::value_type")
  .AddTypedef(type_2423, "std::vector<float>::pointer")
  .AddTypedef(type_4436, "std::vector<float>::const_pointer")
  .AddTypedef(type_4438, "std::vector<float>::reference")
  .AddTypedef(type_4440, "std::vector<float>::const_reference")
  .AddTypedef(type_2333, "std::vector<float>::iterator")
  .AddTypedef(type_2334, "std::vector<float>::const_iterator")
  .AddTypedef(type_1630, "std::vector<float>::const_reverse_iterator")
  .AddTypedef(type_1631, "std::vector<float>::reverse_iterator")
  .AddTypedef(type_854, "std::vector<float>::size_type")
  .AddTypedef(type_684, "std::vector<float>::difference_type")
  .AddTypedef(type_1543, "std::vector<float>::allocator_type")
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7301), "vector", constructor_4449, 0, "__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854, type_4440, type_7301), "vector", constructor_4450, 0, "__n;__value;__a=typename std::_Vector_base<_Tp, _Alloc>::allocator_type()", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_854), "vector", constructor_4451, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void, type_7302), "vector", constructor_4452, 0, "__x", ::Reflex::PUBLIC | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "~vector", destructor_4453, 0, 0, ::Reflex::PUBLIC | ::Reflex::DESTRUCTOR )
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_void), "vector", constructor_x55, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL | ::Reflex::CONSTRUCTOR)
  .AddFunctionMember<void*(void)>("__getNewDelFunctions", method_newdel_1393, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("__getBasesTable", method_x57, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddFunctionMember<void*(void)>("createCollFuncTable", method_x58, 0, 0, ::Reflex::PUBLIC | ::Reflex::ARTIFICIAL)
  .AddOnDemandFunctionMemberBuilder(&__std__vector_float__funcmem_bld);
}

//------Delayed data member builder for class vector<float,std::allocator<float> > -------------------
void __std__vector_float__db_datamem(Reflex::Class*) {

}
//------Delayed function member builder for class vector<float,std::allocator<float> > -------------------
void __std__vector_float__db_funcmem(Reflex::Class* cl) {
  ::Reflex::ClassBuilder(cl)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_7303, type_7302), "operator=", operator_4454, 0, "__x", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_4440), "assign", method_4455, 0, "__n;__val", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2333), "begin", method_4456, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2334), "begin", method_4457, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2333), "end", method_4458, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2334), "end", method_4459, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "size", method_4464, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "max_size", method_4465, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854, type_4440), "resize", method_4466, 0, "__new_size;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "resize", method_4467, 0, "__new_size", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_854), "capacity", method_4468, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2653), "empty", method_4469, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_854), "reserve", method_4470, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4438, type_854), "operator[]", operator_4471, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4440, type_854), "operator[]", operator_4472, 0, "__n", ::Reflex::PUBLIC | ::Reflex::OPERATOR | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4438, type_854), "at", method_4474, 0, "__n", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4440, type_854), "at", method_4475, 0, "__n", ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4438), "front", method_4476, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4440), "front", method_4477, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4438), "back", method_4478, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_4440), "back", method_4479, 0, 0, ::Reflex::PUBLIC | ::Reflex::CONST)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_4440), "push_back", method_4480, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "pop_back", method_4481, 0, 0, ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2333, type_2333, type_4440), "insert", method_4482, 0, "__position;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_2333, type_854, type_4440), "insert", method_4483, 0, "__position;__n;__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2333, type_2333), "erase", method_4484, 0, "__position", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_2333, type_2333, type_2333), "erase", method_4485, 0, "__first;__last", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113, type_7303), "swap", method_4486, 0, "__x", ::Reflex::PUBLIC)
  .AddFunctionMember(::Reflex::FunctionTypeBuilder(type_113), "clear", method_4487, 0, 0, ::Reflex::PUBLIC);
}
namespace {
  struct Dictionaries {
    Dictionaries() {
      Reflex::Instance initialize_reflex;
      __ClassD_dict(); 
      __ClassABase_dict(); 
      __ClassA_dict(); 
      __ClassB_dict(); 
      __ClassC_dict(); 
      __ClassAIns_dict(); 
      __std__vector_ClassDp__dict(); 
      __std__vector_ClassD__dict(); 
      __std__vector_ClassCp__dict(); 
      __std__vector_ClassC__dict(); 
      __std__vector_ClassBp__dict(); 
      __std__vector_ClassB__dict(); 
      __std__vector_ClassAp__dict(); 
      __std__vector_ClassA__dict(); 
      __std__pair_int_double__dict(); 
      __std__vector_std__pair_int_double_s__dict(); 
      __std__vector_double__dict(); 
      __std__pair_int_float__dict(); 
      __std__vector_std__pair_int_float_s__dict(); 
      __std__vector_float__dict(); 
    }
    ~Dictionaries() {
      type_29.Unload(); // class ClassD 
      type_776.Unload(); // class ClassABase 
      type_994.Unload(); // class ClassA 
      type_995.Unload(); // class ClassB 
      type_996.Unload(); // class ClassC 
      type_1031.Unload(); // class ClassAIns 
      type_1382.Unload(); // class std::vector<ClassD*> 
      type_1383.Unload(); // class std::vector<ClassD> 
      type_1384.Unload(); // class std::vector<ClassC*> 
      type_1385.Unload(); // class std::vector<ClassC> 
      type_1386.Unload(); // class std::vector<ClassB*> 
      type_1387.Unload(); // class std::vector<ClassB> 
      type_1388.Unload(); // class std::vector<ClassA*> 
      type_1389.Unload(); // class std::vector<ClassA> 
      type_1430.Unload(); // class std::pair<int,double> 
      type_1390.Unload(); // class std::vector<std::pair<int,double> > 
      type_1391.Unload(); // class std::vector<double> 
      type_1431.Unload(); // class std::pair<int,float> 
      type_1392.Unload(); // class std::vector<std::pair<int,float> > 
      type_1393.Unload(); // class std::vector<float> 
    }
  };
  static Dictionaries instance;
}
} // unnamed namespace
