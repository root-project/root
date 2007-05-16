// Generated at Tue May 15 12:40:48 2007. Do not modify it

// NOTE: as we build this with CompileMacro(), we need to implement the
// Shadows' constructors (undef symbols otherwise, as CompileMacro
// creates a dict of the dict, which requests these symbols)
// Other than that it's generated with "genreflex FOO.h"

/*
GCC-XML version 0.6.0
Configuration settings:
  GCCXML_CONFIG="/afs/cern.ch/sw/lcg/external/gccxml/0.6.0_patch3/slc4_amd64_gcc345/share/gccxml-0.6/gccxml_config"
  GCCXML_COMPILER="g++"
  GCCXML_CXXFLAGS=" "
  GCCXML_EXECUTABLE="/afs/cern.ch/sw/lcg/external/gccxml/0.6.0_patch3/slc4_amd64_gcc345/bin/gccxml_cc1plus"
  GCCXML_CPP="/afs/cern.ch/sw/lcg/external/gccxml/0.6.0_patch3/slc4_amd64_gcc345/bin/gccxml_cc1plus"
  GCCXML_FLAGS="-D__DBL_MIN_EXP__=(-1021) -D__FLT_MIN__=1.17549435e-38F -D__CHAR_BIT__=8 -D__WCHAR_MAX__=2147483647 -D__DBL_DENORM_MIN__=4.9406564584124654e-324 -D__FLT_EVAL_METHOD__=0 -D__x86_64=1 -D__DBL_MIN_10_EXP__=(-307) -D__FINITE_MATH_ONLY__=0 -D__LP64__=1 -D__SHRT_MAX__=32767 -D__LDBL_MAX__=1.18973149535723176502e+4932L -D__linux=1 -D__unix=1 -D__linux__=1 -D__SCHAR_MAX__=127 -D__USER_LABEL_PREFIX__= -D__STDC_HOSTED__=1 -D__DBL_DIG__=15 -D__FLT_EPSILON__=1.19209290e-7F -D__GXX_WEAK__=1 -D__LDBL_MIN__=3.36210314311209350626e-4932L -D__unix__=1 -D__DECIMAL_DIG__=21 -D__gnu_linux__=1 -D__LDBL_HAS_QUIET_NAN__=1 -D__GNUC__=3 -D__MMX__=1 -D__DBL_MAX__=1.7976931348623157e+308 -D__DBL_HAS_INFINITY__=1 -D__cplusplus=1 -D__DEPRECATED=1 -D__DBL_MAX_EXP__=1024 -D__SSE2_MATH__=1 -D__amd64=1 -D__GNUG__=3 -D__LONG_LONG_MAX__=9223372036854775807LL -D__GXX_ABI_VERSION=1002 -D__FLT_MIN_EXP__=(-125) -D__DBL_MIN__=2.2250738585072014e-308 -D__FLT_MIN_10_EXP__=(-37) -D__DBL_HAS_QUIET_NAN__=1 -D__REGISTER_PREFIX__= -D__NO_INLINE__=1 -D__FLT_MANT_DIG__=24 -D__VERSION__="3.4.6 20060404 (Red Hat 3.4.6-3)" -Dunix=1 -D__SIZE_TYPE__=long unsigned int -D__ELF__=1 -D__FLT_RADIX__=2 -D__LDBL_EPSILON__=1.08420217248550443401e-19L -D__GNUC_RH_RELEASE__=3 -D__k8=1 -D__x86_64__=1 -D__FLT_HAS_QUIET_NAN__=1 -D__FLT_MAX_10_EXP__=38 -D__LONG_MAX__=9223372036854775807L -D__FLT_HAS_INFINITY__=1 -Dlinux=1 -D__EXCEPTIONS=1 -D__LDBL_MANT_DIG__=64 -D__k8__=1 -D__WCHAR_TYPE__=int -D__FLT_DIG__=6 -D__INT_MAX__=2147483647 -D__FLT_MAX_EXP__=128 -D__DBL_MANT_DIG__=53 -D__WINT_TYPE__=unsigned int -D__SSE__=1 -D__LDBL_MIN_EXP__=(-16381) -D__amd64__=1 -D__LDBL_MAX_EXP__=16384 -D__LDBL_MAX_10_EXP__=4932 -D__DBL_EPSILON__=2.2204460492503131e-16 -D_LP64=1 -D__GNUC_PATCHLEVEL__=6 -D__LDBL_HAS_INFINITY__=1 -D__tune_k8__=1 -D__FLT_DENORM_MIN__=1.40129846e-45F -D__FLT_MAX__=3.40282347e+38F -D__SSE2__=1 -D__GNUC_MINOR__=4 -D__DBL_MAX_10_EXP__=308 -D__LDBL_DENORM_MIN__=3.64519953188247460253e-4951L -D__PTRDIFF_TYPE__=long int -D__LDBL_MIN_10_EXP__=(-4931) -D__SSE_MATH__=1 -D__LDBL_DIG__=18 -D_GNU_SOURCE=1 -D__declspec(x)= -D__attribute__(x)= -iwrapper/afs/cern.ch/sw/lcg/external/gccxml/0.6.0_patch3/slc4_amd64_gcc345/share/gccxml-0.6/GCC/3.4 -I/usr/lib/gcc/x86_64-redhat-linux/3.4.6/../../../../include/c++/3.4.6 -I/usr/lib/gcc/x86_64-redhat-linux/3.4.6/../../../../include/c++/3.4.6/x86_64-redhat-linux -I/usr/lib/gcc/x86_64-redhat-linux/3.4.6/../../../../include/c++/3.4.6/backward -I/usr/local/include -I/usr/lib/gcc/x86_64-redhat-linux/3.4.6/include -I/usr/include -include /afs/cern.ch/sw/lcg/external/gccxml/0.6.0_patch3/slc4_amd64_gcc345/share/gccxml-0.6/GCC/3.4/gccxml_builtins.h "
  GCCXML_USER_FLAGS=""
  GCCXML_ROOT="/afs/cern.ch/sw/lcg/external/gccxml/0.6.0_patch3/slc4_amd64_gcc345/share/gccxml-0.6"

Compiler info:
g++ (GCC) 3.4.6 20060404 (Red Hat 3.4.6-3)
Copyright (C) 2006 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

*/

#ifdef _WIN32
#pragma warning ( disable : 4786 )
#endif
#include "FOO.h"
#include "Reflex/Builder/ReflexBuilder.h"
#include <typeinfo>
using namespace ROOT::Reflex;

namespace {
  Type type_void = TypeBuilder("void");
  Type type_97 = TypeBuilder("int");
  Type type_102 = TypeBuilder("double");
  Type type_4 = TypeBuilder("BAR");
  Type type_111 = PointerBuilder(type_4);
  Type type_3 = TypeBuilder("FOO");
  Type type_3c = ConstBuilder(type_3);
  Type type_112 = ReferenceBuilder(type_3c);
  Type type_101 = TypeBuilder("float");
  Type type_4c = ConstBuilder(type_4);
  Type type_113 = ReferenceBuilder(type_4c);
} // unnamed namespace

// Shadow classes to obtain the data member offsets 
namespace __shadow__ {
class __FOO {
  public:
  __FOO(){};
  int a;
  double b;
  ::BAR* bar;
};
class __BAR {
  public:
  __BAR(){};
  int i;
  float f;
};
}

namespace {
//------Stub functions for class FOO -------------------------------
static void* constructor_88( void* mem, const std::vector<void*>& arg, void*) {
  return ::new(mem) ::FOO(*(const ::FOO*)arg[0]);
}

static void* constructor_89( void* mem, const std::vector<void*>&, void*) {
  return ::new(mem) ::FOO();
}

static void* method_90( void* o, const std::vector<void*>&, void*)
{
  static int ret;
  ret = ((const ::FOO*)o)->A();
  return &ret;
}

static void* method_91( void* o, const std::vector<void*>&, void*)
{
  static int ret;
  ret = ((::FOO*)o)->A();
  return &ret;
}

static void* method_92( void* o, const std::vector<void*>&, void*)
{
  static double ret;
  ret = ((::FOO*)o)->B();
  return &ret;
}

static void* method_x0( void*, const std::vector<void*>&, void*)
{
  static NewDelFunctions s_funcs;
  s_funcs.fNew         = NewDelFunctionsT< ::FOO >::new_T;
  s_funcs.fNewArray    = NewDelFunctionsT< ::FOO >::newArray_T;
  s_funcs.fDelete      = NewDelFunctionsT< ::FOO >::delete_T;
  s_funcs.fDeleteArray = NewDelFunctionsT< ::FOO >::deleteArray_T;
  s_funcs.fDestructor  = NewDelFunctionsT< ::FOO >::destruct_T;
  return &s_funcs;
}

//------Dictionary for class FOO -------------------------------
void __FOO_dict() {
  ClassBuilder("FOO", typeid(FOO), sizeof(FOO), PUBLIC, CLASS)
  .AddDataMember(type_97, "a", OffsetOf(__shadow__::__FOO, a), PUBLIC)
  .AddDataMember(type_102, "b", OffsetOf(__shadow__::__FOO, b), PUBLIC)
  .AddDataMember(type_111, "bar", OffsetOf(__shadow__::__FOO, bar), PUBLIC)
  .AddFunctionMember(FunctionTypeBuilder(type_void, type_112), "FOO", constructor_88, 0, "_ctor_arg", PUBLIC | ARTIFICIAL | CONSTRUCTOR)
  .AddFunctionMember(FunctionTypeBuilder(type_void), "FOO", constructor_89, 0, 0, PUBLIC | ARTIFICIAL | CONSTRUCTOR)
  .AddFunctionMember(FunctionTypeBuilder(type_97), "A", method_90, 0, 0, PUBLIC | CONST)
  .AddFunctionMember(FunctionTypeBuilder(type_97), "A", method_91, 0, 0, PUBLIC)
  .AddFunctionMember(FunctionTypeBuilder(type_102), "B", method_92, 0, 0, PUBLIC)
  .AddFunctionMember<void*(void)>("__getNewDelFunctions", method_x0, 0, 0, PUBLIC | ARTIFICIAL);
}

//------Stub functions for class BAR -------------------------------
static void* constructor_95( void* mem, const std::vector<void*>& arg, void*) {
  return ::new(mem) ::BAR(*(const ::BAR*)arg[0]);
}

static void* constructor_96( void* mem, const std::vector<void*>&, void*) {
  return ::new(mem) ::BAR();
}

static void* method_x1( void*, const std::vector<void*>&, void*)
{
  static NewDelFunctions s_funcs;
  s_funcs.fNew         = NewDelFunctionsT< ::BAR >::new_T;
  s_funcs.fNewArray    = NewDelFunctionsT< ::BAR >::newArray_T;
  s_funcs.fDelete      = NewDelFunctionsT< ::BAR >::delete_T;
  s_funcs.fDeleteArray = NewDelFunctionsT< ::BAR >::deleteArray_T;
  s_funcs.fDestructor  = NewDelFunctionsT< ::BAR >::destruct_T;
  return &s_funcs;
}

//------Dictionary for class BAR -------------------------------
void __BAR_dict() {
  ClassBuilder("BAR", typeid(BAR), sizeof(BAR), PUBLIC, CLASS)
  .AddDataMember(type_97, "i", OffsetOf(__shadow__::__BAR, i), PUBLIC)
  .AddDataMember(type_101, "f", OffsetOf(__shadow__::__BAR, f), PUBLIC)
  .AddFunctionMember(FunctionTypeBuilder(type_void, type_113), "BAR", constructor_95, 0, "_ctor_arg", PUBLIC | ARTIFICIAL | CONSTRUCTOR)
  .AddFunctionMember(FunctionTypeBuilder(type_void), "BAR", constructor_96, 0, 0, PUBLIC | ARTIFICIAL | CONSTRUCTOR)
  .AddFunctionMember<void*(void)>("__getNewDelFunctions", method_x1, 0, 0, PUBLIC | ARTIFICIAL);
}

namespace {
  struct Dictionaries {
    Dictionaries() {
      __FOO_dict(); 
      __BAR_dict(); 
    }
    ~Dictionaries() {
      type_3.Unload(); // class FOO 
      type_4.Unload(); // class BAR 
    }
  };
  static Dictionaries instance;
}
} // unnamed namespace
