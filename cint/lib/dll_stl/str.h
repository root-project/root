// lib/dll_stl/str.h

#include <string>

#ifndef __hpux
using namespace std;
#endif

#if (__SUNPRO_CC>=1280)
#include "suncc5_string.h"
#endif

#ifdef __MAKECINT__
#ifndef G__STRING_DLL
#define G__STRING_DLL
#endif
#pragma link C++ global G__STRING_DLL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;
#pragma link C++ class string;
#endif


