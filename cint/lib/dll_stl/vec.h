// lib/dll_stl/vec.h

#include <vector>
#include <algorithm>
#include <memory>
#include <string>

#ifndef __hpux
using namespace std;
#endif

#if (__SUNPRO_CC>=1280)
#include "suncc5_string.h"
#endif

#ifdef __MAKECINT__
#ifndef G__VECTOR_DLL
#define G__VECTOR_DLL
#endif
#pragma link C++ global G__VECTOR_DLL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

#pragma link C++ class vector<char>;
#pragma link C++ class vector<short>;
#pragma link C++ class vector<int>;
#pragma link C++ class vector<long>;
#pragma link C++ class vector<double>;
#pragma link C++ class vector<void*>;
#pragma link C++ class vector<char*>;
#ifdef G__STRING_DLL
#pragma link C++ class vector<string>;
#endif


#endif

