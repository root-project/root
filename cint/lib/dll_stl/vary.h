// lib/dll_stl/vary.h

#include <valarray>
//#include <algorithm>
//#include <memory>
#ifndef __hpux
using namespace std;
#endif

#ifdef __MAKECINT__
#ifndef G__VALARRAY_DLL
#define G__VALARRAY_DLL
#endif
#pragma link C++ global G__VALARRAY_DLL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

#pragma link C++ class valarray<int>;
#pragma link C++ class valarray<long>;
#define G__NOINTOPR
#pragma link C++ class valarray<float>;
#pragma link C++ class valarray<double>;
#undef G__NOINTOPR
#endif

