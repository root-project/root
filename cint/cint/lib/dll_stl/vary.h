/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
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

#pragma link off function valarray<bool>::abs;
#pragma link off function valarray<bool>::acos;
#pragma link off function valarray<bool>::asin;
#pragma link off function valarray<bool>::atan;
#pragma link off function valarray<bool>::atan2;
#pragma link off function valarray<bool>::cos;
#pragma link off function valarray<bool>::cosh;
#pragma link off function valarray<bool>::exp;
#pragma link off function valarray<bool>::log;
#pragma link off function valarray<bool>::log10;
#pragma link off function valarray<bool>::pow;
#pragma link off function valarray<bool>::sin;
#pragma link off function valarray<bool>::sinh;
#pragma link off function valarray<bool>::sqrt;
#pragma link off function valarray<bool>::tan;
#pragma link off function valarray<bool>::tanh;

#endif

