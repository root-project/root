/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// lib/dll_stl/clim.h

#include <cstddef>
#include <climits>
#include <limits>
#ifndef __hpux
using namespace std;
#endif

#ifdef __MAKECINT__

#ifndef G__CLIMITS_DLL
#define G__CLIMITS_DLL
#endif

#pragma link C++ global G__CLIMITS_DLL;

#pragma link C++  class numeric_limits<bool> ;

#pragma link C++  class numeric_limits<char> ;
#pragma link C++  class numeric_limits<signed char> ;
#pragma link C++  class numeric_limits<unsigned char> ;
#pragma link C++  class numeric_limits<wchar_t> ;

#pragma link C++  class numeric_limits<short> ;
#pragma link C++  class numeric_limits<int> ;
#pragma link C++  class numeric_limits<long> ;
#pragma link C++  class numeric_limits<unsigned short> ;
#pragma link C++  class numeric_limits<unsigned int> ;
#pragma link C++  class numeric_limits<unsigned long> ;

#pragma link C++  class numeric_limits<float> ;
#pragma link C++  class numeric_limits<double> ;
#pragma link C++  class numeric_limits<long double> ;

#endif

