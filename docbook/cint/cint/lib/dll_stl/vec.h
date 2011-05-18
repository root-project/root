/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
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
#ifndef G__ROOT
#pragma link C++ class vector<int>;
#endif
#pragma link C++ class vector<long>;
#pragma link C++ class vector<long long>;

#pragma link C++ class vector<unsigned char>;
#pragma link C++ class vector<unsigned short>;
#pragma link C++ class vector<unsigned int>;
#pragma link C++ class vector<unsigned long>;
#pragma link C++ class vector<unsigned long long>;

#pragma link C++ class vector<float>;
#pragma link C++ class vector<double>;
//#if (G__GNUC<3 || G__GNUC_MINOR<1) && !defined(G__KCC)
//#if (!(G__GNUC==3 && G__GNUC_MINOR==1)) && !defined(G__KCC)
//#if (!(G__GNUC==3 && G__GNUC_MINOR==1)) && !defined(G__KCC) && (!defined(G__VISUAL) || G__MSC_VER<1300)
#if (G__GNUC_VER!=3001&&G__GNUC_VER!=3002) && !defined(G__KCC) && (!defined(G__VISUAL) || G__MSC_VER<1300)
// gcc3.1,3.2 has a problem with iterator<void*,...,void&>
#pragma link C++ class vector<void*>;
#endif
#pragma link C++ class vector<char*>;
#if 0
// currently does not work on most platform
#pragma link C++ class vector<const char*>;
#endif
#if defined(G__STRING_DLL) // || defined(G__ROOT)
#pragma link C++ class vector<string>;
#endif

//#if (G__GNUC>=3 && G__GNUC_MINOR>=1)
#if defined(G__GNUC_VER) && (G__GNUC_VER>=3001) 
#ifdef G__OLDIMPLEMENTATION1703
#pragma link C++ namespace __gnu_cxx;
#endif
#endif


#endif

