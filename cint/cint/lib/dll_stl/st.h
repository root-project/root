/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// lib/dll_stl/st.h

#include <set>
#include <algorithm>
#include <string>
#ifndef __hpux
using namespace std;
#endif

#ifdef __MAKECINT__
#ifndef G__SET_DLL
#define G__SET_DLL
#endif
#pragma link C++ global G__SET_DLL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

#pragma link C++ class set<int>;
#pragma link C++ class set<long>;
#pragma link C++ class set<float>;
#pragma link C++ class set<double>;
#pragma link C++ class set<void*>;
#pragma link C++ class set<char*>;
#if defined(G__STRING_DLL) || defined(G__ROOT)
#pragma link C++ class set<string>;
#endif

#endif


