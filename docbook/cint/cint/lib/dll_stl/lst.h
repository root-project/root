/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// lib/dll_stl/lst.h

#include <list>
#include <algorithm>
#include <string>
#ifndef __hpux
using namespace std;
#endif

#ifdef __MAKECINT__
#ifndef G__LIST_DLL
#define G__LIST_DLL
#endif
#pragma link C++ global G__LIST_DLL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

//#pragma link C++ class list<char>;
#pragma link C++ class list<int>;
#pragma link C++ class list<long>;
#pragma link C++ class list<float>;
#pragma link C++ class list<double>;
#pragma link C++ class list<void*>;
#pragma link C++ class list<char*>;
#if defined(G__STRING_DLL) || defined(G__ROOT)
#pragma link C++ class list<string>;
#endif

#endif

