/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// lib/dll_stl/dqu.h

#include <deque>
#include <algorithm>
#include <string>

#ifndef __hpux
using namespace std;
#endif

#if (__SUNPRO_CC>=1280) && !defined(G__AIX)
#include "suncc5_deque.h"
#endif

#ifdef __MAKECINT__
#ifndef G__DEQUE_DLL
#define G__DEQUE_DLL
#endif
#pragma link C++ global G__DEQUE_DLL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

#pragma link C++ class deque<int>;
#pragma link C++ class deque<long>;
#pragma link C++ class deque<float>;
#pragma link C++ class deque<double>;
#pragma link C++ class deque<void*>;
#pragma link C++ class deque<char*>;
#if defined(G__STRING_DLL) || defined(G__ROOT)
//#pragma link C++ class deque<string>; // maybe too complex for compiler
#endif

#endif

