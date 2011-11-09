/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// lib/dll_stl/stk.h

#include <deque>
#include <stack>
#include <algorithm>
#include <string>

#ifndef __hpux
using namespace std;
#endif

#if (__SUNPRO_CC>=1280) && !defined(G__AIX)
#include "suncc5_deque.h"
#endif

#ifdef __MAKECINT__
#ifndef G__STACK_DLL
#define G__STACK_DLL
#endif
#pragma link C++ global G__STACK_DLL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

#pragma link C++ class stack<int>;
#pragma link C++ class stack<long>;
#pragma link C++ class stack<double>;
#pragma link C++ class stack<void*>;
#pragma link C++ class stack<char*>;
#if defined(G__STRING_DLL) || defined(G__ROOT)
#pragma link C++ class stack<string>;
#endif

#endif

