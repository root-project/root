/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// lib/dll_stl/que.h

#include <deque>
#include <queue>
#include <algorithm>
#include <string>

#ifndef __hpux
using namespace std;
#endif

#if (__SUNPRO_CC>=1280) && !defined(G__AIX)
#include "suncc5_deque.h"
#endif

#ifdef __MAKECINT__
#ifndef G__QUEUE_DLL
#define G__QUEUE_DLL
#endif
#pragma link C++ global G__QUEUE_DLL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

#pragma link C++ class queue<int>;
#pragma link C++ class queue<long>;
#pragma link C++ class queue<double>;
#pragma link C++ class queue<void*>;
#pragma link C++ class queue<char*>;
#if defined(G__STRING_DLL) || defined(G__ROOT)
#pragma link C++ class queue<string>;
#endif

#endif

