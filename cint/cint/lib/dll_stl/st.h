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
#pragma link C++ operators set<string>::iterator;
#endif

#pragma link C++ operators set<int>::iterator;
#pragma link C++ operators set<long>::iterator;
#pragma link C++ operators set<float>::iterator;
#pragma link C++ operators set<double>::iterator;
#pragma link C++ operators set<void*>::iterator;
#pragma link C++ operators set<char*>::iterator;

#pragma link C++ operators set<int>::const_iterator;
#pragma link C++ operators set<long>::const_iterator;
#pragma link C++ operators set<float>::const_iterator;
#pragma link C++ operators set<double>::const_iterator;
#pragma link C++ operators set<void*>::const_iterator;
#pragma link C++ operators set<char*>::const_iterator;

#pragma link C++ operators set<int>::reverse_iterator;
#pragma link C++ operators set<long>::reverse_iterator;
#pragma link C++ operators set<float>::reverse_iterator;
#pragma link C++ operators set<double>::reverse_iterator;
#pragma link C++ operators set<void*>::reverse_iterator;
#pragma link C++ operators set<char*>::reverse_iterator;

#endif


