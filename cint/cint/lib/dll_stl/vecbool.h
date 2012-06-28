/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// lib/dll_stl/vecbool.h

#ifndef G__VECTORBOOL
#define G__VECTORBOOL
#endif
#include <vector>

#ifndef __hpux
using namespace std;
#endif


#ifdef __MAKECINT__
//typedef vector<bool,allocator<bool> > vector<bool>;

#pragma link off all classes;
#pragma link off all functions;
#pragma link off all globals;

#pragma link C++ class vector<bool>;
#if defined(G__VISUAL) 
#pragma link C++ class vector<bool>::iterator;
#pragma link C++ class vector<bool>::reverse_iterator;
#pragma link off class vector<bool>::reference;
#endif

#endif


