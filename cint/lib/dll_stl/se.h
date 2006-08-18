/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// lib/dll_stl/se.h

#include <stdexcept>
#ifndef __hpux
using namespace std;
#endif
#ifdef __SUNPRO_CC
#define exception std::exception
#endif

#ifdef __MAKECINT__
#ifndef G__STDEXCEPT_DLL
#define G__STDEXCEPT_DLL
#endif

#pragma link C++ global G__STDEXCEPT_DLL;
#pragma link C++ class logic_error;
#pragma link C++ class domain_error;
#pragma link C++ class invalid_argument;
#pragma link C++ class length_error;
#pragma link C++ class out_of_range;
#pragma link C++ class runtime_error;
#pragma link C++ class range_error;
#pragma link C++ class overflow_error;
#pragma link C++ class underflow_error;

#endif

