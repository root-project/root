/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// lib/dll_stl/str.h

#include <iostream>
#ifdef __MAKECINT__
#pragma link off all classes;
#pragma link off all functions;
#pragma link off all globals;
#endif

#include <string>

#ifndef __hpux
using namespace std;
#endif

#if 0 && (__SUNPRO_CC>=1280)
#include "suncc5_string.h"  // obsolete
#endif

#ifdef __MAKECINT__
#pragma ifndef G__STRING_DLL
#pragma define G__STRING_DLL
#pragma endif
#pragma link C++ global G__STRING_DLL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

//#pragma link C++ basic_string<char, char_traits<char>, allocator<char> >;
//#pragma link C++ typedef string;

#pragma link C++ class string;
#pragma link C++ typedef string::value_type;

#pragma if  (defined (G__VISUAL) && (G__MSC_VER>=1310))
#pragma link C++ class string::iterator;
#pragma link C++ class string::const_iterator;
#if (G__MSC_VER<1600)
#pragma link C++ class _Ranit<char,long,char*,char&>;
#endif
#pragma link C++ class iterator<random_access_iterator_tag,char,long,char*,char&>;
#pragma else
//#pragma if ((G__GNUC>=3 && G__GNUC_MINOR>=1) && !defined(G__INTEL_COMPILER)) 
#pragma if (G__GNUC_VER>=3001) && !defined(G__INTEL_COMPILER)) 
#pragma link C++ class string::iterator;
#pragma else
#pragma link C++ typedef string::iterator;
#pragma endif
#pragma link C++ typedef string::const_iterator;
#pragma endif

#pragma link C++ typedef string::pointer;
#pragma link C++ typedef string::const_pointer;
#pragma link C++ typedef string::reference;
#pragma link C++ typedef string::difference_type;
#pragma link C++ typedef string::size_type;
#pragma link C++ typedef string::traits_type;
#ifndef G__OLDIMPLEMENTATION1598
#pragma link C++ function operator==(const string&,const string&);
#pragma link C++ function operator!=(const string&,const string&);
#pragma link C++ function operator<(const string&,const string&);
#pragma link C++ function operator>(const string&,const string&);
#pragma link C++ function operator<=(const string&,const string&);
#pragma link C++ function operator>=(const string&,const string&);
#pragma link C++ function operator+(const string&,const string&);
#pragma link C++ function operator+(char,const string&);
#pragma link C++ function operator+(const string&,char);
#endif

//#if G__ROOT
#pragma link C++ function operator<(const char*,const string&);
#pragma link C++ function operator>(const char*,const string&);
#pragma link C++ function operator==(const char*,const string&);
#pragma link C++ function operator!=(const char*,const string&);
#pragma link C++ function operator<=(const char*,const string&);
#pragma link C++ function operator>=(const char*,const string&);
#pragma link C++ function operator+(const char*,const string&);

#pragma link C++ function operator<(const string&,const char*);
#pragma link C++ function operator>(const string&,const char*);
#pragma link C++ function operator==(const string&,const char*);
#pragma link C++ function operator!=(const string&,const char*);
#pragma link C++ function operator<=(const string&,const char*);
#pragma link C++ function operator>=(const string&,const char*);
#pragma link C++ function operator+(const string&,const char*);

#pragma link C++ function swap(string&,string&);
#pragma link C++ function operator>>(istream&,string&);
#pragma link C++ function operator<<(ostream&,const string&);
#pragma link C++ function getline;
//#endif // G__ROOT

#pragma if (G__GNUC_VER>=3001) && !defined(G__INTEL_COMPILER)
#pragma link C++ options=stub function operator==(const string::iterator&,const string::iterator&);
#pragma link C++ options=stub function operator!=(const string::iterator&,const string::iterator&);
#pragma endif
#endif // __MAKECINT__


