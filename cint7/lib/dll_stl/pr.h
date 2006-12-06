/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// lib/dll_stl/pr.h

#include <utility>
#if !defined(G__SUNPRO_C) && !defined(__SUNPRO_CC)
#include <string>
#endif
#ifndef __hpux
using namespace std;
#endif

#ifdef __MAKECINT__

#ifndef G__PAIR_DLL
#define G__PAIR_DLL
#endif
#pragma link C++ global G__PAIR_DLL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

#pragma link C++ class pair<const long,int>;
#pragma link C++ class pair<const long,long>;
#pragma link C++ class pair<const long,float>;
#pragma link C++ class pair<const long,double>;
#pragma link C++ class pair<const long,void*>;
#pragma link C++ class pair<const long,char*>;

#pragma link C++ class pair<const double,int>;
#pragma link C++ class pair<const double,long>;
#pragma link C++ class pair<const double,float>;
#pragma link C++ class pair<const double,double>;
#pragma link C++ class pair<const double,void*>;
#pragma link C++ class pair<const double,char*>;

#pragma link off function pair<const long,int>::operator=;
#pragma link off function pair<const long,long>::operator=;
#pragma link off function pair<const long,float>::operator=;
#pragma link off function pair<const long,double>::operator=;
#pragma link off function pair<const long,void*>::operator=;
#pragma link off function pair<const long,char*>::operator=;
#pragma link off function pair<const double,int>::operator=;
#pragma link off function pair<const double,long>::operator=;
#pragma link off function pair<const double,float>::operator=;
#pragma link off function pair<const double,double>::operator=;
#pragma link off function pair<const double,void*>::operator=;
#pragma link off function pair<const double,char*>::operator=;

#pragma link C++ class pair<const char*,int>;
#pragma link C++ class pair<const char*,long>;
#pragma link C++ class pair<const char*,float>;
#pragma link C++ class pair<const char*,double>;
#pragma link C++ class pair<const char*,void*>;
#pragma link C++ class pair<const char*,char*>;

#if defined(G__STRING_DLL) || defined(G__ROOT)
#pragma link C++ class pair<const string,int>;
#pragma link C++ class pair<const string,long>;
#pragma link C++ class pair<const string,float>;
#pragma link C++ class pair<const string,double>;
#pragma link C++ class pair<const string,void*>;
//#pragma link C++ class pair<const string,string>;

#if 0
#if defined(G__GNUC) && (G__GNUC>=3)
#pragma link off class  pair<const string,int>;
#pragma link off class  pair<const string,long>;
#pragma link off class  pair<const string,float>;
#pragma link off class  pair<const string,double>;
#pragma link off class  pair<const string,void*>;

#pragma link off function pair<const string,int>::operator=;
#pragma link off function pair<const string,long>::operator=;
#pragma link off function pair<const string,float>::operator=;
#pragma link off function pair<const string,double>::operator=;
#pragma link off function pair<const string,void*>::operator=;
#endif // GNUC
#endif 

#endif // G__STRING_DLL


#endif // __MAKECINT__
