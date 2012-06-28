/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// lib/dll_stl/multmp.h

#ifdef __CINT__
#include <multimap>
#else
#include <map>
#endif
#include <algorithm>
#include <string>
#ifndef __hpux
using namespace std;
#endif

#ifdef __MAKECINT__
#ifndef G__MULTIMAP_DLL
#define G__MULTIMAP_DLL
#endif
#pragma link C++ global G__MULTIMAP_DLL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

#ifdef G__MAP2
#pragma link C++ class multimap<long,int>;
#pragma link C++ class multimap<long,long>;
#pragma link C++ class multimap<long,double>;
#pragma link C++ class multimap<long,void*>;
#pragma link C++ class multimap<long,char*>;

#pragma link C++ class multimap<double,int>;
#pragma link C++ class multimap<double,long>;
#pragma link C++ class multimap<double,double>;
#pragma link C++ class multimap<double,void*>;
#pragma link C++ class multimap<double,char*>;

#pragma link off function pair<const long,int>::operator=;
#pragma link off function pair<const long,long>::operator=;
#pragma link off function pair<const long,double>::operator=;
#pragma link off function pair<const long,void*>::operator=;
#pragma link off function pair<const long,char*>::operator=;
#pragma link off function pair<const double,int>::operator=;
#pragma link off function pair<const double,long>::operator=;
#pragma link off function pair<const double,double>::operator=;
#pragma link off function pair<const double,void*>::operator=;
#pragma link off function pair<const double,char*>::operator=;
#endif

#ifndef G__MAP2
#pragma link C++ class multimap<char*,int>;
#pragma link C++ class multimap<char*,long>;
#pragma link C++ class multimap<char*,double>;
#pragma link C++ class multimap<char*,void*>;
#pragma link C++ class multimap<char*,char*>;

#if defined(G__STRING_DLL) || defined(G__ROOT)
#pragma link C++ class multimap<string,int>;
#pragma link C++ class multimap<string,long>;
#pragma link C++ class multimap<string,double>;
#pragma link C++ class multimap<string,void*>;
//#pragma link C++ class multimap<string,string>;
#pragma link off function pair<const string,int>::operator=;
#pragma link off function pair<const string,long>::operator=;
#pragma link off function pair<const string,double>::operator=;
#pragma link off function pair<const string,void*>::operator=;
#endif

#endif // G__MAP2

#if defined(G__ROOT)

#pragma link off class pair<char*,int>;
#pragma link off class pair<char*,long>;
#pragma link off class pair<char*,float>;
#pragma link off class pair<char*,double>;
#pragma link off class pair<char*,void*>;
#pragma link off class pair<char*,char*>;
#pragma link off class pair<string,int>;
#pragma link off class pair<string,long>;
#pragma link off class pair<string,float>;
#pragma link off class pair<string,double>;
#pragma link off class pair<string,void*>;
#pragma link off class pair<long,int>;
#pragma link off class pair<long,long>;
#pragma link off class pair<long,float>;
#pragma link off class pair<long,double>;
#pragma link off class pair<long,void*>;
#pragma link off class pair<long,char*>;
#pragma link off class pair<double,int>;
#pragma link off class pair<double,long>;
#pragma link off class pair<double,float>;
#pragma link off class pair<double,double>;
#pragma link off class pair<double,void*>;
#pragma link off class pair<double,char*>;

#pragma link off class pair<const char*,int>;
#pragma link off class pair<const char*,long>;
#pragma link off class pair<const char*,float>;
#pragma link off class pair<const char*,double>;
#pragma link off class pair<const char*,void*>;
#pragma link off class pair<const char*,char*>;
#pragma link off class pair<const string,int>;
#pragma link off class pair<const string,long>;
#pragma link off class pair<const string,float>;
#pragma link off class pair<const string,double>;
#pragma link off class pair<const string,void*>;
#pragma link off class pair<const long,int>;
#pragma link off class pair<const long,long>;
#pragma link off class pair<const long,float>;
#pragma link off class pair<const long,double>;
#pragma link off class pair<const long,void*>;
#pragma link off class pair<const long,char*>;
#pragma link off class pair<const double,int>;
#pragma link off class pair<const double,long>;
#pragma link off class pair<const double,float>;
#pragma link off class pair<const double,double>;
#pragma link off class pair<const double,void*>;
#pragma link off class pair<const double,char*>;
#endif

#endif


