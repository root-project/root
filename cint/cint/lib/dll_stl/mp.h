/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// lib/dll_stl/mp.h

#include <map>
#include <algorithm>
#ifndef G__OLDIMPLEMENTATION2023
#include <string>
#else // 2023
#if !defined(G__SUNPRO_C) && !defined(__SUNPRO_CC)
#include <string>
#endif
#endif // 2023
#ifndef __hpux
using namespace std;
#endif

#ifdef __MAKECINT__

#ifndef G__MAP_DLL
#define G__MAP_DLL
#endif
#pragma link C++ global G__MAP_DLL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

#ifdef G__MAP2
#pragma link C++ class map<int,int>;
#pragma link C++ class map<long,int>;
#pragma link C++ class map<long,long>;
#pragma link C++ class map<long,float>;
#pragma link C++ class map<long,double>;
#pragma link C++ class map<long,void*>;
#pragma link C++ class map<long,char*>;

#pragma link C++ class map<double,int>;
#pragma link C++ class map<double,long>;
#pragma link C++ class map<double,float>;
#pragma link C++ class map<double,double>;
#pragma link C++ class map<double,void*>;
#pragma link C++ class map<double,char*>;

#pragma link C++ operators map<int,int>::iterator;
#pragma link C++ operators map<long,int>::iterator;
#pragma link C++ operators map<long,long>::iterator;
#pragma link C++ operators map<long,float>::iterator;
#pragma link C++ operators map<long,double>::iterator;
#pragma link C++ operators map<long,void*>::iterator;
#pragma link C++ operators map<long,char*>::iterator;

#pragma link C++ operators map<double,int>::iterator;
#pragma link C++ operators map<double,long>::iterator;
#pragma link C++ operators map<double,float>::iterator;
#pragma link C++ operators map<double,double>::iterator;
#pragma link C++ operators map<double,void*>::iterator;
#pragma link C++ operators map<double,char*>::iterator;

#pragma link C++ operators map<int,int>::const_iterator;
#pragma link C++ operators map<long,int>::const_iterator;
#pragma link C++ operators map<long,long>::const_iterator;
#pragma link C++ operators map<long,float>::const_iterator;
#pragma link C++ operators map<long,double>::const_iterator;
#pragma link C++ operators map<long,void*>::const_iterator;
#pragma link C++ operators map<long,char*>::const_iterator;

#pragma link C++ operators map<double,int>::const_iterator;
#pragma link C++ operators map<double,long>::const_iterator;
#pragma link C++ operators map<double,float>::const_iterator;
#pragma link C++ operators map<double,double>::const_iterator;
#pragma link C++ operators map<double,void*>::const_iterator;
#pragma link C++ operators map<double,char*>::const_iterator;

#pragma link C++ operators map<int,int>::reverse_iterator;
#pragma link C++ operators map<long,int>::reverse_iterator;
#pragma link C++ operators map<long,long>::reverse_iterator;
#pragma link C++ operators map<long,float>::reverse_iterator;
#pragma link C++ operators map<long,double>::reverse_iterator;
#pragma link C++ operators map<long,void*>::reverse_iterator;
#pragma link C++ operators map<long,char*>::reverse_iterator;

#pragma link C++ operators map<double,int>::reverse_iterator;
#pragma link C++ operators map<double,long>::reverse_iterator;
#pragma link C++ operators map<double,float>::reverse_iterator;
#pragma link C++ operators map<double,double>::reverse_iterator;
#pragma link C++ operators map<double,void*>::reverse_iterator;
#pragma link C++ operators map<double,char*>::reverse_iterator;

#pragma link off function pair<const int,int>::operator=;
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
#endif

#ifndef G__MAP2
#pragma link C++ class map<char*,int>;
#pragma link C++ class map<char*,long>;
#pragma link C++ class map<char*,float>;
#pragma link C++ class map<char*,double>;
#pragma link C++ class map<char*,void*>;
#pragma link C++ class map<char*,char*>;

#pragma link C++ operators map<char*,int>::iterator;
#pragma link C++ operators map<char*,long>::iterator;
#pragma link C++ operators map<char*,float>::iterator;
#pragma link C++ operators map<char*,double>::iterator;
#pragma link C++ operators map<char*,void*>::iterator;
#pragma link C++ operators map<char*,char*>::iterator;

#pragma link C++ operators map<char*,int>::const_iterator;
#pragma link C++ operators map<char*,long>::const_iterator;
#pragma link C++ operators map<char*,float>::const_iterator;
#pragma link C++ operators map<char*,double>::const_iterator;
#pragma link C++ operators map<char*,void*>::const_iterator;
#pragma link C++ operators map<char*,char*>::const_iterator;

#pragma link C++ operators map<char*,int>::reverse_iterator;
#pragma link C++ operators map<char*,long>::reverse_iterator;
#pragma link C++ operators map<char*,float>::reverse_iterator;
#pragma link C++ operators map<char*,double>::reverse_iterator;
#pragma link C++ operators map<char*,void*>::reverse_iterator;
#pragma link C++ operators map<char*,char*>::reverse_iterator;

#if defined(G__STRING_DLL) || defined(G__ROOT)
#pragma link C++ class map<string,int>;
#pragma link C++ class map<string,long>;
#pragma link C++ class map<string,float>;
#pragma link C++ class map<string,double>;
#pragma link C++ class map<string,void*>;
//#pragma link C++ class map<string,string>;

#pragma link C++ operators map<string,int>::iterator;
#pragma link C++ operators map<string,long>::iterator;
#pragma link C++ operators map<string,float>::iterator;
#pragma link C++ operators map<string,double>::iterator;
#pragma link C++ operators map<string,void*>::iterator;
//#pragma link C++ operators map<string,string>::iterator;

#pragma link C++ operators map<string,int>::const_iterator;
#pragma link C++ operators map<string,long>::const_iterator;
#pragma link C++ operators map<string,float>::const_iterator;
#pragma link C++ operators map<string,double>::const_iterator;
#pragma link C++ operators map<string,void*>::const_iterator;
//#pragma link C++ operators map<string,string>::const_iterator;

#pragma link C++ operators map<string,int>::reverse_iterator;
#pragma link C++ operators map<string,long>::reverse_iterator;
#pragma link C++ operators map<string,float>::reverse_iterator;
#pragma link C++ operators map<string,double>::reverse_iterator;
#pragma link C++ operators map<string,void*>::reverse_iterator;
//#pragma link C++ operators map<string,string>::reverse_iterator;

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
#endif // G__STRING_DLL

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
#pragma link off class pair<int,int>;
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
#pragma link off class pair<const int,int>;
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

#endif // __MAKECINT__
