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

#ifdef G__STRING_DLL
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

#endif


