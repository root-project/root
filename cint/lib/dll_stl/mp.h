// lib/dll_stl/mp.h

#include <map>
#include <algorithm>
#if !defined(G__SUNPRO_C) && !defined(__SUNPRO_CC)
#include <string>
#endif
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
#pragma link C++ class map<long,int>;
#pragma link C++ class map<long,long>;
#pragma link C++ class map<long,double>;
#pragma link C++ class map<long,void*>;
#pragma link C++ class map<long,char*>;

#pragma link C++ class map<double,int>;
#pragma link C++ class map<double,long>;
#pragma link C++ class map<double,double>;
#pragma link C++ class map<double,void*>;
#pragma link C++ class map<double,char*>;
#endif

#ifndef G__MAP2
#pragma link C++ class map<char*,int>;
#pragma link C++ class map<char*,long>;
#pragma link C++ class map<char*,double>;
#pragma link C++ class map<char*,void*>;
#pragma link C++ class map<char*,char*>;

#ifdef G__STRING_DLL
#pragma link C++ class map<string,int>;
#pragma link C++ class map<string,long>;
#pragma link C++ class map<string,double>;
#pragma link C++ class map<string,void*>;
//#pragma link C++ class map<string,string>;
#endif

#endif // G__MAP2

#endif // __MAKECINT__
