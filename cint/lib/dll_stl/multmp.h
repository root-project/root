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
#endif

#endif // G__MAP2

#endif


