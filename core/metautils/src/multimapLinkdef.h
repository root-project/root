#ifdef __CINT__
#include <multimap>
#else
#include <map>
#endif
#include <string>
#ifndef __hpux
using namespace std;
#endif

#pragma create TClass multimap<char*,int>;
#pragma create TClass multimap<char*,long>;
#pragma create TClass multimap<char*,double>;
#pragma create TClass multimap<char*,void*>;
#pragma create TClass multimap<char*,char*>;

#pragma create TClass multimap<string,int>;
#pragma create TClass multimap<string,long>;
#pragma create TClass multimap<string,double>;
#pragma create TClass multimap<string,void*>;
//#pragma create TClass multimap<string,string>;

