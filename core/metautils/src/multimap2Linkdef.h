#include <multimap>
#include <string>
#ifndef __hpux
using namespace std;
#endif

#pragma create TClass multimap<long,int>;
#pragma create TClass multimap<long,long>;
#pragma create TClass multimap<long,double>;
#pragma create TClass multimap<long,void*>;
#pragma create TClass multimap<long,char*>;

#pragma create TClass multimap<double,int>;
#pragma create TClass multimap<double,long>;
#pragma create TClass multimap<double,double>;
#pragma create TClass multimap<double,void*>;
#pragma create TClass multimap<double,char*>;
