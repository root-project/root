#include <multimap>
#include <string>
#ifndef __hpux
using namespace std;
#endif

#ifdef G__MAP2
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
#endif

#ifndef G__MAP2
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
#endif // G__MAP2

