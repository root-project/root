// For backward compatibility only
#ifdef __CINT__
#include <multimap>
#else
#include <map>
#endif
#include <string>

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
