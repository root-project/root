#include <unordered_map>
#include <string>
#ifndef __hpux
using namespace std;
#endif

#pragma create TClass unordered_multimap<long,int>;
#pragma create TClass unordered_multimap<long,long>;
#pragma create TClass unordered_multimap<long,double>;
#pragma create TClass unordered_multimap<long,void*>;
#pragma create TClass unordered_multimap<long,char*>;

#pragma create TClass unordered_multimap<double,int>;
#pragma create TClass unordered_multimap<double,long>;
#pragma create TClass unordered_multimap<double,double>;
#pragma create TClass unordered_multimap<double,void*>;
#pragma create TClass unordered_multimap<double,char*>;

#pragma create TClass unordered_multimap<char*,int>;
#pragma create TClass unordered_multimap<char*,long>;
#pragma create TClass unordered_multimap<char*,double>;
#pragma create TClass unordered_multimap<char*,void*>;
#pragma create TClass unordered_multimap<char*,char*>;

#pragma create TClass unordered_multimap<string,int>;
#pragma create TClass unordered_multimap<string,long>;
#pragma create TClass unordered_multimap<string,double>;
#pragma create TClass unordered_multimap<string,void*>;

