#include <map>
#include <string>
#ifndef __hpux
using namespace std;
#endif

#pragma create TClass map<int,int>;
#pragma create TClass map<long,int>;
#pragma create TClass map<long,long>;
#pragma create TClass map<long,float>;
#pragma create TClass map<long,double>;
#pragma create TClass map<long,void*>;
#pragma create TClass map<long,char*>;

#pragma create TClass map<double,int>;
#pragma create TClass map<double,long>;
#pragma create TClass map<double,float>;
#pragma create TClass map<double,double>;
#pragma create TClass map<double,void*>;
#pragma create TClass map<double,char*>;
