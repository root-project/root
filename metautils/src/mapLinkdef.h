#include <map>
#include <string>
#ifndef __hpux
using namespace std;
#endif

#ifdef G__MAP2
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

#endif

#ifndef G__MAP2
#pragma create TClass map<char*,int>;
#pragma create TClass map<char*,long>;
#pragma create TClass map<char*,float>;
#pragma create TClass map<char*,double>;
#pragma create TClass map<char*,void*>;
#pragma create TClass map<char*,char*>;

#pragma create TClass map<string,int>;
#pragma create TClass map<string,long>;
#pragma create TClass map<string,float>;
#pragma create TClass map<string,double>;
#pragma create TClass map<string,void*>;
#endif // G__MAP2
