#include <multimap>
#include <string>
#ifndef __hpux
using namespace std;
#endif

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

#pragma link C++ class multimap<string,int>;
#pragma link C++ class multimap<string,long>;
#pragma link C++ class multimap<string,double>;
#pragma link C++ class multimap<string,void*>;
//#pragma link C++ class multimap<string,string>;
#endif // G__MAP2

