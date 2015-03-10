#include <list>
#include <string>
#ifndef __hpux
using namespace std;
#endif

#pragma create TClass list<int>;
#pragma create TClass list<long>;
#pragma create TClass list<float>;
#pragma create TClass list<double>;
#pragma create TClass list<void*>;
#pragma create TClass list<char*>;
#pragma create TClass list<string>;

// 
