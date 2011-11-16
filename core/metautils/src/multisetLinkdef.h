#ifdef __CINT__
#include <multiset>
#else
#include <set>
#endif
#include <string>
#ifndef __hpux
using namespace std;
#endif

#pragma create TClass multiset<int>;
#pragma create TClass multiset<long>;
#pragma create TClass multiset<float>;
#pragma create TClass multiset<double>;
#pragma create TClass multiset<void*>;
#pragma create TClass multiset<char*>;
#pragma create TClass multiset<string>;
