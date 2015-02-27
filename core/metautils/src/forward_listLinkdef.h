#include <forward_list>
#include <string>
#ifndef __hpux
using namespace std;
#endif

#pragma create TClass forward_list<int>;
#pragma create TClass forward_list<long>;
#pragma create TClass forward_list<float>;
#pragma create TClass forward_list<double>;
#pragma create TClass forward_list<void*>;
#pragma create TClass forward_list<char*>;
#pragma create TClass forward_list<string>;

// 
