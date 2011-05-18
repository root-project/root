#include <vector>
#include <deque>
#ifndef __hpux
using namespace std;
#endif

#pragma create TClass deque<int>;
#pragma create TClass deque<long>;
#pragma create TClass deque<float>;
#pragma create TClass deque<double>;
#pragma create TClass deque<void*>;
#pragma create TClass deque<char*>;
//Current not generate in cintdll 
//#pragma create TClass deque<string>;

