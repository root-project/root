#include <list>
#include <string>
#ifndef __hpux
using namespace std;
#endif

#pragma link C++ class list<int>;
#pragma link C++ class list<long>;
#pragma link C++ class list<float>;
#pragma link C++ class list<double>;
#pragma link C++ class list<void*>;
#pragma link C++ class list<char*>;
#pragma link C++ class list<string>;

