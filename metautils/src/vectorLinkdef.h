#include <string>
#include <vector>
#ifndef __hpux
using namespace std;
#endif

#pragma create TClass vector<bool>;
#pragma create TClass vector<char>;
#pragma create TClass vector<short>;
#pragma create TClass vector<long>;
#pragma create TClass vector<unsigned char>;
#pragma create TClass vector<unsigned short>;
#pragma create TClass vector<unsigned int>;
#pragma create TClass vector<unsigned long>;
#pragma create TClass vector<float>;
#pragma create TClass vector<double>;
#pragma create TClass vector<char*>;
#pragma create TClass vector<const char*>;
#pragma create TClass vector<string>;

#if (!(G__GNUC==3 && G__GNUC_MINOR==1)) && !defined(G__KCC) && (!defined(G__VISUAL) || G__MSC_VER<1300)
// gcc3.1,3.2 has a problem with iterator<void*,...,void&>
#pragma create TClass vector<void*>;
#endif
