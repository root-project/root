// lib/dll_stl/str.h

#include <string>

#ifndef __hpux
using namespace std;
#endif

#if (__SUNPRO_CC>=1280)
#include "suncc5_string.h"
#endif

#ifdef __MAKECINT__
#pragma ifndef G__STRING_DLL
#pragma define G__STRING_DLL
#pragma endif
#pragma link C++ global G__STRING_DLL;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;

//#pragma link C++ basic_string<char, char_traits<char>, allocator<char> >;
//#pragma link C++ typedef string;

#pragma link C++ class string;
#ifndef G__OLDIMPLEMENTATION1598
#pragma link C++ function operator==(const string&,const string&);
#pragma link C++ function operator!=(const string&,const string&);
#pragma link C++ function operator<(const string&,const string&);
#pragma link C++ function operator>(const string&,const string&);
#pragma link C++ function operator<=(const string&,const string&);
#pragma link C++ function operator>=(const string&,const string&);
#pragma link C++ function operator+(const string&,const string&);
#endif
#endif


