// lib/dll_stl/str.h

#include <string>

#ifndef __hpux
using namespace std;
#endif

#if 0 && (__SUNPRO_CC>=1280)
#include "suncc5_string.h"  // obsolete
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
#pragma link C++ typedef string::value_type;

#pragma if (G__GNUC>=3 && G__GNUC_MINOR>=1) 
#pragma link C++ class string::iterator;
#pragma else
#pragma link C++ typedef string::iterator;
#pragma endif

#pragma link C++ typedef string::const_iterator;
#pragma link C++ typedef string::pointer;
#pragma link C++ typedef string::const_pointer;
#pragma link C++ typedef string::reference;
#pragma link C++ typedef string::difference_type;
#pragma link C++ typedef string::size_type;
#pragma link C++ typedef string::traits_type;
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


