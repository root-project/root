#define VERSION 2
#include "ClassConv.h"
#ifdef __MAKECLING__
#pragma link C++ namespace MyLib;
#pragma link C++ class MyLib::Inside+;
#pragma link C++ class TopLevel+;
#pragma link C++ class MyLib::Typedefed+;
#pragma link C++ typedef Typedefed;
#endif
