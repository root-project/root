#define VERSION 2
#include "ClassConv.h"
#ifdef __MAKECINT__
#pragma link C++ namespace MyLib;
#pragma link C++ class MyLib::Inside+;
#pragma link C++ class TopLevel+;
#endif
