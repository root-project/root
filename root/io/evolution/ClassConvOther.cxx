#define VERSION 3
#include "ClassConv.h"
#ifdef __MAKECINT__
#pragma link C++ namespace OtherLib;
#pragma link C++ class OtherLib::Inside+;
#pragma link C++ class TopLevel+;
#endif
