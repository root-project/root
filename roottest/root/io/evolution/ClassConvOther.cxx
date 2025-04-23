#define VERSION 3
#include "ClassConv.h"
#ifdef __MAKECINT__
#pragma link C++ namespace OtherLib;
#pragma link C++ class OtherLib::Inside+;
#pragma link C++ class TopLevel+;
#pragma link C++ class OtherLib::Typedefed+;
#pragma link C++ typedef Typedefed;
#endif
