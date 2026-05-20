#define VERSION 1
#include "MyClass.h"
#ifdef __ROOTCLING__
#pragma link C++ class MyClass+;
#pragma link C++ class Cont+;
#endif
