#define VERSION 2
#include "MyClass.h"
#ifdef __ROOTCLING__
#pragma link C++ class MyClass+;
#pragma link C++ class Cont+;
#pragma read sourceClass="MyClass" targetClass="MyClass" \
  source="int n; float arr[20];" version="[1]" target="arr" \
  code="{ if (onfile.n > 0) { arr = new float[onfile.n]; for(int i=0; i<onfile.n; ++i) arr[i] = onfile.arr[i]; } else { arr = nullptr; } }"
#endif
