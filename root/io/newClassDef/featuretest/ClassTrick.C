#define private public
#define protected public 
 
#include "Class.h"

bool ClassTrick() {
  MyClass m;
#ifndef _WIN32
  // This is known to fail on windows
  return m.GetPublic();
#else
  return false;
#endif
}
