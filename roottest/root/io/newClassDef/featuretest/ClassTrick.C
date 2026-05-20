#define private public
#define protected public

#include "Class.h"

bool ClassTrick() {
  MyClass m;
#ifndef _WIN32
  // This is known to fail on windows
  return m.GetPublic();
#else
  (void) m; // avoid unused warnings
  return false;
#endif
}

// This file gets included in the dictionary and the interpreter,
// make sure we clean up behind us!
#undef private
#undef protected
