#define private public
#define protected public 
 
#include "Class.h"

bool ClassTrick() {
  MyClass m;
  return m.GetPublic();
}
