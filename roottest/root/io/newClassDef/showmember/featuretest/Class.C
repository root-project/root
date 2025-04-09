#include "Class.h"

void Class() {
  MyClass m;
  m.GetPublic();
}


bool MyClass::GetProtected() {
  cerr << "Properly run the protected function!" << endl;
  return true;
};
