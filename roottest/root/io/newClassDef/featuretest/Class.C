#include "Class.h"

#include <iostream>

void Class() {
  MyClass m;
  m.GetPublic();
}


bool MyClass::GetProtected() {
  std::cerr << "Properly run the protected function!" << std::endl;
  return true;
};
