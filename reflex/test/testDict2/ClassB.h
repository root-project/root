#ifndef DICT2_CLASSB_H
#define DICT2_CLASSB_H

#include "ClassA.h"

class ClassB : virtual public ClassA {
 public:
  ClassB() : fB('b') {}
  virtual ~ClassB() {}
  int b() { return fB; }
  void setB(int v) { fB = v; }
 private:
  int fB;
};


#endif // DICT2_CLASSB_H
