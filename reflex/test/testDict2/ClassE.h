#ifndef DICT2_CLASSE_H
#define DICT2_CLASSE_H

#include "ClassC.h"

class ClassE: virtual public ClassC {
 public:
  ClassE() : fE('e') {}
  virtual ~ClassE() {}
  int e() { return fE; }
  void setE(int v) { fE = v; }
 private:
  int fE;
};


#endif // DICT2_CLASSE_H
