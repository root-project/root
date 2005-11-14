#ifndef DICT2_CLASSH_H
#define DICT2_CLASSH_H

#include "ClassG.h"

class ClassH: public ClassG {
 public:
  ClassH() : fH('h') {}
  virtual ~ClassH() {}
  int h() { return fH; }
  void setH(int v) { fH = v; }
 private:
  int fH;
};


#endif // DICT2_CLASSH_H
