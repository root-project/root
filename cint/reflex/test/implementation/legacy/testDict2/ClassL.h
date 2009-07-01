#ifndef DICT2_CLASSL_H
#define DICT2_CLASSL_H

#include "ClassK.h"

class ClassL: public ClassK {
public:
   ClassL(): fL('l') {}

   virtual ~ClassL() {}

   int
   l() { return fL; }

   void
   setL(int v) { fL = v; }

private:
   int fL;
};


#endif // DICT2_CLASSL_H
