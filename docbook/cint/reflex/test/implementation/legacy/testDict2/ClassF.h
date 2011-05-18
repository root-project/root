#ifndef DICT2_CLASSF_H
#define DICT2_CLASSF_H

#include "ClassD.h"
#include "ClassE.h"

class ClassF: virtual public ClassD,
   virtual public ClassE {
public:
   ClassF(): fF('f') {}

   virtual ~ClassF() {}

   int
   f() { return fF; }

   void
   setF(int v) { fF = v; }

private:
   int fF;
};


#endif // DICT2_CLASSF_H
