#ifndef DICT2_CLASSN_H
#define DICT2_CLASSN_H

#include "ClassI.h"
#include "ClassL.h"

class ClassN: /* public ClassI, */ public ClassL {
public:
   ClassN(): fN('n') {}

   virtual ~ClassN() {}

   int
   n() { return fN; }

   void
   setN(int v) { fN = v; }

private:
   int fN;
};


#endif // DICT2_CLASSN_H
