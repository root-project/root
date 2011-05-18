#ifndef DICT2_CLASSK_H
#define DICT2_CLASSK_H

#include "ClassC.h"

class ClassK: virtual public ClassC {
public:
   ClassK(): fK('k') {}

   virtual ~ClassK() {}

   int
   k() { return fK; }

   void
   setK(int v) { fK = v; }

private:
   int fK;
};


#endif // DICT2_CLASSK_H
