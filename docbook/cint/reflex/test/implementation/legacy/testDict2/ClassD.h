#ifndef DICT2_CLASSD_H
#define DICT2_CLASSD_H

#include "ClassB.h"

class ClassD: virtual public ClassB {
public:
   ClassD(): fD('d') {}

   virtual ~ClassD() {}

   int
   d() { return fD; }

   void
   setD(int v) { fD = v; }

private:
   int fD;
};


#endif // DICT2_CLASSD_H
