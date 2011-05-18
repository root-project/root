#ifndef DICT2_CLASSG_H
#define DICT2_CLASSG_H

#include "ClassF.h"

class ClassG: public ClassF {
public:
   ClassG(): fG('g') {}

   virtual ~ClassG() {}

   int
   g() { return fG; }

   void
   setG(int v = 11) { fG = v; }

private:
   int fG;
};


#endif // DICT2_CLASSG_H
