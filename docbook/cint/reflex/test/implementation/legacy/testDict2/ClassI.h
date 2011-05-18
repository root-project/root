#ifndef DICT2_CLASSI_H
#define DICT2_CLASSI_H

#include "ClassE.h"
#include "ClassK.h"

class ClassI: public ClassE,
   public ClassK {
public:
   ClassI(): fI('i') {}

   virtual ~ClassI() {}

   int
   i() { return fI; }

   void
   setI(int v) { fI = v; }

private:
   int fI;
};


#endif // DICT2_CLASSI_H
