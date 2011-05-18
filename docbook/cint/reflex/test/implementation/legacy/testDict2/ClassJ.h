#ifndef DICT2_CLASSJ_H
#define DICT2_CLASSJ_H

#include "ClassI.h"

class ClassJ: public ClassI {
public:
   ClassJ(): fJ('j') {}

   virtual ~ClassJ() {}

   int
   j() { return fJ; }

   void
   setJ(int v) { fJ = v; }

private:
   int fJ;
};


#endif // DICT2_CLASSJ_H
