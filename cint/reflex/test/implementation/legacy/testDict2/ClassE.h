#ifndef DICT2_CLASSE_H
#define DICT2_CLASSE_H

#include "ClassC.h"

class ClassE: virtual public ClassC {
public:
   class PublicInner {
   public:
      class PublicInnerInner {};

   private:
      class PrivateInnerInner {};
   };

   class Ambigous {};

   ClassE(): fE('e') {}

   virtual ~ClassE() {}

   int
   e() { return fE; }

   void
   setE(int v) { fE = v; }

private:
   class EPrivateInner {};

   int fE;
};


#endif // DICT2_CLASSE_H
