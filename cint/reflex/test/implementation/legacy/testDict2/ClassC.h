#ifndef DICT2_CLASSC_H
#define DICT2_CLASSC_H

#include "ClassA.h"

class ClassC: virtual public ClassA {
public:
   class PublicInner {
   public:
      class PublicInnerInner {};

   private:
      class PrivateInnerInner {};
   };


   ClassC(): fC('c') {}

   virtual ~ClassC() {}

   int
   c() { return fC; }

   void
   setC(int v) { fC = v; }

private:
   int fC;
};


#endif // DICT2_CLASSC_H
