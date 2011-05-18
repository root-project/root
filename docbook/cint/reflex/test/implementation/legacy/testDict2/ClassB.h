#ifndef DICT2_CLASSB_H
#define DICT2_CLASSB_H

#include "ClassA.h"

class ClassB: virtual public ClassA {
public:
   class Ambigous {};

   ClassB(): fB('b') {}

   virtual ~ClassB() {}

   int
   funWithManyArgs(int i0,
                   int i1,
                   int i2,
                   int i3,
                   int i4,
                   int i5,
                   int i6,
                   int i7,
                   int i8,
                   int i9,
                   int i10,
                   int i11,
                   int i12,
                   int i13,
                   int i14,
                   int i15,
                   int i16,
                   int i17,
                   int i18,
                   int i19) {
      return i0 + i1 + i2 + i3 + i4 + i5 + i6 + i7 + i8 + i9 + i10 + i11 + i12 + i13 + i14 + i15 + i16 + i17 + i18 + i19;
   }


   int
   b() { return fB; }

   void
   setB(int v) { fB = v; }

private:
   int fB;
};


#endif // DICT2_CLASSB_H
