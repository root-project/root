#ifndef InheritMulti_H
#define InheritMulti_H

#include "TObject.h"
#include <stdio.h>

// Test for multi-inheritance objects.

class MyTop {
public:
   int t;
   MyTop() {}
   MyTop(int it) : t(it) {}
   virtual ~MyTop() {}
   ClassDef(MyTop,1)
};

class MyMulti : public TObject, public MyTop {
public:
   float m;
   MyMulti() {}
   MyMulti(int it, float im) : TObject(),MyTop(it),m(im) {}
   ~MyMulti() override {}
   ClassDefOverride(MyMulti,1)
};

class MyInverseMulti : public MyTop, public TObject {
public:
   int i;
   MyInverseMulti() {}
   MyInverseMulti(int it, int ii) : MyTop(it),TObject(),i(ii) {}
   ~MyInverseMulti() override {}
   ClassDefOverride(MyInverseMulti,1)
};


class A : public TObject {
public:
   int a;
   A() {}
   A(int ia) : a(ia) {}
   ~A() override {}

   ClassDefOverride(A,1)
};


class B {
public:
   B() :  b(0), a(0) {}
   B(int ia, float ib) : b(ib) { a = new A(ia); }
   virtual ~B() {}

   float b;
   A *a;

   void Dump()
   {
      fprintf(stderr,"Printing object of type \"B\" at %p\n",this);
      fprintf(stderr,"\tb = %f\n",b);
      fprintf(stderr,"\ta = 0x%p\n",a);
      if (a) fprintf(stderr,"\t\ta = %d\n",a->a);
   }
   ClassDef(B,1)
};


bool InheritMulti_driver();

#endif
