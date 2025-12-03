#ifndef InheritMulti_H
#define InheritMulti_H

#include "TObject.h"

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

bool InheritMulti_driver();

#endif
