#include "TObject.h"

class A:public TObject{

public:
   A(){}
   ClassDefOverride(A,1)

};

class B:public TObject{

public:
   B(){}
   void initializeTransients(){transient_=2;}
   ClassDefOverride(B,1)
private:
   double transient_; //!

};

