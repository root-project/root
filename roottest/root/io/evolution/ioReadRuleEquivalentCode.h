#include "TObject.h"

class A:public TObject{

public:
   A(){}
   ClassDef(A,1)

};

class B:public TObject{

public:
   B(){}
   void initializeTransients(){transient_=2;}
   ClassDef(B,1)
private:
   double transient_; //!

};

