#ifndef CustomStreamClass_h
#define CustomStreamClass_h

#include "TObject.h"
#include <iostream>

class MyClass : public TObject
{
   Int_t x;
   
   public:
   MyClass(){};
   virtual ~MyClass(){};
   
   void PrintX() { std::cout << "x=" << x << std::endl; }
   void SetX(Int_t i){x=i;}
   
   ClassDef(MyClass,1)//MyClass      
};

#endif
