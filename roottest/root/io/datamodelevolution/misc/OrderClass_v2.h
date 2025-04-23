#ifndef ORDERCLASS_H
#define ORDERCLASS_H

#include "TObject.h"

class MyClass : public TObject
{
public:

   //MyClass() : TObject() {ver = 2; fArray[0] = fArray[1] = -1; }
   MyClass() : TObject() , transientMember(0) {ver = 2; }

   void addSomeData() {

   }
   virtual void Print(Option_t* option = "") const;

private:
   int ver;
   int fArray[2];
   bool transientMember; //!

   ClassDef(MyClass, 2)
};

#ifdef __MAKECINT__
#pragma link C++ class MyClass+;

#pragma read sourceClass="MyClass" version="[1]" source="int ver" \
  targetClass="MyClass" target="ver" \
  include="iostream" \
  code="{ std::cout << \"rule reading class version: \" << onfile.ver << \"\\n\"; ver = 99; }"

#pragma read sourceClass="MyClass" version="[1-]" source="" \
  targetClass="MyClass" target="transientMember" \
  code="{ std::cout << \"rule setting transientMember\\n \"; transientMember = 1; }"

#endif

#endif