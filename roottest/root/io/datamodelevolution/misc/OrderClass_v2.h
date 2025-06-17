#ifndef ORDERCLASS_H
#define ORDERCLASS_H

#include "TObject.h"

class MyClass : public TObject
{
public:

   MyClass() : TObject(), transientMember(false) { ver = 2; }

   void addSomeData() { }

   void Print(Option_t* option = "") const override;

private:
   int ver;
   int fArray[2];
   bool transientMember; //!

   ClassDefOverride(MyClass, 2)
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