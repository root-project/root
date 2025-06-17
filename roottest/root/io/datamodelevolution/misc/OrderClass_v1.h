#ifndef ORDERCLASS_H
#define ORDERCLASS_H

#include "TObject.h"

class MyClass : public TObject
{
public:

   MyClass() : TObject(), inOneOnly(0), transientMember(false) { ver = 1; }

   void addSomeData() { }

   void Print(Option_t* option = "") const override;

private:
   // rule is only applied if fArray is first!
   int ver;
   std::vector<int> fArray;
   int inOneOnly; // !
   bool transientMember; //!

   ClassDefOverride(MyClass, 1)
};

#ifdef __MAKECINT__
#pragma link C++ class MyClass+;
#endif

#endif