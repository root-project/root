#ifndef ORDERCLASS_H
#define ORDERCLASS_H

#include "TObject.h"

class MyClass : public TObject
{
public:

   MyClass() : TObject(), inOneOnly(0),transientMember(0) { ver = 1;}

   void addSomeData() {
      //fArray.push_back(123);
      //fArray.push_back(456);
   }
   virtual void Print(Option_t* option = "") const;

private:
   // rule is only applied if fArray is first!
   int ver;
   std::vector<int> fArray;
   int inOneOnly; // !
   bool transientMember; //!

   ClassDef(MyClass, 1)
};

#ifdef __MAKECINT__
#pragma link C++ class MyClass+;
#endif

#endif