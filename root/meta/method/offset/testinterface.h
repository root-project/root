#ifndef TESTINTERF
#define TESTINTERF

#include "TObject.h"
#include "TClass.h"
#include "TMethod.h"

class TestInterface {
   int someMember;
public:
   TestInterface() {someMember=42;};
   virtual ~TestInterface() {};
   void foo() const;
   void fooVirtual() const;
   ClassDef(TestInterface,0);
};

#endif
