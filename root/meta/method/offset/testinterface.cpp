#include "testinterface.h"

#include "testobject.h"

ClassImp(TestInterface);

void TestInterface::foo() const {
   //printf("<TestInterface>Performing dynamic_cast of thisptr %lx to TObject inside TestInterface...\n", (Long_t) this);
   //const TObject* obj = dynamic_cast<const TObject*>(this);
   //printf("<TestInterface>Result is %lx\n", (Long_t) obj);

   TestObj checkObj;
   const TestInterface* checkObjAsTestIface = dynamic_cast<const TestInterface*>(&checkObj);
   const TObject* checkObjAsTObj = dynamic_cast<const TObject*>(&checkObj);
   Long_t checkTheRealOffsetDiff = ((Long_t)checkObjAsTestIface) - ((Long_t)checkObjAsTObj);

   Long_t This = (Long_t)this;
   const TObject* obj = dynamic_cast<const TObject*>(this);
   Long_t tobj = (Long_t)obj;
   printf("<TestInterface>Performing dynamic_cast of thisptr to TObject inside TestInterface...\n");
   printf("<TestInterface>The difference is %s\n",
          This - tobj == checkTheRealOffsetDiff  ? "expected." : "UNEXPECTED!");

   if (obj == NULL) {
      printf("***THIS SHOULD NOT HAPPEN!***\n");
   }
   printf("<TestInterface>Member is %d, should be 42\n", someMember);
}

void TestInterface::fooVirtual() const {
   foo();
}

