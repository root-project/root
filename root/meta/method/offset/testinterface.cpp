#include "testinterface.h"

ClassImp(TestInterface);

void TestInterface::foo() const {
   //printf("<TestInterface>Performing dynamic_cast of thisptr %lx to TObject inside TestInterface...\n", (Long_t) this);
   //const TObject* obj = dynamic_cast<const TObject*>(this);
   //printf("<TestInterface>Result is %lx\n", (Long_t) obj);

   Long_t This = (Long_t)this;
   const TObject* obj = dynamic_cast<const TObject*>(this);
   Long_t tobj = (Long_t)obj;
   printf("<TestInterface>Performing dynamic_cast of thisptr to TObject inside TestInterface...\n");
   printf("<TestInterface>The difference is %lx\n", This - tobj );

   if (obj == NULL) {
      printf("***THIS SHOULD NOT HAPPEN!***\n");
   }
   printf("<TestInterface>Member is %d, should be 42\n", someMember);
}

void TestInterface::fooVirtual() const {
   foo();
}

