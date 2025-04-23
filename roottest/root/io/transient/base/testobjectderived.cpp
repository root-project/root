#include "testobjectderived.h"

ClassImp(TestObj);

TestObjDerived::TestObjDerived() : TestObj()
{
}

void TestObjDerived::storeInAnother(Double_t foo)
{
   lStorage = foo;
   printf("Stored %5.2lf!\r\n", foo);
}

Double_t TestObjDerived::retrieveFromAnother() const
{
   Double_t foo = lStorage;
   printf("Retrieved %5.2lf!\r\n", foo);
   return foo;
}
