#include "testobject.h"

ClassImp(TestObj);

TestObj::TestObj()
{
}

void TestObj::store(Double_t foo)
{
   lStorage = foo;
   printf("Stored %5.2lf!\r\n", foo);
}

Double_t TestObj::retrieve() const
{
   Double_t foo = lStorage;
   printf("Retrieved %5.2lf!\r\n", foo);
   return foo;
}
