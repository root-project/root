#ifndef TESTOBJ
#define TESTOBJ

#include "TObject.h"
#include "TClass.h"
#include "TMethod.h"

#include "testinterface.h"

class TestObj : public TObject, public TestInterface {
public:
   TestObj() {};
   virtual ~TestObj() {};
   ClassDefOverride(TestObj,0);
};

#endif
