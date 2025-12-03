#ifndef TESTOBJDERIVED
#define TESTOBJDERIVED

#include "TObject.h"
#include "testobject.h"

class TestObjDerived : public TestObj {
protected:
    Double_t lAnotherStorage;
public:
    TestObjDerived();
    virtual ~TestObjDerived() {};
    virtual void storeInAnother(Double_t foo);
    virtual Double_t retrieveFromAnother() const;
    ClassDefOverride(TestObjDerived,1);
};

#endif
