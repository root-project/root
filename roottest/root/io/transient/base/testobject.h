#ifndef TESTOBJ
#define TESTOBJ

#include "TObject.h"

#include <map>

class First;
class Second;

class TestObj : public TObject {
protected:
    Double_t lStorage;

   std::map<int,First*>  fFirst;
   std::map<int,Second*> fSecond; //!
public:
    TestObj();
    virtual ~TestObj() {};
    virtual void store(Double_t foo);
    virtual Double_t retrieve() const;
    ClassDefOverride(TestObj,0);
};

#endif
