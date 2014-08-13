#ifndef TESTOBJ
#define TESTOBJ

#include "TObject.h"

class TestObj : public TObject {
protected:
	Double_t lStorage;
public:
	TestObj();
	virtual ~TestObj() {};
	virtual void store(Double_t foo);
	virtual Double_t retrieve() const;
	ClassDef(TestObj,0);
};

#endif
