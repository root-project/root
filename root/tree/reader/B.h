
#ifndef B_HEADER
#define B_HEADER

#include "TObject.h"

class B : public TObject {
public:
	Int_t dummy;

	B() : dummy(0) {}
	B(Int_t dummyArg) : dummy(dummyArg) {}

	void SetDummy(Int_t dummyArg) { dummy = dummyArg; }

	Int_t GetDummy() const { return dummy; }

	ClassDef(B, 1);
};

#endif
