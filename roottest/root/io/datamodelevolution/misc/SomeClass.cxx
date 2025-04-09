
#include "TObject.h"
#include "TString.h"

class SomeClass : public TObject
{
public:

	SomeClass(const char* name = "no name") : TObject(), fName(name) {}
	virtual const char* GetName() const { return fName; }
	void SetName(const char* name) { fName = name; }

private:
	TString fName;
	
	ClassDef(SomeClass, 1) // Class with a name.
};

ClassImp(SomeClass)
