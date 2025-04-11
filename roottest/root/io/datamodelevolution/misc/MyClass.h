
#include "TObject.h"
#include "TObjArray.h"

class MyClass : public TObject
{
public:

	MyClass() : TObject(), fArray() {}
	void Add(TObject* obj) { if (obj != NULL) {fArray.Add(obj);} }
	Int_t Nentries() const { return fArray.GetEntriesFast(); }
	const TObject* Entry(Int_t i) { return fArray[i]; }
   TObjArray& Array() { return fArray; }
	virtual void Print(Option_t* option = "") const;

private:
	TObjArray fArray;
	
	ClassDef(MyClass, 1) // Class with an array.
};
