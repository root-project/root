#include "TObject.h"
#include "MemberMyObject.h"

class MyObject : public TObject {
public:
	MyObject() : memberMyObject()  {}
	MyObject(int allocated, int filled) : memberMyObject(allocated, filled)  {}
	MemberMyObject memberMyObject;
	ClassDef(MyObject, 1)
};
