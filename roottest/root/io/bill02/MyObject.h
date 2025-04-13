#include "TObject.h"
#include "BaseMyObject.h"

class MyObject : public TObject, public BaseMyObject {
public:
	MyObject() : BaseMyObject()  {}
	ClassDef(MyObject, 1)
};
