#include "TObject.h"
#include "BaseMyObject.h"

class MyObject : public TObject, public BaseMyObject {
public:
	MyObject() : BaseMyObject()  {}
	ClassDefOverride(MyObject, 1)
};
