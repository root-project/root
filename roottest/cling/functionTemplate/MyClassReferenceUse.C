#include "MyClassReferenceUse.h"

MyClass& GetMyClassReference()
{
	static MyClass m;
	return m;
}
