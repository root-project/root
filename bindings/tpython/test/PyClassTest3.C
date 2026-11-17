#include "TPython.h"

// python classes live in "MyPyClass.py" and "MyModule.py", which must have been
// loaded in a separate line or macro, or this one won't compile

// test derivabilty of python classes
class Derived1 : public MyPyClass {
public:
   Derived1( int val ) : fVal( val ) { }
   int fVal;
};

class Derived2 : public MyModule::MyPyClass {
public:
   Derived2( double val ) : fVal( val ) { }
   double fVal;
};
