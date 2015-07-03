#include "TPython.h"
#include <iostream>

void PyClassTest4() {
// python classes live in "MyPyClass.py" and "MyModule.py", which must have been
// loaded in a separate line or macro, or this one won't compile

// test access to C++ classes derived from python classes
   Derived1* a = (Derived1*)TPython::Eval( "ROOT.Derived1( 42 )" );
   std::cout << "Derived1 (42):  " << a->fVal << std::endl;

   Derived2* b = (Derived2*)TPython::Eval( "ROOT.Derived2( 5.0 )" );
   std::cout << "Derived2 (5.0): " << b->fVal << std::endl;
}
