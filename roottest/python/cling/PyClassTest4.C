#include "TPython.h"
#include <iostream>

void PyClassTest4()
{
   // python classes live in "MyPyClass.py" and "MyModule.py", which must have been
   // loaded in a separate line or macro, or this one won't compile

   // test access to C++ classes derived from python classes
   std::any result;
   TPython::Exec(R"(val = ROOT.Derived1( 42 ); _anyresult = ROOT.std.make_any["Derived1*", "Derived1*"](val))", &result);
   std::cout << "Derived1 (42):  " << std::any_cast<Derived1*>(result)->fVal << std::endl;

   TPython::Exec(R"(val = ROOT.Derived2( 5.0 ); _anyresult = ROOT.std.make_any["Derived2*", "Derived2*"](val))", &result);
   std::cout << "Derived2 (5.0): " << std::any_cast<Derived2*>(result)->fVal << std::endl;
}
