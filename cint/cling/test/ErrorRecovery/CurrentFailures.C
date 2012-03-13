// RUN: cat %s | %cling -I%p | FileCheck %s
// XFAIL: *

// Class defined in the wrapper. Most probably the DeclExtractor pulls it out
// in a wrong way.
class MyClass {
  struct {
    int a;
    error_here; // expected-error {{C++ requires a type specifier for all declarations}}
  };
};


#include "Overloads.h"

#include "SubsequentDecls.h"
