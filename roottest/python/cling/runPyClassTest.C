#include "TPython.h"

// This test is no longer one script as it once was: Cling compiles code
// within a single macro, so there is a problem of initialization of the
// python interpreter, and of call order: the python classes need to be
// fully loaded into C++ before they can be used in declarations. This is
// different from the line-to-line behaviour of CINT (and the ROOT CLI).

void runPyClassTest() {

// load a python class and test its use
   TPython::LoadMacro( "MyPyClass.py" );
   gROOT->ProcessLine( ".x PyClassTest1.C" );

// load another python class and test it (note that this test was setup for
// a CINT-specific problem of not being able to build closures; that is in
// principle a non-issue with Cling)
   TPython::LoadMacro( "MyOtherPyClass.py" );
   gROOT->ProcessLine( ".x PyClassTest2.C" );

// test derivation of C++ classes from Python classes
// make sure that MyModule is in the path
   TPython::Exec("import sys");
   TPython::Exec("sys.path.append('./')");
   TPython::Import( "MyModule" );
   gROOT->ProcessLine( ".L PyClassTest3.C" );
   gROOT->ProcessLine( ".x PyClassTest4.C" );
}
