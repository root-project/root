// Bindings
#include "CPyCppyy.h"
#include "CPyCppyy/TPyException.h"


//______________________________________________________________________________
//                 C++ exception for throwing python exceptions
//                 ============================================
// Purpose: A C++ exception class for throwing python exceptions
//          through C++ code.
// Created: Apr, 2004, Scott Snyder, from the version in D0's python_util.
//
// Note: Don't be tempted to declare the virtual functions defined here
//       as inline.
//       If you do, you may not be able to properly throw these
//       exceptions across shared libraries.


//- constructors/destructor --------------------------------------------------
CPyCppyy::TPyException::TPyException()
{
// default constructor
}

CPyCppyy::TPyException::~TPyException() noexcept
{
// destructor
}


//- public members -----------------------------------------------------------
const char* CPyCppyy::TPyException::what() const noexcept
{
// Return reason for throwing this exception: a python exception was raised.
    return "python exception";
}
