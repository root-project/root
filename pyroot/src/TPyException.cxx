// @(#)root/pyroot:$Name:  $:$Id: TPyException.cxx,v 1.5 2007/01/17 07:32:39 brun Exp $
// Author: Scott Snyder, Apr 2004

// Bindings
#include "PyROOT.h"
#include "TPyException.h"


//______________________________________________________________________________
//                 C++ exception for throwing python exceptions
//                 ============================================
// Purpose: A C++ exception class for throwing python exceptions
//          through C++ code.
// Created: Apr, 2004, sss, from the version in D0's python_util.
//
// Note: Don't be tempted to declare the virtual functions defined here
//       as inline.
//       If you do, you may not be able to properly throw these
//       exceptions across shared libraries.


//- data ---------------------------------------------------------------------
ClassImp(PyROOT::TPyException)
void* PyROOT::TPyExceptionMagic = (PyObject*)1;
namespace {

   struct TManageMagic {  // ensures that no valid PyObject can occupy the magic
      TManageMagic() { PyROOT::TPyExceptionMagic = new int(1); }
      ~TManageMagic() { delete (int*)PyROOT::TPyExceptionMagic; }
   } manageMagic;

} // unnamed namespace


//- constructors/destructor --------------------------------------------------
PyROOT::TPyException::TPyException()
{
// default constructor
}

PyROOT::TPyException::~TPyException() throw()
{
// destructor
}


//- public members -----------------------------------------------------------
const char* PyROOT::TPyException::what() const throw()
{
// Return reason for throwing this exception: a python exception was raised.
   return "python exception";
}
