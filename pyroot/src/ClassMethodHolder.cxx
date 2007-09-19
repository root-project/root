// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Aug 2004

// Bindings
#include "PyROOT.h"
#include "ClassMethodHolder.h"


//- constructors/destructor -----------------------------------------------------
PyROOT::TClassMethodHolder::TClassMethodHolder( TClass* klass, TFunction* method ) :
      TMethodHolder( klass, method )
{
}


//- public members --------------------------------------------------------------
PyObject* PyROOT::TClassMethodHolder::operator()( ObjectProxy*, PyObject* args, PyObject* )
{
// setup as necessary
   if ( ! Initialize() )
      return 0;                              // important: 0, not Py_None

// translate the arguments
   if ( ! SetMethodArgs( args ) )
      return 0;                              // important: 0, not Py_None

// execute function
   return Execute( 0 );
}
