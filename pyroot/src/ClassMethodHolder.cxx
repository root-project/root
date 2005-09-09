// @(#)root/pyroot:$Name:  $:$Id: ClassMethodHolder.cxx,v 1.3 2005/03/30 05:16:19 brun Exp $
// Author: Wim Lavrijsen, Aug 2004

// Bindings
#include "PyROOT.h"
#include "ClassMethodHolder.h"


//- constructor -----------------------------------------------------------------
PyROOT::TClassMethodHolder::TClassMethodHolder( TClass* klass, TMethod* method ) :
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
