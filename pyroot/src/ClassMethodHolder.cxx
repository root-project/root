// @(#)root/pyroot:$Name:  $:$Id: ClassMethodHolder.cxx,v 1.2 2005/03/04 07:44:11 brun Exp $
// Author: Wim Lavrijsen, Aug 2004

// Bindings
#include "PyROOT.h"
#include "ClassMethodHolder.h"


//- constructor -----------------------------------------------------------------
PyROOT::ClassMethodHolder::ClassMethodHolder( TClass* klass, TMethod* method ) :
      MethodHolder( klass, method )
{
}


//- public members --------------------------------------------------------------
PyObject* PyROOT::ClassMethodHolder::operator()( ObjectProxy*, PyObject* args, PyObject* )
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
