// @(#)root/pyroot:$Name:  $:$Id: ClassMethodHolder.cxx,v 1.68 2005/01/28 05:45:41 brun Exp $
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
      return 0;                              // important: 0, not PyNone

// translate the arguments
   if ( ! SetMethodArgs( args ) )
      return 0;                              // important: 0, not PyNone

// execute function
   return Execute( 0 );
}
