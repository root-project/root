// Author: Wim Lavrijsen, Apr 2005

// Bindings
#include "PyROOT.h"
#include "FunctionHolder.h"


//- constructor -----------------------------------------------------------------
PyROOT::FunctionHolder::FunctionHolder( TFunction* function ) :
      MethodHolder( function )
{
}


//- public members --------------------------------------------------------------
PyObject* PyROOT::FunctionHolder::operator()( ObjectProxy* self, PyObject* args, PyObject* kwds )
{
// setup as necessary
   if ( ! Initialize() )
      return 0;                              // important: 0, not Py_None
   
// fetch self, verify, and put the arguments in usable order
   if ( ! ( args = FilterArgs( self, args, kwds ) ) )
      return 0;
      
// translate the arguments
   bool bConvertOk = SetMethodArgs( args );
   Py_DECREF( args );
   
   if ( bConvertOk == false )
      return 0;
   
// execute function
   return Execute( 0 );
}
