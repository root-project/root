// Author: Wim Lavrijsen, Apr 2005

// Bindings
#include "PyROOT.h"
#include "FunctionHolder.h"


//- constructor -----------------------------------------------------------------
PyROOT::TFunctionHolder::TFunctionHolder( TFunction* function ) :
      TMethodHolder( function )
{
}


//- public members --------------------------------------------------------------
PyObject* PyROOT::TFunctionHolder::operator()( ObjectProxy* self, PyObject* args, PyObject* kwds )
{
// setup as necessary
   if ( ! Initialize() )
      return 0;                              // important: 0, not Py_None
   
// fetch self, verify, and put the arguments in usable order
   if ( ! ( args = FilterArgs( self, args, kwds ) ) )
      return 0;
      
// translate the arguments
   Bool_t bConvertOk = SetMethodArgs( args );
   Py_DECREF( args );
   
   if ( bConvertOk == kFALSE )
      return 0;
   
// execute function
   return Execute( 0 );
}
