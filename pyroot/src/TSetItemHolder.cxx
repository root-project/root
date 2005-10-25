// Author: Wim Lavrijsen, Oct 2005

// Bindings
#include "PyROOT.h"
#include "TSetItemHolder.h"
#include "Executors.h"

// ROOT
#include "TFunction.h"


//- protected members --------------------------------------------------------
Bool_t PyROOT::TSetItemHolder::InitExecutor_( TExecutor*& executor )
{
// basic call will do
   if ( ! TMethodHolder::InitExecutor_( executor ) )
      return kFALSE;

// check to make sure we're dealing with a RefExecutor
   if ( ! dynamic_cast< TRefExecutor* >( executor ) ) {
      PyErr_Format( PyExc_NotImplementedError,
         "no __setitem__ handler for return type (%s)", GetMethod()->GetReturnTypeName() );
      return kFALSE;
   }

   return kTRUE;
}


//- constructor --------------------------------------------------------------
PyROOT::TSetItemHolder::TSetItemHolder( TClass* klass, TMethod* method ) :
      TMethodHolder( klass, method )
{
}


//____________________________________________________________________________
PyObject* PyROOT::TSetItemHolder::FilterArgs( ObjectProxy*& self, PyObject* args, PyObject* kwds )
{
   int nArgs = PyTuple_GET_SIZE( args );
   if ( nArgs <= 1 ) {
      PyErr_SetString( PyExc_TypeError, "insufficient arguments to __setitem__" );
      return 0;
   }

// strip the last element of args to be used on return
   ((TRefExecutor*)GetExecutor())->SetAssignable( PyTuple_GET_ITEM( args, nArgs - 1 ) );
   PyObject* subset = PyTuple_GetSlice( args, 0, nArgs - 1 );
   PyObject* result = TMethodHolder::FilterArgs( self, subset, kwds );
   Py_DECREF( subset );
   return result;
}
