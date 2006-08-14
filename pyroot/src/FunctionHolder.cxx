// Author: Wim Lavrijsen, Apr 2005

// Bindings
#include "PyROOT.h"
#include "FunctionHolder.h"
#include "ObjectProxy.h"

// ROOT
#include "TClass.h"


//- data and local helpers ---------------------------------------------------
namespace {

   TClassRef GetGlobalNamespace() {
      static TClass c;
      return &c;
   }

} // unnamed namespace


//- constructor -----------------------------------------------------------------
PyROOT::TFunctionHolder::TFunctionHolder( TFunction* function ) :
      TMethodHolder( GetGlobalNamespace(), function )
{
}

//- public members --------------------------------------------------------------
PyObject* PyROOT::TFunctionHolder::FilterArgs( ObjectProxy*& self, PyObject* args, PyObject* )
{
// no self means called as a free function; all ok
   if ( ! self ) {
      Py_INCREF( args );
      return args;
   }

// otherwise, add self as part of the function arguments (means bound member)
   Py_ssize_t sz = PyTuple_GET_SIZE( args );
   PyObject* newArgs = PyTuple_New( sz + 1 );
   for ( int i = 0; i < sz; ++i ) {
      PyObject* item = PyTuple_GET_ITEM( args, i );
      Py_INCREF( item );
      PyTuple_SET_ITEM( newArgs, i + 1, item );
   }

   Py_INCREF( self );
   PyTuple_SET_ITEM( newArgs, 0, (PyObject*)self );

   return newArgs;
}

//____________________________________________________________________________
PyObject* PyROOT::TFunctionHolder::operator()( ObjectProxy* self, PyObject* args, PyObject* kwds )
{
// setup as necessary
   if ( ! Initialize() )
      return 0;                              // important: 0, not Py_None

// reorder self into args, if necessary
   if ( ! ( args = FilterArgs( self, args, kwds ) ) )
      return 0;

// translate the arguments
   Bool_t bConvertOk = SetMethodArgs( args );
   Py_DECREF( args );

   if ( bConvertOk == kFALSE )
      return 0;                              // important: 0, not Py_None

// execute function
   return Execute( 0 );
}
