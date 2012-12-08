// Author: Wim Lavrijsen, Apr 2005

// Bindings
#include "PyROOT.h"
#include "FunctionHolder.h"
#include "ObjectProxy.h"
#include "Adapters.h"

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
namespace PyROOT {

#ifdef PYROOT_USE_REFLEX
template<>
TFunctionHolder< ROOT::Reflex::Scope, ROOT::Reflex::Member >::TFunctionHolder(
      const ROOT::Reflex::Member& function ) :
   TMethodHolder< ROOT::Reflex::Scope, ROOT::Reflex::Member >( ROOT::Reflex::Scope(), function )
{
}
#endif

} // namespace PyROOT

template< class T, class M >
PyROOT::TFunctionHolder< T, M >::TFunctionHolder( const M& function ) :
      TMethodHolder< T, M >( GetGlobalNamespace().GetClass(), function )
{
}

template< class T, class M >
PyROOT::TFunctionHolder< T, M >::TFunctionHolder( const T& scope, const M& function ) :
      TMethodHolder< T, M >( scope, function )
{
}

//- public members --------------------------------------------------------------
template< class T, class M >
PyObject* PyROOT::TFunctionHolder< T, M >::FilterArgs(
      ObjectProxy*& self, PyObject* args, PyObject* )
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
template< class T, class M >
PyObject* PyROOT::TFunctionHolder< T, M >::operator()(
      ObjectProxy* self, PyObject* args, PyObject* kwds, Long_t user, Bool_t release_gil )
{
// preliminary check in case keywords are accidently used (they are ignored otherwise)
   if ( kwds != 0 && PyDict_Size( kwds ) ) {
      PyErr_SetString( PyExc_TypeError, "keyword arguments are not yet supported" );
      return 0;
   }

// setup as necessary
   if ( ! this->Initialize() )
      return 0;                              // important: 0, not Py_None

// reorder self into args, if necessary
   if ( ! ( args = this->FilterArgs( self, args, kwds ) ) )
      return 0;

// translate the arguments
   Bool_t bConvertOk = this->SetMethodArgs( args, user );
   Py_DECREF( args );

   if ( bConvertOk == kFALSE )
      return 0;                              // important: 0, not Py_None

// execute function
   return this->Execute( 0, release_gil );
}

//____________________________________________________________________________
template class PyROOT::TFunctionHolder< PyROOT::TScopeAdapter, PyROOT::TMemberAdapter >;
#ifdef PYROOT_USE_REFLEX
template class PyROOT::TFunctionHolder< ROOT::Reflex::Scope, ROOT::Reflex::Member >;
#endif
