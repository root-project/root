// Author: Wim Lavrijsen, Oct 2005

// Bindings
#include "PyROOT.h"
#include "TSetItemHolder.h"
#include "Executors.h"
#include "Adapters.h"


//- protected members --------------------------------------------------------
template< class T, class M >
Bool_t PyROOT::TSetItemHolder< T, M >::InitExecutor_( TExecutor*& executor )
{
// basic call will do
   if ( ! TMethodHolder< T, M >::InitExecutor_( executor ) )
      return kFALSE;

// check to make sure we're dealing with a RefExecutor
   if ( ! dynamic_cast< TRefExecutor* >( executor ) ) {
      PyErr_Format( PyExc_NotImplementedError,
         "no __setitem__ handler for return type (%s)",
         this->GetMethod().TypeOf().ReturnType().Name( ROOT::Reflex::Q | ROOT::Reflex::S ).c_str() );
      return kFALSE;
   }

   return kTRUE;
}


//- constructor --------------------------------------------------------------
template< class T, class M >
PyROOT::TSetItemHolder< T, M >::TSetItemHolder( const T& klass, const M& method ) :
      TMethodHolder< T, M >( klass, method )
{
}


//____________________________________________________________________________
template< class T, class M >
PyObject* PyROOT::TSetItemHolder< T, M >::FilterArgs(
      ObjectProxy*& self, PyObject* args, PyObject* kwds )
{
// Prepare executor with a buffer for the return value.

   int nArgs = PyTuple_GET_SIZE( args );
   if ( nArgs <= 1 ) {
      PyErr_SetString( PyExc_TypeError, "insufficient arguments to __setitem__" );
      return 0;
   }

// strip the last element of args to be used on return
   ((TRefExecutor*)this->GetExecutor())->SetAssignable( PyTuple_GET_ITEM( args, nArgs - 1 ) );
   PyObject* subset = PyTuple_GetSlice( args, 0, nArgs - 1 );

// see whether any of the arguments is a tuple itself
   Py_ssize_t realsize = 0;
   for ( int i = 0; i < nArgs - 1; ++i ) {
      PyObject* item = PyTuple_GetItem( subset, i );
      realsize += PyTuple_Check( item ) ? PyTuple_GET_SIZE( item ) : 1;
   }

// unroll any tuples, if present in the arguments
   PyObject* unrolled = 0;
   if ( realsize != nArgs - 1 ) {
      unrolled = PyTuple_New( realsize );

      int current = 0;
      for ( int i = 0; i < nArgs - 1; ++i, ++current ) {
         PyObject* item = PyTuple_GetItem( subset, i );
         if ( PyTuple_Check( item ) ) {
            for ( int j = 0; j < PyTuple_GET_SIZE( item ); ++j, ++current ) {
               PyObject* subitem = PyTuple_GetItem( item, j );
               Py_INCREF( subitem );
               PyTuple_SetItem( unrolled, current, subitem );
            }
         } else {
            Py_INCREF( item );
            PyTuple_SetItem( unrolled, current, item );
         }
      }
   }

// actual call into C++
   PyObject* result = TMethodHolder< T, M >::FilterArgs( self, unrolled ? unrolled : subset, kwds );
   Py_XDECREF( unrolled );
   Py_DECREF( subset );
   return result;
}

//____________________________________________________________________________
template class PyROOT::TSetItemHolder< PyROOT::TScopeAdapter, PyROOT::TMemberAdapter >;
#ifdef PYROOT_USE_REFLEX
template class PyROOT::TSetItemHolder< ROOT::Reflex::Scope, ROOT::Reflex::Member >;
#endif
