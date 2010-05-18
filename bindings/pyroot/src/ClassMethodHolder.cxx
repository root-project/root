// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Aug 2004

// Bindings
#include "PyROOT.h"
#include "ClassMethodHolder.h"
#include "Adapters.h"


//- constructors/destructor -----------------------------------------------------
template< class T, class M >
PyROOT::TClassMethodHolder< T, M >::TClassMethodHolder( const T& klass, const M& method ) :
      TMethodHolder< T, M >( klass, method )
{
}


//- public members --------------------------------------------------------------
template< class T, class M >
PyObject* PyROOT::TClassMethodHolder< T, M >::operator()(
      ObjectProxy*, PyObject* args, PyObject* kwds, Long_t user )
{
// preliminary check in case keywords are accidently used (they are ignored otherwise)
   if ( kwds != 0 && PyDict_Size( kwds ) ) {
      PyErr_SetString( PyExc_TypeError, "keyword arguments are not yet supported" );
      return 0;
   }

// setup as necessary
   if ( ! this->Initialize() )
      return 0;                              // important: 0, not Py_None

// translate the arguments
   if ( ! this->SetMethodArgs( args, user ) )
      return 0;                              // important: 0, not Py_None

// execute function
   return this->Execute( 0 );
}

//____________________________________________________________________________
template class PyROOT::TClassMethodHolder< PyROOT::TScopeAdapter, PyROOT::TMemberAdapter >;
#ifdef PYROOT_USE_REFLEX
template class PyROOT::TClassMethodHolder< ROOT::Reflex::Scope, ROOT::Reflex::Member >;
#endif
