#ifdef PYROOT_USE_REFLEX

// Bindings
#include "PyROOT.h"
#include "TRflxCallback.h"
#include "RootWrapper.h"

// ROOT
#include "Reflex/Base.h"
#include "Reflex/Member.h"
#include "Reflex/Scope.h"
#include "Reflex/Type.h"


//- data ---------------------------------------------------------------------
std::auto_ptr< ROOT::Reflex::ICallback > PyROOT::TRflxCallback::gCallback;


//- static methods -----------------------------------------------------------
PyObject* PyROOT::TRflxCallback::Enable()
{
// Setup callback to receive Reflex notification of new types

// TODO: add python warning message if gCallback, set and return 0
   gCallback.reset( new TRflxCallback );

   Py_INCREF( Py_True );
   return Py_True;
}

//____________________________________________________________________________
PyObject* PyROOT::TRflxCallback::Disable()
{
// Remove notification from Reflex
   if ( ! gCallback.get() ) {
   // TODO: add python error message and return 0
      Py_INCREF( Py_False );
      return Py_False;
   }

   gCallback.reset();

   Py_INCREF( Py_True );
   return Py_True;
}


//- constructor and destructor -----------------------------------------------
PyROOT::TRflxCallback::TRflxCallback()
{
// Install self as callback to receive notifications from Reflex
   ROOT::Reflex::InstallClassCallback( this );
}

//____________________________________________________________________________
PyROOT::TRflxCallback::~TRflxCallback()
{
// Remove self as callback to receive notifications from Reflex
   ROOT::Reflex::UninstallClassCallback( this );
}


//- public members -----------------------------------------------------------
void PyROOT::TRflxCallback::operator() ( const ROOT::Reflex::Type& t )
{
   PyObject* pyclass = MakeRootClassFromString< ROOT::Reflex::Scope,\
      ROOT::Reflex::Base, ROOT::Reflex::Member >( t.Name( ROOT::Reflex::SCOPED ) );
   Py_XDECREF( pyclass );
}

//____________________________________________________________________________
void PyROOT::TRflxCallback::operator() ( const ROOT::Reflex::Member& /* m */ )
{
   /* nothing yet */
}

#endif // PYROOT_USE_REFLEX
