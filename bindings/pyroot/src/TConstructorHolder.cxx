// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "TConstructorHolder.h"
#include "Adapters.h"
#include "Executors.h"
#include "ObjectProxy.h"
#include "TMemoryRegulator.h"

// ROOT
#include "TClass.h"
#include "TFunction.h"

// Standard
#include <string>


//- protected members --------------------------------------------------------
Bool_t PyROOT::TConstructorHolder::InitExecutor_( TExecutor*& executor )
{
// pick up special case new object executor
   executor = (gExecFactories[ "__init__" ])();
   return kTRUE;
}

//- constructors -------------------------------------------------------------
PyROOT::TConstructorHolder::TConstructorHolder( const TScopeAdapter& klass ) :
      TMethodHolder( klass, (TFunction*)0 )
{
}

//____________________________________________________________________________
PyROOT::TConstructorHolder::TConstructorHolder( const TScopeAdapter& klass, const TMemberAdapter& method ) :
      TMethodHolder( klass, method )
{
}

//- public members -----------------------------------------------------------
PyObject* PyROOT::TConstructorHolder::GetDocString()
{
// GetMethod() may return an empty function if this is just a special case place holder
   std::string clName = this->GetClass().Name();
   return PyROOT_PyUnicode_FromFormat( "%s::%s%s",
      clName.c_str(), clName.c_str(), this->GetMethod() ? this->GetSignatureString().c_str() : "()" );
}

//____________________________________________________________________________
PyObject* PyROOT::TConstructorHolder::operator()(
      ObjectProxy* self, PyObject* args, PyObject* kwds, Long_t user, Bool_t release_gil )
{
// preliminary check in case keywords are accidently used (they are ignored otherwise)
   if ( kwds != 0 && PyDict_Size( kwds ) ) {
      PyErr_SetString( PyExc_TypeError, "keyword arguments are not yet supported" );
      return 0;
   }

// do not allow instantiation of abstract classes
   if ( this->GetClass().IsAbstract() ) {
      PyErr_Format( PyExc_TypeError,
         "%s is abstract and can not be instantiated", this->GetClass().Name().c_str() );
      return 0;
   }

// setup as necessary
   if ( ! this->Initialize() )
      return 0;                              // important: 0, not Py_None

// fetch self, verify, and put the arguments in usable order
   if ( ! ( args = this->FilterArgs( self, args, kwds ) ) )
      return 0;

// translate the arguments
   if ( ! this->SetMethodArgs( args, user ) ) {
      Py_DECREF( args );
      return 0;
   }

   TClass* klass = (TClass*)this->GetClass().Id();

// perform the call, 0 makes the other side allocate the memory
   Long_t address = (Long_t)this->Execute( 0, release_gil );

// CLING WORKAROUND --
// Due to #100389, the constructor may not be a valid method. This happens if
// there is no (default) contructor, so just return a right-size bit of memory.
   if ( ! address ) {
      PyErr_Clear();
      // CLING WORKAROUND --
      // Theoretically, #100389 is fixed, but there are subtle ways that this
      // still fails. For whatever reason (TBD) simply calling the method again
      // has a high success rate
      address = (Long_t)this->Execute( 0, release_gil );
      // -- END (NESTED) CLING WORKAROUND
      if ( ! address ) {
         PyErr_Clear();
         address = (Long_t)klass->New();
      }
   }
// -- END CLING WORKAROUND

// done with filtered args
   Py_DECREF( args );

// return object if successful, lament if not
   if ( address != 0 ) {
      Py_INCREF( self );

   // note: constructors are no longer set to take ownership by default; instead that is
   // decided by the method proxy (which carries a creator flag) upon return
      self->Set( (void*)address );

   // allow lookup upon destruction on the ROOT/CINT side for TObjects
      TObject* object = (TObject*) klass->DynamicCast( TObject::Class(), (void*)address );
      if ( object )
         TMemoryRegulator::RegisterObject( self, object );

   // done with self
      Py_DECREF( self );

      Py_INCREF( Py_None );
      return Py_None;                        // by definition
   }

   if ( ! PyErr_Occurred() )   // should be set, otherwise write a generic error msg
      PyErr_SetString( PyExc_TypeError, const_cast< char* >(
         ( std::string( klass->GetName() ) + " constructor failed" ).c_str() ) );

// do not throw an exception, '0' might trigger the overload handler to choose a
// different constructor, which if all fails will throw an exception
   return 0;
}
