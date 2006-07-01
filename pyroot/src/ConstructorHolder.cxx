// @(#)root/pyroot:$Name:  $:$Id: ConstructorHolder.cxx,v 1.8 2006/03/09 09:07:02 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "ConstructorHolder.h"
#include "Executors.h"
#include "ObjectProxy.h"
#include "MemoryRegulator.h"

// ROOT
#include "TClass.h"
#include "TMethod.h"

// Standard
#include <string>


//- protected members --------------------------------------------------------
Bool_t PyROOT::TConstructorHolder::InitExecutor_( TExecutor*& executor )
{
// pick up special case new object executor
   executor = (gExecFactories[ "__init__" ])();
   return kTRUE;
}


//- constructor --------------------------------------------------------------
PyROOT::TConstructorHolder::TConstructorHolder( TClass* klass, TMethod* method ) :
      TMethodHolder( klass, method )
{
}


//- public members -----------------------------------------------------------
PyObject* PyROOT::TConstructorHolder::GetDocString()
{
// GetMethod() may return zero if this is just a special case place holder
   const char* clName = GetClass()->GetName();
   return PyString_FromFormat( "%s::%s%s",
      clName, clName, GetMethod() ? GetMethod()->GetSignature() : "()" );
}

//____________________________________________________________________________
PyObject* PyROOT::TConstructorHolder::operator()( ObjectProxy* self, PyObject* args, PyObject* kwds )
{
// setup as necessary
   if ( ! Initialize() )
      return 0;                              // important: 0, not Py_None
   
// fetch self, verify, and put the arguments in usable order
   if ( ! ( args = FilterArgs( self, args, kwds ) ) )
      return 0;
      
// translate the arguments
   if ( ! SetMethodArgs( args ) ) {
      Py_DECREF( args );
      return 0;
   }
   
   TClass* klass = GetClass();

// perform the call (TODO: fails for loaded macro's, and New() is insufficient)
   Long_t address = (Long_t)Execute( klass );
   if ( ! address ) {
   // we're probably dealing with an interpreted class
      if ( PyTuple_GET_SIZE( args ) == 0 )
         address = (Long_t)klass->New();      // attempt default ctor

   // else fail ...

   // CAUTION: creating an interpreted class doesn't work if it has STL type data
   // members that are initialized or otherwise touched in the ctor!
   }

// done with filtered args
   Py_DECREF( args );

// return object if successful, lament if not
   if ( address != 0 ) {
      Py_INCREF( self );

   // note "kIsOwner" for ROOT object deletion from the python side
      self->Set( (void*) address, klass, ObjectProxy::kIsOwner );

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
