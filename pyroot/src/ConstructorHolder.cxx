// @(#)root/pyroot:$Name:  $:$Id: ConstructorHolder.cxx,v 1.68 2005/01/28 05:45:41 brun Exp $
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
#include <assert.h>
#include <string>


//- protected members --------------------------------------------------------
bool PyROOT::ConstructorHolder::InitExecutor_( Executor*& executor )
{
   executor = (gExecFactories[ "__init__" ])();
   return true;
}


//- constructor --------------------------------------------------------------
PyROOT::ConstructorHolder::ConstructorHolder( TClass* klass, TMethod* method ) :
      MethodHolder( klass, method )
{
}


//- public members -----------------------------------------------------------
PyObject* PyROOT::ConstructorHolder::GetDocString()
{
   return PyString_FromFormat(
      "%s::__init__%s", GetClass()->GetName(), GetMethod()->GetSignature() );
}

//____________________________________________________________________________
PyObject* PyROOT::ConstructorHolder::operator()( ObjectProxy* self, PyObject* args, PyObject* kwds )
{
// setup as necessary
   if ( ! Initialize() )
      return 0;                              // important: 0, not PyNone

// verify and put the arguments in usable order
   if ( ! FilterArgs( self, args, kwds ) )
      return 0;

// translate the arguments
   if ( ! SetMethodArgs( args ) )
      return 0;                              // important: 0, not PyNone

   TClass* klass = GetClass();

// perform the call (TODO: fails for loaded macro's, and New() is insufficient)
   long address = (long)Execute( klass );
   if ( ! address ) {
   // we're probably dealing with an interpreted class
      if ( PyTuple_GET_SIZE( args ) == 0 )
         address = (long) klass->New();      // attempt default ctor

   // else fail ...

   // CAUTION: creating an interpreted class doesn't work if it has STL type data
   // members that are initialized or otherwise touched in the ctor!
   }

// return object if successful, lament if not
   if ( address != 0 ) {
      Py_INCREF( self );

   // note "kIsOwner" for ROOT object deletion from the python side
      self->Set( (void*) address, klass, ObjectProxy::kIsOwner );

   // allow lookup upon destruction on the ROOT/CINT side for TObjects
      TObject* object = (TObject*) klass->DynamicCast( TObject::Class(), (void*)address );
      if ( object )
         MemoryRegulator::RegisterObject( self, object );

   // done with self
      Py_DECREF( self );

      Py_INCREF( Py_None );
      return Py_None;                        // by definition
   }

   PyErr_SetString( PyExc_TypeError, const_cast< char* >(
      ( std::string( klass->GetName() ) + " constructor failed" ).c_str() ) );

// do not throw an exception, '0' might trigger the overload handler to choose a
// different constructor, which if all fails will throw an exception
   return 0;
}
