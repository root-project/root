// @(#)root/pyroot:$Name:  $:$Id: ConstructorDispatcher.cxx,v 1.1 2004/04/27 06:28:48 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "ConstructorDispatcher.h"
#include "ObjectHolder.h"
#include "MemoryRegulator.h"
#include "Utility.h"

// ROOT
#include "TClass.h"
#include "TMethod.h"
#include "TMethodCall.h"


//- helpers ---------------------------------------------------------------------
namespace {

// FIXME: this is a dupe
   void destroyObjectHolder( void* oh ) {
      delete reinterpret_cast< PyROOT::ObjectHolder* >( oh );
   }

}


//- constructor -----------------------------------------------------------------
PyROOT::ConstructorDispatcher::ConstructorDispatcher( TClass* cls, TMethod* tm ) :
      MethodHolder( cls, tm ) {
}


//- public members --------------------------------------------------------------
PyObject* PyROOT::ConstructorDispatcher::operator()( PyObject* aTuple, PyObject* /* aDict */ ) {
// setup as necessary
   if ( ! initialize() )
      return 0;                              // important: 0, not PyNone

// translate the arguments
   if ( ! setMethodArgs( aTuple ) )
      return 0;                              // important: 0, not PyNone

   TClass* cls = getClass();

// perform the call, return object if successful
   long address = 0;
   if ( execute( cls, address ) && address != 0 ) {
      PyObject* self = PyTuple_GetItem( aTuple, 0 );
      Py_INCREF( self );

   // Note "false" for ROOT object deletion from the python side!
   // The reason for this is that ROOT does not share in python's reference counting
   // and scopes between C++ and python are different. Thus, ROOT could hold on to
   // objects internally, even though objects have phased out on the python side.
      PyObject* cobj = PyCObject_FromVoidPtr(
         new ObjectHolder( (void*)address, cls, false ), NULL ); //destroyObjectHolder );
      PyObject_SetAttr( self, Utility::theObjectString_, cobj );

   // allow lookup upon destruction on the ROOT/CINT side
      MemoryRegulator::RegisterObject( self, (void*)address );

   // done with this object
      Py_DECREF( cobj );
      Py_DECREF( self );

      Py_INCREF( Py_None );
      return Py_None;                        // by definition
   }

// do not throw an exception, '0' might trigger the overload handler to choose a
// different constructor, which if all fails will throw an exception
   return 0;
}
