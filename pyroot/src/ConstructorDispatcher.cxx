// @(#)root/pyroot:$Name:  $:$Id: ConstructorDispatcher.cxx,v 1.5 2004/11/05 09:05:45 brun Exp $
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

// Standard
#include <string>


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
   if ( ! setMethodArgs( aTuple, 1 ) )
      return 0;                              // important: 0, not PyNone

   TClass* cls = getClass();

// perform the call (TODO: fails for loaded macro's, and New() is insufficient)
   long address = 0;
   if ( ! execute( cls, address ) || ! address ) {
   // we're probably dealing with an interpreted class
      if ( PyTuple_GET_SIZE( aTuple ) == 1 )
         address = (long) cls->New();        // attempt default ctor

   // else fail ...

   // CAUTION: creating an interpreted class doesn't work if it has STL type data
   // members that are initialized or otherwise touched in the ctor!
   }

// return object if successful, lament if not
   if ( address != 0 ) {
      PyObject* self = PyTuple_GetItem( aTuple, 0 );
      Py_INCREF( self );

   // note "true" (== default) for ROOT object deletion from the python side
      PyObject* cobj = PyCObject_FromVoidPtr(
         new ObjectHolder( (void*)address, cls ), &destroyObjectHolder );
      PyObject_SetAttr( self, Utility::theObjectString_, cobj );

   // allow lookup upon destruction on the ROOT/CINT side for TObjects
      TObject* tobj = (TObject*) cls->DynamicCast( TObject::Class(), (void*)address );
      if ( tobj )
         MemoryRegulator::RegisterObject( self, tobj );

   // done with this object
      Py_DECREF( cobj );

   // done with self
      Py_DECREF( self );

      Py_INCREF( Py_None );
      return Py_None;                        // by definition
   }

   PyErr_SetString( PyExc_TypeError, const_cast< char* >(
      ( std::string( cls->GetName() ) + " constructor failed" ).c_str() ) );

// do not throw an exception, '0' might trigger the overload handler to choose a
// different constructor, which if all fails will throw an exception
   return 0;
}
