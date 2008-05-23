// Author: Wim Lavrijsen, Aug 2007

// Bindings
#include "PyROOT.h"
#include "TPyDispatcher.h"

// ROOT
#include "TObject.h"

// Standard
#include <stdarg.h>


//______________________________________________________________________________
//                         Python callback dispatcher
//                         ==========================
//
// The TPyDispatcher class acts as a functor that can be used for TFn's and GUIs
// to install callbacks from CINT.


//- data ---------------------------------------------------------------------
ClassImp(TPyDispatcher)


//- constructors/destructor --------------------------------------------------
TPyDispatcher::TPyDispatcher( PyObject* callable ) : fCallable( 0 )
{
// Construct a TPyDispatcher from a callable python object. Applies python
// object reference counting.
   Py_XINCREF( callable );
   fCallable = callable;
}

//____________________________________________________________________________
TPyDispatcher::TPyDispatcher( const TPyDispatcher& other ) : TObject ( other )
{
// Copy constructor. Applies python object reference counting.
   Py_XINCREF( other.fCallable );
   fCallable = other.fCallable;
}

//____________________________________________________________________________
TPyDispatcher& TPyDispatcher::operator=( const TPyDispatcher& other )
{
// Assignment operator. Applies python object reference counting.
   if ( this != &other ) {
      this->TObject::operator=( other );

      Py_XDECREF( fCallable );
      Py_XINCREF( other.fCallable );
      fCallable = other.fCallable;
   }

   return *this;
}

//____________________________________________________________________________
TPyDispatcher::~TPyDispatcher() {
// Destructor. Reference counting for the held python object is in effect.
   Py_XDECREF( fCallable );
}


//- public members -----------------------------------------------------------
PyObject* TPyDispatcher::DispatchVA( const char* format, ... )
{
// Dispatch the arguments to the held callable python object, using format to
// interpret the types of the arguments. Note that format is in python style,
// not in C printf style. See: http://docs.python.org/api/arg-parsing.html .
   PyObject* args = 0;

   if ( format ) {
      va_list va;
      va_start( va, format );

      args = Py_VaBuildValue( (char*)format, va );

      va_end( va );

      if ( ! args ) {
         PyErr_Print();
         return 0;
      }

      if ( ! PyTuple_Check( args ) ) {    // if only one arg ...
         PyObject* t = PyTuple_New( 1 );
         PyTuple_SET_ITEM( t, 0, args );
         args = t;
      }

   }

   PyObject* result = PyObject_CallObject( fCallable, args );
   Py_XDECREF( args );

   if ( ! result ) {
      PyErr_Print();
      return 0;
   }

   return result;
}
