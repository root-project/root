// @(#)root/pyroot:$Name:  $:$Id: TPyReturn.cxx,v 1.68 2005/01/28 05:45:41 brun Exp $
// Author: Wim Lavrijsen, May 2004

// Bindings
#include "PyROOT.h"
#include "TPyReturn.h"

// ROOT
#include "TClass.h"
#include "TObject.h"
#include "TInterpreter.h"

// Standard
#include <stdexcept>


//______________________________________________________________________________
//                        Python expression eval result
//                        =============================
//
// Transport class for bringing objects from python (dynamically typed) to CINT
// (statically typed). Upon cast to another value, either implicitly (builtin
// types) or explicitly (pointers to ROOT objects), the TPyReturn object goes
// out of existence. For this reason, it can not be copied and it should not
// be held by reference.


//- data ---------------------------------------------------------------------
ClassImp(TPyReturn)


//- private helpers ----------------------------------------------------------
void TPyReturn::AutoDestruct_() const
{
// Private harakiri method.
   if ( gInterpreter != 0 )
      gInterpreter->DeleteGlobal( (void*) this );
   delete this;
}


//- constructors/destructor --------------------------------------------------
TPyReturn::TPyReturn() : fClass( 0 )
{
// Construct a TPyReturn object from Py_None.
   Py_INCREF( Py_None );
   fPyObject = Py_None;
}

//____________________________________________________________________________
TPyReturn::TPyReturn( PyObject* pyobject, TClass* klass ) :
   fPyObject( pyobject ), fClass( klass )
{
// Construct a TPyReturn from a python object. If the python object holds on to
// a ROOT object, the TClass should be given. Reference counting for the python
// object is in effect.
}

//____________________________________________________________________________
TPyReturn::TPyReturn( const TPyReturn& other ) : TObject( other )
{
// Private copy constructor; throws if called.
   throw std::runtime_error( "TPyReturn objects may not be copied!" );
}

//____________________________________________________________________________
TPyReturn& TPyReturn::operator=( const TPyReturn& )
{
// Private assignment operator; throws if called.
   throw std::runtime_error( "TPyReturn objects may not be assigned to!" );
   return *this;
}

//____________________________________________________________________________
TPyReturn::~TPyReturn()
{
// Destructor. Reference counting for the held python object is in effect.
   Py_XDECREF( fPyObject );
}


//- public members -----------------------------------------------------------
TClass* TPyReturn::IsA() const
{
// Return the held object TClass (not the TPyReturn TClass).
   return fClass;
}

//____________________________________________________________________________
TPyReturn::operator const char*() const
{
   const char* s = PyString_AsString( fPyObject );
   AutoDestruct_();

   if ( PyErr_Occurred() ) {
      PyErr_Print();
      return "";
   }

   return s;
}

//____________________________________________________________________________
TPyReturn::operator long() const
{
   long l = PyLong_AsLong( fPyObject );
   AutoDestruct_();

   if ( PyErr_Occurred() )
      PyErr_Print();

   return l;
}

//____________________________________________________________________________
TPyReturn::operator int() const
{
   return (int) operator long();
}

//____________________________________________________________________________
TPyReturn::operator double() const
{
   double d = PyFloat_AsDouble( fPyObject );
   AutoDestruct_();

   if ( PyErr_Occurred() )
      PyErr_Print();

   return d;
}

//____________________________________________________________________________
TPyReturn::operator float() const
{
   return (float) operator double();
}

//____________________________________________________________________________
TPyReturn::operator TObject*() const
{
   return (TObject*) this;
}
