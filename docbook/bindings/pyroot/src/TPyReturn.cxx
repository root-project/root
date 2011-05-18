// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, May 2004

// Bindings
#include "PyROOT.h"
#include "TPyReturn.h"
#include "ObjectProxy.h"

// ROOT
#include "TObject.h"
#include "TInterpreter.h"

// Standard
#include <stdexcept>


//______________________________________________________________________________
//                        Python expression eval result
//                        =============================
//
// Transport class for bringing objects from python (dynamically typed) to CINT
// (statically typed). It is best to immediately cast a TPyReturn to the real
// type, either implicitly (for builtin types) or explicitly (through a void*
// cast for pointers to ROOT objects).
//
// Examples:
//
//  root [0] TBrowser* b = (void*)TPython::Eval( "ROOT.TBrowser()" );
//  root [1] int i = TPython::Eval( "1+1" );
//  root [2] i
//  (int)2
//  root [3] double d = TPython::Eval( "1+3.1415" );
//  root [4] d
//  (double)4.14150000000000063e+00


//- data ---------------------------------------------------------------------
ClassImp(TPyReturn)


//- constructors/destructor --------------------------------------------------
TPyReturn::TPyReturn()
{
// Construct a TPyReturn object from Py_None.
   Py_INCREF( Py_None );
   fPyObject = Py_None;
}

//____________________________________________________________________________
TPyReturn::TPyReturn( PyObject* pyobject )
{
// Construct a TPyReturn from a python object. The python object may represent
// a ROOT object. Steals reference to given python object.
   if ( ! pyobject ) {
      Py_INCREF( Py_None );
      fPyObject = Py_None;
   } else
      fPyObject = pyobject;
}

//____________________________________________________________________________
TPyReturn::TPyReturn( const TPyReturn& other )
{
// Copy constructor. Applies python object reference counting.
   Py_INCREF( other.fPyObject );
   fPyObject = other.fPyObject;
}

//____________________________________________________________________________
TPyReturn& TPyReturn::operator=( const TPyReturn& other )
{
// Assignment operator. Applies python object reference counting.
   if ( this != &other ) {
      Py_INCREF( other.fPyObject );
      Py_DECREF( fPyObject );
      fPyObject = other.fPyObject;
   }
   
   return *this;
}

//____________________________________________________________________________
TPyReturn::~TPyReturn()
{
// Destructor. Reference counting for the held python object is in effect.
   Py_DECREF( fPyObject );
}


//- public members -----------------------------------------------------------
TPyReturn::operator const char*() const
{
// Cast python return value to C-style string (may fail).
   const char* s = PyBytes_AsString( fPyObject );

   if ( PyErr_Occurred() ) {
      PyErr_Print();
      return "";                   // returning 0 may be better?
   }

   return s;
}

//____________________________________________________________________________
TPyReturn::operator Char_t() const
{
// Cast python return value to C++ char (may fail).
   std::string s = operator const char*();
   if ( s.size() )
      return s[0];

   return '\0';
}

//____________________________________________________________________________
TPyReturn::operator Long_t() const
{
// Cast python return value to C++ long (may fail).
   Long_t l = PyLong_AsLong( fPyObject );

   if ( PyErr_Occurred() )
      PyErr_Print();

   return l;
}

//____________________________________________________________________________
TPyReturn::operator ULong_t() const
{
// Cast python return value to C++ unsigned long (may fail).
   ULong_t ul = PyLong_AsUnsignedLong( fPyObject );

   if ( PyErr_Occurred() )
      PyErr_Print();

   return ul;
}


//____________________________________________________________________________
TPyReturn::operator Double_t() const
{
// Cast python return value to to C++ double (may fail).
   Double_t d = PyFloat_AsDouble( fPyObject );

   if ( PyErr_Occurred() )
      PyErr_Print();

   return d;
}

//____________________________________________________________________________
TPyReturn::operator void*() const
{
// Cast python return value to ROOT object with dictionary (may fail; note that
// you have to use the void* converter, as CINT will not call any other).
   if ( fPyObject == Py_None )
      return 0;

   Py_INCREF( fPyObject );
   if ( PyROOT::ObjectProxy_Check( fPyObject ) )
      return ((PyROOT::ObjectProxy*)fPyObject)->GetObject();
   else 
      return fPyObject;
}

//____________________________________________________________________________
TPyReturn::operator PyObject*() const
{
// Direct return of the held PyObject; note the new reference.
   if ( fPyObject == Py_None )
      return 0;

   Py_INCREF( fPyObject );
   return fPyObject;
}
