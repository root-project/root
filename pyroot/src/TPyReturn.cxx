// @(#)root/pyroot:$Name:  $:$Id: TPyReturn.cxx,v 1.4 2005/03/30 05:16:19 brun Exp $
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
// (statically typed). Upon cast to another value, either implicitly (builtin
// types) or explicitly (pointers to ROOT objects), the TPyReturn object goes
// out of existence. For this reason, it can not be copied and it should not
// be held by reference.


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
   if ( this != & other ) {
      Py_INCREF( other.fPyObject );
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
   const char* s = PyString_AsString( fPyObject );

   if ( PyErr_Occurred() ) {
      PyErr_Print();
      return "";                   // returning 0 may be better?
   }

   return s;
}

//____________________________________________________________________________
TPyReturn::operator Char_t() const
{
   std::string s = operator const char*();
   if ( s.size() )
      return s[0];

   return '\0';
}

//____________________________________________________________________________
TPyReturn::operator Long_t() const
{
   Long_t l = PyLong_AsLong( fPyObject );

   if ( PyErr_Occurred() )
      PyErr_Print();

   return l;
}

//____________________________________________________________________________
TPyReturn::operator ULong_t() const
{
   ULong_t ul = PyLong_AsUnsignedLong( fPyObject );

   if ( PyErr_Occurred() )
      PyErr_Print();

   return ul;
}


//____________________________________________________________________________
TPyReturn::operator Double_t() const
{
   Double_t d = PyFloat_AsDouble( fPyObject );

   if ( PyErr_Occurred() )
      PyErr_Print();

   return d;
}

//____________________________________________________________________________
TPyReturn::operator void*() const
{
   if ( fPyObject == Py_None )
      return 0;

   Py_INCREF( fPyObject );
   if ( PyROOT::ObjectProxy_Check( fPyObject ) )
      return ((PyROOT::ObjectProxy*)fPyObject)->GetObject();
   else 
      return fPyObject;
}
