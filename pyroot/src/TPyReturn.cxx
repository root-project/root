// Author: Wim Lavrijsen, May 2004

// Bindings
#include "PyROOT.h"
#include "ObjectHolder.h"
#include "Utility.h"
#include "TPyReturn.h"

// ROOT
#include "TClass.h"
#include "TObject.h"
#include "TInterpreter.h"

// Standard
#include <stdexcept>


//- private helpers ----------------------------------------------------------
void TPyReturn::autoDestruct() const {
   if ( gInterpreter != 0 )
      gInterpreter->DeleteGlobal( (void*) this );
   delete this;
}


//- constructors/destructor --------------------------------------------------
TPyReturn::TPyReturn() : m_class( 0 ) {
   Py_INCREF( Py_None );
   m_object = Py_None;
}

TPyReturn::TPyReturn( PyObject* obj, TClass* cls ) :
   m_object( obj ), m_class( cls ) {}

TPyReturn::TPyReturn( const TPyReturn& s ) : TObject( s ) {
   throw std::runtime_error( "TPyReturn objects may not be copied!" );
}

TPyReturn& TPyReturn::operator=( const TPyReturn& ) {
   throw std::runtime_error( "TPyReturn objects may not be assigned to!" );
   return *this;
}

TPyReturn::~TPyReturn() {
   Py_XDECREF( m_object );
}


//- public members -----------------------------------------------------------
TClass* TPyReturn::IsA() const {
   return m_class;
}


TPyReturn::operator const char*() const {
   const char* s = PyString_AsString( m_object );
   autoDestruct();

   if ( PyErr_Occurred() ) {
      PyErr_Print();
      return "";
   }

   return s;
}

TPyReturn::operator long() const {
   long l = PyLong_AsLong( m_object );
   autoDestruct();

   if ( PyErr_Occurred() )
      PyErr_Print();

   return l;
}

TPyReturn::operator int() const {
   return (int) operator long();
}

TPyReturn::operator double() const {
   double d = PyFloat_AsDouble( m_object );
   autoDestruct();

   if ( PyErr_Occurred() )
      PyErr_Print();

   return d;
}

TPyReturn::operator float() const {
   return (float) operator double();
}

TPyReturn::operator TObject*() const {
   return (TObject*) this;
}
