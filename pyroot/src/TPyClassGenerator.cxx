// Author: Wim Lavrijsen, May 2004

// CINT
#include "Api.h"

// Bindings
#include "PyROOT.h"
#include "TPyClassGenerator.h"

// ROOT
#include "TClass.h"
#include "TObject.h"

// Standard
#include <iostream>
#include <string>


//- public members -----------------------------------------------------------
TClass* TPyClassGenerator::GetClass( const char* classname, Bool_t load ) {
   if ( ! load || ! classname )
      return 0;

   std::string fullName = classname;

// determine module and class name part
   std::string::size_type pos = fullName.rfind( '.' );

   if ( pos == std::string::npos )
      return 0;                              // this isn't a python style class

   std::string className = fullName.substr( pos+1, std::string::npos );
   std::string moduleName = fullName.substr( 0, pos );

// locate and get class
   PyObject* mod = PyImport_AddModule( const_cast< char* >( moduleName.c_str() ) );
   if ( ! mod ) {
      PyErr_Clear();
      return 0;                              // the module is no longer available?!
   }

   PyObject* pyclass =
      PyDict_GetItemString( PyModule_GetDict( mod ), const_cast< char* >( className.c_str() ) );
   Py_XINCREF( pyclass );
   Py_DECREF( mod );

   if ( ! pyclass ) {
      PyErr_Clear();                         // the class is no longer available?!
      return 0;
   }

// build CINT class placeholder (yes, this is down-right silly :) )
   G__exec_text( ( "class " + className + " {};" ).c_str() );

// TODO: construct ROOT class
   Py_DECREF( pyclass );

   TClass* cls = new TClass( className.c_str() );
   gROOT->AddClass( cls );

   return cls;
}


TClass* TPyClassGenerator::GetClass( const type_info& typeinfo, Bool_t load ) {
   return GetClass( typeinfo.name(), load );
}
