// @(#)root/pyroot:$Name:  $:$Id: Utility.cxx,v 1.16 2005/05/06 15:02:52 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "Utility.h"
#include "ObjectProxy.h"
#include "RootWrapper.h"

// ROOT
#include "TClassEdit.h"

// CINT
#include "Api.h"

// Standard
#include <stdlib.h>
#include <string.h>


//- data _____________________________________________________________________
PyROOT::dictlookup PyROOT::gDictLookupOrg = 0;
bool PyROOT::gDictLookupActive = false;


PyROOT::Utility::TC2POperatorMapping_t PyROOT::Utility::gC2POperatorMapping;

namespace {

   using namespace PyROOT::Utility;

   class InitOperatorMapping_ {
   public:
      InitOperatorMapping_() {
         gC2POperatorMapping[ "[]" ]  = "__getitem__";
         gC2POperatorMapping[ "()" ]  = "__call__";
         gC2POperatorMapping[ "+" ]   = "__add__";
         gC2POperatorMapping[ "-" ]   = "__sub__";
         gC2POperatorMapping[ "*" ]   = "__mul__";
         gC2POperatorMapping[ "/" ]   = "__div__";
         gC2POperatorMapping[ "%" ]   = "__mod__";
         gC2POperatorMapping[ "**" ]  = "__pow__";
         gC2POperatorMapping[ "<<" ]  = "__lshift__";
         gC2POperatorMapping[ ">>" ]  = "__rshift__";
         gC2POperatorMapping[ "&" ]   = "__and__";
         gC2POperatorMapping[ "|" ]   = "__or__";
         gC2POperatorMapping[ "^" ]   = "__xor__";
         gC2POperatorMapping[ "+=" ]  = "__iadd__";
         gC2POperatorMapping[ "-=" ]  = "__isub__";
         gC2POperatorMapping[ "*=" ]  = "__imul__";
         gC2POperatorMapping[ "/=" ]  = "__idiv__";
         gC2POperatorMapping[ "/=" ]  = "__imod__";
         gC2POperatorMapping[ "**=" ] = "__ipow__";
         gC2POperatorMapping[ "<<=" ] = "__ilshift__";
         gC2POperatorMapping[ ">>=" ] = "__irshift__";
         gC2POperatorMapping[ "&=" ]  = "__iand__";
         gC2POperatorMapping[ "|=" ]  = "__ior__";
         gC2POperatorMapping[ "^=" ]  = "__ixor__";
         gC2POperatorMapping[ "==" ]  = "__eq__";
         gC2POperatorMapping[ "!=" ]  = "__ne__";
         gC2POperatorMapping[ ">" ]   = "__gt__";
         gC2POperatorMapping[ "<" ]   = "__lt__";
         gC2POperatorMapping[ ">=" ]  = "__ge__";
         gC2POperatorMapping[ "<=" ]  = "__le__";
      }
   } initOperatorMapping_;

} // unnamed namespace


//- public functions ---------------------------------------------------------
bool PyROOT::Utility::AddToClass(
      PyObject* pyclass, const char* label, PyCFunction cfunc, int flags )
{
   PyMethodDef* pdef = new PyMethodDef;
   pdef->ml_name  = const_cast< char* >( label );
   pdef->ml_meth  = cfunc;
   pdef->ml_flags = flags;
   pdef->ml_doc   = NULL;

   PyObject* func = PyCFunction_New( pdef, NULL );
   PyObject* method = PyMethod_New( func, NULL, pyclass );
   PyObject_SetAttrString( pyclass, pdef->ml_name, method );
   Py_DECREF( func );
   Py_DECREF( method );

   if ( PyErr_Occurred() )
      return false;

   return true;
}

//____________________________________________________________________________
bool PyROOT::Utility::AddToClass( PyObject* pyclass, const char* label, const char* func )
{
   PyObject* pyfunc = PyObject_GetAttrString( pyclass, const_cast< char* >( func ) );
   if ( ! pyfunc )
      return false;

   return PyObject_SetAttrString( pyclass, const_cast< char* >( label ), pyfunc ) == 0;
}


//____________________________________________________________________________
bool PyROOT::Utility::InitProxy( PyObject* module, PyTypeObject* pytype, const char* name )
{
// finalize proxy type
   if ( PyType_Ready( pytype ) < 0 )
      return false;

   if ( PyErr_Occurred() )
      PyErr_Print();

// add proxy type to the given (ROOT) module
   Py_INCREF( pytype );         // PyModule_AddObject steals reference
   if ( PyModule_AddObject( module, (char*)name, (PyObject*)pytype ) < 0 ) {
      Py_DECREF( pytype );
      return false;
   }

// declare success
   return true;
}

//____________________________________________________________________________
PyROOT::Utility::EDataType PyROOT::Utility::effectiveType( const std::string& name )
{
   EDataType effType = kOther;

   G__TypeInfo ti( name.c_str() );
   if ( ti.Property() & G__BIT_ISENUM )
      return EDataType( (int) kEnum );

   std::string shortName = TClassEdit::ShortType( ti.TrueName(), 1 );

   const int isp = isPointer( name );
   const int mask = isp == 1 ? kPtrMask : 0;

   if ( shortName == "bool" )
      effType = EDataType( (int) kBool | mask );
   else if ( shortName == "char" )
      effType = EDataType( (int) kChar | mask );
   else if ( shortName == "short" )
      effType = EDataType( (int) kShort | mask );
   else if ( shortName == "int" )
      effType = EDataType( (int) kInt | mask );
   else if ( shortName == "unsigned int" )
      effType = EDataType( (int) kUInt | mask );
   else if ( shortName == "long" )
      effType = EDataType( (int) kLong | mask );
   else if ( shortName == "unsigned long" )
      effType = EDataType( (int) kULong | mask );
   else if ( shortName == "long long" )
      effType = EDataType( (int) kLongLong | mask );
   else if ( shortName == "float" )
      effType = EDataType( (int) kFloat | mask );
   else if ( shortName == "double" )
      effType = EDataType( (int) kDouble | mask );
   else if ( shortName == "void" )
      effType = EDataType( (int) kVoid | mask );
   else if ( shortName == "string" && isp == 0 )
      effType = kSTLString;
   else if ( name == "#define" ) {
      effType = kMacro;
   }
   else
      effType = kOther;

   return effType;
}

//____________________________________________________________________________
int PyROOT::Utility::isPointer( const std::string& name )
{
   int isp = 0;
   for ( std::string::const_reverse_iterator it = name.rbegin(); it != name.rend(); ++it ) {
      if ( *it == '*' ) {
         isp = 1;
         break;
      } else if ( *it == '&' ) {
         isp = 2;
         break;
      } else if ( isalnum( *it ) )
         break;
   }

   return isp;
}

//____________________________________________________________________________
void PyROOT::Utility::ErrMsgCallback( char* msg ) {
// Translate CINT error/warning into python equivalent

// ignore the "*** Interpreter error recovered ***" message
   if ( strstr( msg, "error recovered" ) )
      return;

// ignore FILE/LINE messages (picked up directly when needed)
   if ( strstr( msg, "FILE:" ) )
      return;

// get file name and line number
   char* errFile = strstr( G__get_ifile()->name, "./" );
   if ( ! errFile )
      errFile = G__get_ifile()->name;
   int errLine = G__get_ifile()->line_number;

// strip newline, if any
   int len = strlen( msg );
   if ( msg[ len-1 ] == '\n' )
      msg[ len-1 ] = '\0';

// concatenate message if already in error processing mode (e.g. if multiple CINT errors)
   if ( PyErr_Occurred() ) {
      PyObject *etype, *value, *trace;
      PyErr_Fetch( &etype, &value, &trace );           // clears current exception

   // need to be sure that error can be added; otherwise leave earlier error in place
      if ( PyString_Check( value ) ) {
         if ( ! PyErr_GivenExceptionMatches( etype, PyExc_IndexError ) )
            PyString_ConcatAndDel( &value, PyString_FromString( (char*)"\n  " ) );
         PyString_ConcatAndDel( &value, PyString_FromString( msg ) );
      }

      PyErr_Restore( etype, value, trace );
      return;
   }

// else, tranlate known errors and warnings, or simply accept the default
   char* format = (char*)"(file \"%s\", line %d) %s";
   char* p = 0;
   if ( ( p = strstr( msg, "Syntax Error:" ) ) )
      PyErr_Format( PyExc_SyntaxError, format, errFile, errLine, p+14 );
   else if ( ( p = strstr( msg, "Error: Array" ) ) )
      PyErr_Format( PyExc_IndexError, format, errFile, errLine, p+7 );
   else if ( ( p = strstr( msg, "Error:" ) ) )
      PyErr_Format( PyExc_RuntimeError, format, errFile, errLine, p+7 );
   else if ( ( p = strstr( msg, "Limitation:" ) ) )
      PyErr_Format( PyExc_NotImplementedError, format, errFile, errLine, p+12 );
   else if ( ( p = strstr( msg, "Internal Error: malloc" ) ) )
      PyErr_Format( PyExc_MemoryError, format, errFile, errLine, p+16 );
   else if ( ( p = strstr( msg, "Internal Error:" ) ) )
      PyErr_Format( PyExc_SystemError, format, errFile, errLine, p+16 );
   else if ( ( p = strstr( msg, "Warning:" ) ) ) {
   // either printout or raise exception, depending on user settings
      PyErr_WarnExplicit( NULL, p+9, errFile, errLine, (char*)"CINT", NULL );
      return;                                // NOTE: return after warning is set
   }
   else
      PyErr_Format( PyExc_RuntimeError, format, errFile, errLine, msg );
}
