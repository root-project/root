// @(#)root/pyroot:$Name:  $:$Id: Utility.cxx,v 1.29 2006/05/28 19:05:24 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "Utility.h"
#include "ObjectProxy.h"
#include "MethodProxy.h"
#include "RootWrapper.h"
#include "PyCallable.h"

// ROOT
#include "TClassEdit.h"

// CINT
#include "Api.h"

// Standard
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <list>


//- data _____________________________________________________________________
PyROOT::DictLookup_t PyROOT::gDictLookupOrg = 0;
Bool_t PyROOT::gDictLookupActive = kFALSE;

PyROOT::Utility::EMemoryPolicy PyROOT::Utility::gMemoryPolicy = PyROOT::Utility::kHeuristics;

// this is just a data holder for linking; actual value is set in RootModule.cxx
PyROOT::Utility::ESignalPolicy PyROOT::Utility::gSignalPolicy = PyROOT::Utility::kSafe;

PyROOT::Utility::TC2POperatorMapping_t PyROOT::Utility::gC2POperatorMapping;

namespace {

   using namespace PyROOT::Utility;

   struct InitOperatorMapping_t {
   public:
      InitOperatorMapping_t() {
         // gC2POperatorMapping[ "[]" ]  = "__getitem__";   // depends on return type
         // gC2POperatorMapping[ "[]" ]  = "__setitem__";   // id.
         // gC2POperatorMapping[ "()" ]  = "__call__";      // depends on return type
         gC2POperatorMapping[ "+" ]   = "__add__";
         gC2POperatorMapping[ "-" ]   = "__sub__";
         // gC2POperatorMapping[ "*" ]   = "__mul__";       // double meaning in C++
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
Bool_t PyROOT::Utility::SetMemoryPolicy( EMemoryPolicy e )
{
   if ( kHeuristics <= e && e <= kStrict ) {
      gMemoryPolicy = e;
      return kTRUE;
   }
   return kFALSE;
}

//____________________________________________________________________________
Bool_t PyROOT::Utility::SetSignalPolicy( ESignalPolicy e )
{
   if ( kFast <= e && e <= kSafe ) {
      gSignalPolicy = e;
      return kTRUE;
   }
   return kFALSE;
}

//____________________________________________________________________________
Bool_t PyROOT::Utility::AddToClass(
      PyObject* pyclass, const char* label, PyCFunction cfunc, int flags )
{
// use list for clean-up (.so's are unloaded only at interpreter shutdown)
   static std::list< PyMethodDef > s_pymeths;

   s_pymeths.push_back( PyMethodDef() );
   PyMethodDef* pdef = &s_pymeths.back();
   pdef->ml_name  = const_cast< char* >( label );
   pdef->ml_meth  = cfunc;
   pdef->ml_flags = flags;
   pdef->ml_doc   = NULL;

   PyObject* func = PyCFunction_New( pdef, NULL );
   PyObject* method = PyMethod_New( func, NULL, pyclass );
   PyObject_SetAttrString( pyclass, pdef->ml_name, method );
   Py_DECREF( method );
   Py_DECREF( func );

   if ( PyErr_Occurred() )
      return kFALSE;

   return kTRUE;
}

//____________________________________________________________________________
Bool_t PyROOT::Utility::AddToClass( PyObject* pyclass, const char* label, const char* func )
{
   PyObject* pyfunc = PyObject_GetAttrString( pyclass, const_cast< char* >( func ) );
   if ( ! pyfunc )
      return kFALSE;

   return PyObject_SetAttrString( pyclass, const_cast< char* >( label ), pyfunc ) == 0;
}

//____________________________________________________________________________
Bool_t PyROOT::Utility::AddToClass( PyObject* pyclass, const char* label, PyCallable* pfunc )
{
   MethodProxy* method =
      (MethodProxy*)PyObject_GetAttrString( pyclass, const_cast< char* >( label ) );

   if ( ! method )
      return kFALSE;

   method->AddMethod( pfunc );

   Py_DECREF( method );
   return kTRUE;
}

//____________________________________________________________________________
Bool_t PyROOT::Utility::InitProxy( PyObject* module, PyTypeObject* pytype, const char* name )
{
// finalize proxy type
   if ( PyType_Ready( pytype ) < 0 )
      return kFALSE;

// add proxy type to the given (ROOT) module
   Py_INCREF( pytype );         // PyModule_AddObject steals reference
   if ( PyModule_AddObject( module, (char*)name, (PyObject*)pytype ) < 0 ) {
      Py_DECREF( pytype );
      return kFALSE;
   }

// declare success
   return kTRUE;
}

//____________________________________________________________________________
int PyROOT::Utility::GetBuffer( PyObject* pyobject, char tc, int size, void*& buf, Bool_t check )
{
// special case: don't handle strings here (yes, they're buffers, but not quite)
   if ( PyString_Check( pyobject ) )
      return 0;

// attempt to retrieve pointer to buffer interface
   PyBufferProcs* bufprocs = pyobject->ob_type->tp_as_buffer;
   PySequenceMethods* seqmeths = pyobject->ob_type->tp_as_sequence;
   if ( seqmeths != 0 && bufprocs != 0 && bufprocs->bf_getwritebuffer != 0 &&
        (*(bufprocs->bf_getsegcount))( pyobject, 0 ) == 1 ) {

   // get the buffer
      int buflen = (*(bufprocs->bf_getwritebuffer))( pyobject, 0, &buf );

      if ( check == kTRUE ) {
      // determine buffer compatibility (use "buf" as a status flag)
         PyObject* pytc = PyObject_GetAttrString( pyobject, const_cast< char* >( "typecode" ) );
         if ( pytc != 0 ) {     // for array objects
            if ( PyString_AS_STRING( pytc )[0] != tc )
               buf = 0;         // no match
            Py_DECREF( pytc );
         } else if ( buflen / (*(seqmeths->sq_length))( pyobject ) == size ) {
         // this is a gamble ... may or may not be ok, but that's for the user
            PyErr_Clear();
         } else
            buf = 0;                      // not compatible
      }

      return buflen;
   }

   return 0;
}

//____________________________________________________________________________
PyROOT::Utility::EDataType PyROOT::Utility::EffectiveType( const std::string& name )
{
   EDataType effType = kOther;

   G__TypeInfo ti( name.c_str() );
   if ( ti.Property() & G__BIT_ISENUM )
      return EDataType( (int) kEnum );

   std::string shortName = TClassEdit::ShortType( ti.TrueName(), 1 );

   const std::string& cpd = Compound( name );
   const int mask = cpd == "*" ? kPtrMask : 0;

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
   else if ( shortName == "string" && cpd == "" )
      effType = kSTLString;
   else if ( name == "#define" ) {
      effType = kMacro;
   }
   else
      effType = kOther;

   return effType;
}

//____________________________________________________________________________
const std::string PyROOT::Utility::Compound( const std::string& name )
{
   std::string compound = "";
   for ( int pos = (int)name.size()-1; 0 <= pos; --pos ) {
      if ( isspace( name[pos] ) ) continue;
      if ( isalnum( name[pos] ) || name[pos] == '>' ) break;

      compound = name[pos] + compound;
   }

   return compound;
}

//____________________________________________________________________________
void PyROOT::Utility::ErrMsgCallback( char* msg ) {
// Translate CINT error/warning into python equivalent

// ignore the "*** Interpreter error recovered ***" message
   if ( strstr( msg, "error recovered" ) )
      return;

// ignore CINT-style FILE/LINE messages
   if ( strstr( msg, "FILE:" ) )
      return;

// get file name and line number
   char* errFile = G__stripfilename( G__get_ifile()->name );
   int errLine = G__get_ifile()->line_number;

// ignore ROOT-style FILE/LINE messages
   char buf[256];
   snprintf( buf, 256, "%s:%d:", errFile, errLine );
   if ( strstr( msg, buf ) )
      return;

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
      PyErr_Format( PyExc_IndexError, format, errFile, errLine, p+12 );
   else if ( ( p = strstr( msg, "Error:" ) ) )
      PyErr_Format( PyExc_RuntimeError, format, errFile, errLine, p+7 );
   else if ( ( p = strstr( msg, "Exception:" ) ) )
      PyErr_Format( PyExc_RuntimeError, format, errFile, errLine, p+11 );
   else if ( ( p = strstr( msg, "Limitation:" ) ) )
      PyErr_Format( PyExc_NotImplementedError, format, errFile, errLine, p+12 );
   else if ( ( p = strstr( msg, "Internal Error: malloc" ) ) )
      PyErr_Format( PyExc_MemoryError, format, errFile, errLine, p+23 );
   else if ( ( p = strstr( msg, "Internal Error:" ) ) )
      PyErr_Format( PyExc_SystemError, format, errFile, errLine, p+16 );
   else if ( ( p = strstr( msg, "Warning:" ) ) )
// either printout or raise exception, depending on user settings
      PyErr_WarnExplicit( NULL, p+9, errFile, errLine, (char*)"CINT", NULL );
   else if ( ( p = strstr( msg, "Note:" ) ) )
      fprintf( stdout, "Note: (file \"%s\", line %d) %s\n", errFile, errLine, p+6 );
   else   // unknown: printing it to screen is the safest action
      fprintf( stdout, "Message: (file \"%s\", line %d) %s\n", errFile, errLine, msg );
}
