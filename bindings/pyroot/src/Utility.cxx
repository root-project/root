// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "PyStrings.h"
#include "Utility.h"
#include "ObjectProxy.h"
#include "MethodProxy.h"
#include "TCustomPyTypes.h"
#include "RootWrapper.h"
#include "PyCallable.h"

// ROOT
#include "TClassEdit.h"
#include "TError.h"

// CINT
#include "Api.h"

// Standard
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <list>
#include <utility>


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
         // gC2POperatorMapping[ "+" ]   = "__add__";       // depends on # of args (see __pos__)
         // gC2POperatorMapping[ "-" ]   = "__sub__";       // id. (eq. __neg__)
         // gC2POperatorMapping[ "*" ]   = "__mul__";       // double meaning in C++

         gC2POperatorMapping[ "/" ]   = "__div__";
         gC2POperatorMapping[ "%" ]   = "__mod__";
         gC2POperatorMapping[ "**" ]  = "__pow__";
         gC2POperatorMapping[ "<<" ]  = "__lshift__";
         gC2POperatorMapping[ ">>" ]  = "__rshift__";
         gC2POperatorMapping[ "&" ]   = "__and__";
         gC2POperatorMapping[ "|" ]   = "__or__";
         gC2POperatorMapping[ "^" ]   = "__xor__";
         gC2POperatorMapping[ "~" ]   = "__inv__";
         gC2POperatorMapping[ "+=" ]  = "__iadd__";
         gC2POperatorMapping[ "-=" ]  = "__isub__";
         gC2POperatorMapping[ "*=" ]  = "__imul__";
         gC2POperatorMapping[ "/=" ]  = "__idiv__";
         gC2POperatorMapping[ "%=" ]  = "__imod__";
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

         gC2POperatorMapping[ "int" ]    = "__int__";
         gC2POperatorMapping[ "long" ]   = "__long__";
         gC2POperatorMapping[ "float" ]  = "__float__";
         gC2POperatorMapping[ "double" ] = "__float__";     // python float is double

         gC2POperatorMapping[ "->" ]  = "__follow__";       // not an actual python operator
         gC2POperatorMapping[ "=" ]   = "__assign__";       // id.

         gC2POperatorMapping[ "bool" ] = "__nonzero__";
      }
   } initOperatorMapping_;

// for keeping track of callbacks for CINT-installed methods into python:
   typedef std::pair< PyObject*, Long_t > CallInfo_t;
   std::map< int, CallInfo_t > s_PyObjectCallbacks;

} // unnamed namespace


//- public functions ---------------------------------------------------------
ULong_t PyROOT::PyLongOrInt_AsULong( PyObject* pyobject )
{
// convert <pybject> to C++ unsigned long, with bounds checking, allow int -> ulong
   ULong_t ul = PyLong_AsUnsignedLong( pyobject );
   if ( PyErr_Occurred() && PyInt_Check( pyobject ) ) {
      PyErr_Clear();
      Long_t i = PyInt_AS_LONG( pyobject );
      if ( 0 <= i ) {
         ul = (ULong_t)i;
      } else {
         PyErr_SetString( PyExc_ValueError,
            "can\'t convert negative value to unsigned long" );
      }
   }

   return ul;
}

//____________________________________________________________________________
ULong64_t PyROOT::PyLongOrInt_AsULong64( PyObject* pyobject )
{
// convert <pyobject> to C++ unsigned long long, with bounds checking
   ULong64_t ull = PyLong_AsUnsignedLongLong( pyobject );
   if ( PyErr_Occurred() && PyInt_Check( pyobject ) ) {
      PyErr_Clear();
      Long_t i = PyInt_AS_LONG( pyobject );
      if ( 0 <= i ) {
         ull = (ULong64_t)i;
      } else {
         PyErr_SetString( PyExc_ValueError,
            "can\'t convert negative value to unsigned long long" );
      }
   }

   return ull;
}

//____________________________________________________________________________
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
   PyObject* method = TCustomInstanceMethod_New( func, NULL, pyclass );
   Bool_t isOk = PyObject_SetAttrString( pyclass, pdef->ml_name, method ) == 0;
   Py_DECREF( method );
   Py_DECREF( func );

   if ( PyErr_Occurred() )
      return kFALSE;

   if ( ! isOk ) {
      PyErr_Format( PyExc_TypeError, "could not add method %s", label );
      return kFALSE;
   }

   return kTRUE;
}

//____________________________________________________________________________
Bool_t PyROOT::Utility::AddToClass( PyObject* pyclass, const char* label, const char* func )
{
   PyObject* pyfunc = PyObject_GetAttrString( pyclass, const_cast< char* >( func ) );
   if ( ! pyfunc )
      return kFALSE;

   Bool_t isOk = PyObject_SetAttrString( pyclass, const_cast< char* >( label ), pyfunc ) == 0;

   Py_DECREF( pyfunc );
   return isOk;
}

//____________________________________________________________________________
Bool_t PyROOT::Utility::AddToClass( PyObject* pyclass, const char* label, PyCallable* pyfunc )
{
   MethodProxy* method =
      (MethodProxy*)PyObject_GetAttrString( pyclass, const_cast< char* >( label ) );

   if ( ! method || ! MethodProxy_Check( method ) ) {
   // not adding to existing MethodProxy; add callable directly to the class
      if ( PyErr_Occurred() )
         PyErr_Clear();
      return PyObject_SetAttrString( pyclass, const_cast< char* >( label ), (PyObject*)pyfunc ) == 0;
   }

   method->AddMethod( pyfunc );

   Py_DECREF( method );
   return kTRUE;
}

//____________________________________________________________________________
Bool_t PyROOT::Utility::BuildTemplateName( PyObject*& pyname, PyObject* args, int argoff )
{
// helper to construct the "< type, type, ... >" part of a templated name (either
// for a class as in MakeRootTemplateClass in RootModule.cxx) or for method lookup
// (as in TemplatedMemberHook, below)

   PyString_ConcatAndDel( &pyname, PyString_FromString( "<" ) );

   Py_ssize_t nArgs = PyTuple_GET_SIZE( args );
   for ( int i = argoff; i < nArgs; ++i ) {
   // add type as string to name
      PyObject* tn = PyTuple_GET_ITEM( args, i );
      if ( PyString_Check( tn ) )
         PyString_Concat( &pyname, tn );
      else if ( PyObject_HasAttr( tn, PyStrings::gName ) ) {
      // this works for type objects
         PyObject* tpName = PyObject_GetAttr( tn, PyStrings::gName );

      // special case for strings
         if ( strcmp( PyString_AS_STRING( tpName ), "str" ) == 0 ) {
            Py_DECREF( tpName );
            tpName = PyString_FromString( "std::string" );
         }

         PyString_ConcatAndDel( &pyname, tpName );
      } else {
      // last ditch attempt, works for things like int values
         PyObject* pystr = PyObject_Str( tn );
         if ( ! pystr ) {
            return kFALSE;
         }

         PyString_ConcatAndDel( &pyname, pystr );
      }

   // add a comma, as needed
      if ( i != nArgs - 1 )
         PyString_ConcatAndDel( &pyname, PyString_FromString( "," ) );
   }

// close template name; prevent '>>', which should be '> >'
   if ( PyString_AsString( pyname )[ PyString_Size( pyname ) - 1 ] == '>' )
      PyString_ConcatAndDel( &pyname, PyString_FromString( " >" ) );
   else
      PyString_ConcatAndDel( &pyname, PyString_FromString( ">" ) );

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
      Py_ssize_t buflen = (*(bufprocs->bf_getwritebuffer))( pyobject, 0, &buf );

      if ( check == kTRUE ) {
      // determine buffer compatibility (use "buf" as a status flag)
         PyObject* pytc = PyObject_GetAttr( pyobject, PyStrings::gTypeCode );
         if ( pytc != 0 ) {     // for array objects
            if ( PyString_AS_STRING( pytc )[0] != tc )
               buf = 0;         // no match
            Py_DECREF( pytc );
         } else if ( seqmeths->sq_length &&
                     (int)(buflen / (*(seqmeths->sq_length))( pyobject )) == size ) {
         // this is a gamble ... may or may not be ok, but that's for the user
            PyErr_Clear();
         } else if ( buflen == size ) {
         // also a gamble, but at least 1 item will fit into the buffer, so very likely ok ...
            PyErr_Clear();
         } else {
            buf = 0;                      // not compatible

         // clarify error message
            PyObject* pytype = 0, *pyvalue = 0, *pytrace = 0;
            PyErr_Fetch( &pytype, &pyvalue, &pytrace );
            PyObject* pyvalue2 = PyString_FromFormat(
               (char*)"%s and given element size (%ld) do not match needed (%d)",
               PyString_AS_STRING( pyvalue ),
               seqmeths->sq_length ? (long)(buflen / (*(seqmeths->sq_length))( pyobject )) : (long)buflen,
               size );
            Py_DECREF( pyvalue );
            PyErr_Restore( pytype, pyvalue2, pytrace );
         }
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
   std::string cleanName = name;
   std::string::size_type spos = std::string::npos;
   while ( ( spos = cleanName.find( "const" ) ) != std::string::npos ) {
      cleanName.swap( cleanName.erase( spos, 5 ) );
   }

   std::string compound = "";
   for ( int ipos = (int)cleanName.size()-1; 0 <= ipos; --ipos ) {
      if ( isspace( cleanName[ipos] ) ) continue;
      if ( isalnum( cleanName[ipos] ) || cleanName[ipos] == '>' ) break;

      compound = cleanName[ipos] + compound;
   }

   return compound;
}

//____________________________________________________________________________
void PyROOT::Utility::ErrMsgCallback( char* msg )
{
// Translate CINT error/warning into python equivalent

// ignore the "*** Interpreter error recovered ***" message
   if ( strstr( msg, "error recovered" ) )
      return;

// ignore CINT-style FILE/LINE messages
   if ( strstr( msg, "FILE:" ) )
      return;

// get file name and line number
   char* errFile = (char*)G__stripfilename( G__get_ifile()->name );
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

// else, translate known errors and warnings, or simply accept the default
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

//____________________________________________________________________________
void PyROOT::Utility::ErrMsgHandler( int level, Bool_t abort, const char* location, const char* msg )
{
// Translate ROOT error/warning to python

// initialization from gEnv (the default handler will return w/o msg b/c level too low)
   if ( gErrorIgnoreLevel == kUnset )
      ::DefaultErrorHandler( kUnset - 1, kFALSE, "", "" );

   if ( level < gErrorIgnoreLevel )
      return;

// turn warnings into python warnings
   if (level >= kError)
      ::DefaultErrorHandler( level, abort, location, msg );
   else if ( level >= kWarning ) {
   // either printout or raise exception, depending on user settings
      PyErr_WarnExplicit( NULL, (char*)msg, (char*)location, 0, (char*)"ROOT", NULL );
   }
   else
      ::DefaultErrorHandler( level, abort, location, msg );
}


//____________________________________________________________________________
Long_t PyROOT::Utility::InstallMethod( G__ClassInfo* scope, PyObject* callback, 
   const std::string& mtName, const char* signature, void* func, Int_t npar, Long_t extra )
{
   static Long_t s_fid = (Long_t)PyROOT::Utility::InstallMethod;
   ++s_fid;

// Install a python callable method so that CINT can call it

   if ( ! PyCallable_Check( callback ) )
      return 0;

// create a return type (typically masked/wrapped by a TPyReturn) for the method
   G__linked_taginfo pti;
   pti.tagnum = -1;
   pti.tagtype = 'c';
   const char* cname = scope ? scope->Fullname() : 0;
   std::string tname = cname ? std::string( cname ) + "::" + mtName : mtName;
   pti.tagname = tname.c_str();
   int tagnum = G__get_linked_tagnum( &pti );     // creates entry for new names

   if ( scope ) {   // add method to existing scope
      G__MethodInfo meth = scope->AddMethod( pti.tagname, mtName.c_str(), signature, 0, 0, func );
   } else {         // for free functions, add to global scope and add lookup through tp2f
   // setup a connection between the pointer and the name (only the interface method will be
   // called in the end, the tp2f must only be consistent: s_fid is chosen to allow the same
   // C++ callback to serve multiple python objects)
      Long_t hash = 0, len = 0;
      G__hash( mtName.c_str(), hash, len );
      G__lastifuncposition();
      G__memfunc_setup( mtName.c_str(), hash,
        (G__InterfaceMethod)func, tagnum, tagnum, tagnum, 0, npar, 0, 1, 0, signature, (char*)0, (void*)s_fid, 0 );
      G__resetifuncposition();

   // setup a name in the global namespace (does not result in calls, so the signature does
   // not matter; but it makes subsequent GetMethod() calls work)
      G__MethodInfo meth = G__ClassInfo().AddMethod( mtName.c_str(), mtName.c_str(), signature, 1, 0, func );
   }

// and store callback
   Py_INCREF( callback );
   std::map< int, CallInfo_t >::iterator old = s_PyObjectCallbacks.find( tagnum );
   if ( old != s_PyObjectCallbacks.end() ) {
      PyObject* oldp = old->second.first;
      Py_XDECREF( oldp );
   }
   s_PyObjectCallbacks[ tagnum ] = std::make_pair( callback, extra );

// hard to check result ... assume ok
   return s_fid;
}

//____________________________________________________________________________
PyObject* PyROOT::Utility::GetInstalledMethod( int tagnum, Long_t* extra )
{
// Return the CINT-installed python callable, if any
   CallInfo_t cinfo = s_PyObjectCallbacks[ tagnum ];
   if ( extra )
      *extra = cinfo.second;
   return cinfo.first;
}
