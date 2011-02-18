// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "PyStrings.h"
#include "Utility.h"
#include "ObjectProxy.h"
#include "MethodProxy.h"
#include "FunctionHolder.h"
#include "TCustomPyTypes.h"
#include "RootWrapper.h"
#include "PyCallable.h"
#include "Adapters.h"

// ROOT
#include "TROOT.h"
#include "TObject.h"
#include "TClassEdit.h"
#include "TClassRef.h"
#include "TCollection.h"
#include "TFunction.h"
#include "TMethodArg.h"
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

typedef std::map< std::string, std::string > TC2POperatorMapping_t;
static TC2POperatorMapping_t gC2POperatorMapping;

namespace {

   using namespace PyROOT::Utility;

   struct InitOperatorMapping_t {
   public:
      InitOperatorMapping_t() {
         // gC2POperatorMapping[ "[]" ]  = "__setitem__";   // depends on return type
         // gC2POperatorMapping[ "+" ]   = "__add__";       // depends on # of args (see __pos__)
         // gC2POperatorMapping[ "-" ]   = "__sub__";       // id. (eq. __neg__)
         // gC2POperatorMapping[ "*" ]   = "__mul__";       // double meaning in C++

         gC2POperatorMapping[ "[]" ]  = "__getitem__";
         gC2POperatorMapping[ "()" ]  = "__call__";
         gC2POperatorMapping[ "/" ]   = PYROOT__div__;
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
         gC2POperatorMapping[ "/=" ]  = PYROOT__idiv__;
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

      // the following type mappings are "exact"
         gC2POperatorMapping[ "const char*" ] = "__str__";
         gC2POperatorMapping[ "char*" ]       = "__str__";
         gC2POperatorMapping[ "int" ]         = "__int__";
         gC2POperatorMapping[ "long" ]        = PYROOT__long__;
         gC2POperatorMapping[ "double" ]      = "__float__";

      // the following type mappings are "okay"; the assumption is that they
      // are not mixed up with the ones above or between themselves (and if
      // they are, that it is done consistently)
         gC2POperatorMapping[ "short" ]              = "__int__";
         gC2POperatorMapping[ "unsigned short" ]     = "__int__";
         gC2POperatorMapping[ "unsigned int" ]       = PYROOT__long__;
         gC2POperatorMapping[ "unsigned long" ]      = PYROOT__long__;
         gC2POperatorMapping[ "long long" ]          = PYROOT__long__;
         gC2POperatorMapping[ "unsigned long long" ] = PYROOT__long__;
         gC2POperatorMapping[ "float" ]              = "__float__";

         gC2POperatorMapping[ "->" ]  = "__follow__";       // not an actual python operator
         gC2POperatorMapping[ "=" ]   = "__assign__";       // id.

#if PY_VERSION_HEX < 0x03000000
         gC2POperatorMapping[ "bool" ] = "__nonzero__";
#else
         gC2POperatorMapping[ "bool" ] = "__bool__";
#endif
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
      Py_XDECREF( (PyObject*)method );
      method = MethodProxy_New( label, pyfunc );
      Bool_t isOk = PyObject_SetAttrString( pyclass, const_cast< char* >( label ), (PyObject*)method ) == 0;
      Py_DECREF( method );
      return isOk;
   }

   method->AddMethod( pyfunc );

   Py_DECREF( method );
   return kTRUE;
}

//____________________________________________________________________________
Bool_t PyROOT::Utility::AddUsingToClass( PyObject* pyclass, const char* method )
{
// helper to add base class methods to the derived class one (this covers the
// 'using' cases, which the dictionary does not provide)

   MethodProxy* derivedMethod =
         (MethodProxy*)PyObject_GetAttrString( pyclass, const_cast< char* >( method ) );
   if ( ! MethodProxy_Check( derivedMethod ) ) {
      Py_XDECREF( derivedMethod );
      return kFALSE;
   }

   PyObject* mro = PyObject_GetAttr( pyclass, PyStrings::gMRO );
   if ( ! mro || ! PyTuple_Check( mro ) ) {
      Py_XDECREF( mro );
      Py_DECREF( derivedMethod );
      return kFALSE;
   }

   MethodProxy* baseMethod = 0;
   for ( int i = 1; i < PyTuple_GET_SIZE( mro ); ++i ) {
      baseMethod = (MethodProxy*)PyObject_GetAttrString(
         PyTuple_GET_ITEM( mro, i ), const_cast< char* >( method ) );

      if ( ! baseMethod ) {
         PyErr_Clear();
         continue;
      }

      if ( MethodProxy_Check( baseMethod ) )
         break;

      Py_DECREF( baseMethod );
      baseMethod = 0;
   }

   Py_DECREF( mro );

   if ( ! MethodProxy_Check( baseMethod ) ) {
      Py_XDECREF( baseMethod );
      Py_DECREF( derivedMethod );
      return kFALSE;
   }

   derivedMethod->AddMethod( baseMethod );

   Py_DECREF( baseMethod );
   Py_DECREF( derivedMethod );

   return kTRUE;
}

//____________________________________________________________________________
Bool_t PyROOT::Utility::AddBinaryOperator(
   PyObject* left, PyObject* right, const char* op, const char* label )
{
// install the named operator (op) into the left object's class if such a function
// exists as a global overload; a label must be given if the operator is not in
// gC2POperatorMapping (i.e. if it is ambiguous at the member level)

// this should be a given, nevertheless ...
   if ( ! ObjectProxy_Check( left ) )
      return kFALSE;

// retrieve the class names to match the signature of any found global functions
   std::string rcname = ClassName( right );
   std::string lcname = ClassName( left );
   PyObject* pyclass = PyObject_GetAttr( left, PyStrings::gClass );

   Bool_t result = AddBinaryOperator( pyclass, lcname, rcname, op, label );

   Py_DECREF( pyclass );
   return result;
}

//____________________________________________________________________________
Bool_t PyROOT::Utility::AddBinaryOperator( PyObject* pyclass, const char* op, const char* label )
{
// install binary operator op in pyclass, working on two instances of pyclass
   PyObject* pyname = PyObject_GetAttr( pyclass, PyStrings::gName );
   std::string cname = TClassEdit::ResolveTypedef( PyROOT_PyUnicode_AsString( pyname ) );
   Py_DECREF( pyname ); pyname = 0;

   return AddBinaryOperator( pyclass, cname, cname, op, label );
}

//____________________________________________________________________________
static inline TFunction* FindAndAddOperator( const std::string& lcname, const std::string& rcname,
     const char* op, TCollection* funcs ) {
// helper to find a function with matching signature in 'funcs'
   std::string opname = "operator";
   opname += op;

   TIter ifunc( funcs );

   TFunction* func = 0;
   while ( (func = (TFunction*)ifunc.Next()) ) {
      if ( func->GetListOfMethodArgs()->GetSize() != 2 )
         continue;

      if ( func->GetName() == opname ) {
         if ( ( lcname == TClassEdit::ResolveTypedef( TClassEdit::CleanType(
                  ((TMethodArg*)func->GetListOfMethodArgs()->At(0))->GetTypeName(), 1 ).c_str(), true ) ) &&
              ( rcname == TClassEdit::ResolveTypedef( TClassEdit::CleanType(
                  ((TMethodArg*)func->GetListOfMethodArgs()->At(1))->GetTypeName(), 1 ).c_str(), true ) ) ) {

         // done; break out loop
            return func;
         }

      }
   }

   return 0;
}

Bool_t PyROOT::Utility::AddBinaryOperator( PyObject* pyclass, const std::string& lcname,
   const std::string& rcname, const char* op, const char* label )
{
// find a global function with a matching signature and install the result on pyclass;
// in addition, __gnu_cxx is searched pro-actively (as there's AFAICS no way to unearth
// using information)
   static TClassRef gnucxx( "__gnu_cxx" );

   TFunction* func = 0;
   if ( gnucxx.GetClass() ) {
      func = FindAndAddOperator( lcname, rcname, op, gnucxx->GetListOfMethods() );
      if ( func ) {
         PyCallable* pyfunc = new TFunctionHolder< TScopeAdapter, TMemberAdapter >(
            TScopeAdapter::ByName( "__gnu_cxx" ), func );
         return Utility::AddToClass( pyclass, label ? label : gC2POperatorMapping[ op ].c_str(), pyfunc );
      }
   }

   if ( ! func )
      func = FindAndAddOperator( lcname, rcname, op, gROOT->GetListOfGlobalFunctions( kTRUE ) );

   if ( func ) {
   // found a matching overload; add to class
      PyCallable* pyfunc = new TFunctionHolder< TScopeAdapter, TMemberAdapter >( func );
      return Utility::AddToClass( pyclass, label ? label : gC2POperatorMapping[ op ].c_str(), pyfunc );
   }

   return kFALSE;
}

//____________________________________________________________________________
Bool_t PyROOT::Utility::BuildTemplateName( PyObject*& pyname, PyObject* args, int argoff )
{
// helper to construct the "< type, type, ... >" part of a templated name (either
// for a class as in MakeRootTemplateClass in RootModule.cxx) or for method lookup
// (as in TemplatedMemberHook, below)

   PyROOT_PyUnicode_AppendAndDel( &pyname, PyROOT_PyUnicode_FromString( "<" ) );

   Py_ssize_t nArgs = PyTuple_GET_SIZE( args );
   for ( int i = argoff; i < nArgs; ++i ) {
   // add type as string to name
      PyObject* tn = PyTuple_GET_ITEM( args, i );
      if ( PyROOT_PyUnicode_Check( tn ) )
         PyROOT_PyUnicode_Append( &pyname, tn );
      else if ( PyObject_HasAttr( tn, PyStrings::gName ) ) {
      // this works for type objects
         PyObject* tpName = PyObject_GetAttr( tn, PyStrings::gName );

      // special case for strings
         if ( strcmp( PyROOT_PyUnicode_AsString( tpName ), "str" ) == 0 ) {
            Py_DECREF( tpName );
            tpName = PyROOT_PyUnicode_FromString( "std::string" );
         }

         PyROOT_PyUnicode_AppendAndDel( &pyname, tpName );
      } else {
      // last ditch attempt, works for things like int values
         PyObject* pystr = PyObject_Str( tn );
         if ( ! pystr ) {
            return kFALSE;
         }

         PyROOT_PyUnicode_AppendAndDel( &pyname, pystr );
      }

   // add a comma, as needed
      if ( i != nArgs - 1 )
         PyROOT_PyUnicode_AppendAndDel( &pyname, PyROOT_PyUnicode_FromString( "," ) );
   }

// close template name; prevent '>>', which should be '> >'
   if ( PyROOT_PyUnicode_AsString( pyname )[ PyROOT_PyUnicode_GetSize( pyname ) - 1 ] == '>' )
      PyROOT_PyUnicode_AppendAndDel( &pyname, PyROOT_PyUnicode_FromString( " >" ) );
   else
      PyROOT_PyUnicode_AppendAndDel( &pyname, PyROOT_PyUnicode_FromString( ">" ) );

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
// special case: don't handle character strings here (yes, they're buffers, but not quite)
   if ( PyBytes_Check( pyobject ) )
      return 0;

// attempt to retrieve pointer to buffer interface
   PyBufferProcs* bufprocs = Py_TYPE(pyobject)->tp_as_buffer;

   PySequenceMethods* seqmeths = Py_TYPE(pyobject)->tp_as_sequence;
   if ( seqmeths != 0 && bufprocs != 0
#if  PY_VERSION_HEX < 0x03000000
        && bufprocs->bf_getwritebuffer != 0
        && (*(bufprocs->bf_getsegcount))( pyobject, 0 ) == 1
#else
        && bufprocs->bf_getbuffer != 0
#endif
      ) {

   // get the buffer
#if PY_VERSION_HEX < 0x03000000
      Py_ssize_t buflen = (*(bufprocs->bf_getwritebuffer))( pyobject, 0, &buf );
#else
      Py_buffer bufinfo;
      (*(bufprocs->bf_getbuffer))( pyobject, &bufinfo, PyBUF_WRITABLE );
      buf = (char*)bufinfo.buf;
      Py_ssize_t buflen = bufinfo.len;
#endif

      if ( check == kTRUE ) {
      // determine buffer compatibility (use "buf" as a status flag)
         PyObject* pytc = PyObject_GetAttr( pyobject, PyStrings::gTypeCode );
         if ( pytc != 0 ) {     // for array objects
            if ( PyROOT_PyUnicode_AsString( pytc )[0] != tc )
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
            PyObject* pyvalue2 = PyROOT_PyUnicode_FromFormat(
               (char*)"%s and given element size (%ld) do not match needed (%d)",
               PyROOT_PyUnicode_AsString( pyvalue ),
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
std::string PyROOT::Utility::MapOperatorName( const std::string& name, Bool_t bTakesParams )
{
// map the given C++ operator name on the python equivalent

   if ( 8 < name.size() && name.substr( 0, 8 ) == "operator" ) {
      std::string op = name.substr( 8, std::string::npos );

   // stripping ...
      std::string::size_type start = 0, end = op.size();
      while ( start < end && isspace( op[ start ] ) ) ++start;
      while ( start < end && isspace( op[ end-1 ] ) ) --end;
      op = TClassEdit::ResolveTypedef( op.substr( start, end - start ).c_str(), true );

   // map C++ operator to python equivalent, or made up name if no equivalent exists
      TC2POperatorMapping_t::iterator pop = gC2POperatorMapping.find( op );
      if ( pop != gC2POperatorMapping.end() ) {
         return pop->second;

      } else if ( op == "*" ) {
      // dereference v.s. multiplication of two instances
         return bTakesParams ? "__mul__" : "__deref__";

      } else if ( op == "+" ) {
      // unary positive v.s. addition of two instances
         return bTakesParams ? "__add__" : "__pos__";

      } else if ( op == "-" ) {
      // unary negative v.s. subtraction of two instances
         return bTakesParams ? "__sub__" : "__neg__";

      } else if ( op == "++" ) {
      // prefix v.s. postfix increment
         return bTakesParams ? "__postinc__" : "__preinc__";

      } else if ( op == "--" ) {
      // prefix v.s. postfix decrement
         return bTakesParams ? "__postdec__" : "__predec__";
      }

   }

// might get here, as not all operator methods are handled (new, delete, etc.)
   return name;
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
      char c = cleanName[ipos];
      if ( isspace( c ) ) continue;
      if ( isalnum( c ) || c == '_' || c == '>' ) break;

      compound = c + compound;
   }

   return compound;
}

//____________________________________________________________________________
const std::string PyROOT::Utility::ClassName( PyObject* pyobj )
{
   std::string clname = "<unknown>";
   PyObject* pyclass = PyObject_GetAttr( pyobj, PyStrings::gClass );
   if ( pyclass != 0 ) {
      PyObject* pyname = PyObject_GetAttr( pyclass, PyStrings::gName );

      if ( pyname != 0 ) {
         clname = PyROOT_PyUnicode_AsString( pyname );
         Py_DECREF( pyname );
      } else
         PyErr_Clear();

      Py_DECREF( pyclass );
   } else
      PyErr_Clear();

   return clname;
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
      if ( PyROOT_PyUnicode_Check( value ) ) {
         if ( ! PyErr_GivenExceptionMatches( etype, PyExc_IndexError ) )
            PyROOT_PyUnicode_AppendAndDel( &value, PyROOT_PyUnicode_FromString( (char*)"\n  " ) );
         PyROOT_PyUnicode_AppendAndDel( &value, PyROOT_PyUnicode_FromString( msg ) );
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
   const std::string& mtName, const char* rtype, const char* signature,
   void* func, Int_t npar, Long_t extra )
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
   std::string tagname;                     // used as a buffer
   if ( rtype ) {
      tagname = rtype;
   } else {
      const char* cname = scope ? scope->Fullname() : 0;
      tagname = cname ? std::string( cname ) + "::" + mtName : mtName;
   }
   pti.tagname = tagname.c_str();
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

//____________________________________________________________________________
PyObject* PyROOT::Utility::PyErr_Occurred_WithGIL()
{
// re-acquire the GIL before calling PyErr_Occurred() in case it has been
// released; note that the p2.2 code assumes that there are no callbacks in
// C++ to python (or at least none returning errors)
#if PY_VERSION_HEX >= 0x02030000
   PyGILState_STATE gstate = PyGILState_Ensure();
   PyObject* e = PyErr_Occurred();
   PyGILState_Release( gstate );
#else
   if ( PyThreadState_GET() )
      return PyErr_Occurred();
   PyObject* e = 0;
#endif

   return e;
}
