// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Jan 2005

// Bindings
#include "PyROOT.h"
#include "PyStrings.h"
#include "Executors.h"
#include "ObjectProxy.h"
#include "TPyBufferFactory.h"
#include "RootWrapper.h"
#include "Utility.h"

// ROOT
#include "TClassEdit.h"           // for ShortType and CleanType
#include "TInterpreter.h"         // for by-value returns
#include "TInterpreterValue.h"    // id.

// Standard
#include <cstring>
#include <utility>
#include <sstream>


//- data ______________________________________________________________________
namespace PyROOT {

   typedef TExecutor* (*ExecutorFactory_t) ();
   typedef std::map< std::string, ExecutorFactory_t > ExecFactories_t;
   ExecFactories_t gExecFactories;

   R__EXTERN PyObject* gNullPtrObject;
}


//- helpers -------------------------------------------------------------------
namespace {

   class GILControl {
   public:
      GILControl( PyROOT::TCallContext* ctxt ) :
         fRelease( ReleasesGIL( ctxt ) ), fSave( nullptr ) {
#ifdef WITH_THREAD
         if ( fRelease ) fSave = PyEval_SaveThread();
#endif
      }
      ~GILControl() {
#ifdef WITH_THREAD
         if ( fRelease ) PyEval_RestoreThread( fSave );
#endif
      }
   private:
      Bool_t fRelease;
      PyThreadState* fSave;
   };

} // unnamed namespace

static inline void GILCallV(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, PyROOT::TCallContext* ctxt ) {
   GILControl gc( ctxt ); 
   Cppyy::CallV( method, self, &ctxt->fArgs );
}

static inline Long_t GILCallL(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, PyROOT::TCallContext* ctxt ) {
   GILControl gc( ctxt );
   return Cppyy::CallL( method, self, &ctxt->fArgs );
}

static inline Double_t GILCallD(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, PyROOT::TCallContext* ctxt ) {
   GILControl gc( ctxt ); 
   return Cppyy::CallD( method, self, &ctxt->fArgs );
}

static inline Char_t* GILCallS(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, PyROOT::TCallContext* ctxt ) {
   GILControl gc( ctxt );
   return Cppyy::CallS( method, self, &ctxt->fArgs );
}

static inline Cppyy::TCppObject_t GILCallO( Cppyy::TCppMethod_t method,
      Cppyy::TCppObject_t self, PyROOT::TCallContext* ctxt, Cppyy::TCppType_t klass ) {
   GILControl gc( ctxt );
   return Cppyy::CallO( method, self, &ctxt->fArgs, klass );
}

static inline Cppyy::TCppObject_t GILCallConstructor(
      Cppyy::TCppMethod_t method, Cppyy::TCppType_t klass, PyROOT::TCallContext* ctxt ) {
   GILControl gc( ctxt );
   return Cppyy::CallConstructor( method, klass, &ctxt->fArgs );
}

static inline PyObject* PyROOT_PyUnicode_FromInt( Int_t c ) {
   if ( c < 0 ) return PyInt_FromLong( c ); // python chars are range(256)
   return PyROOT_PyUnicode_FromFormat( "%c", c );
}

static inline PyObject* PyROOT_PyBool_FromInt( Int_t b ) {
   PyObject* result = (Bool_t)b ? Py_True : Py_False;
   Py_INCREF( result );
   return result;
}

//- executors for built-ins ---------------------------------------------------
PyObject* PyROOT::TBoolExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt )
{
// execute <method> with argument <self, ctxt>, construct python bool return value
   Bool_t retval;
   { GILControl gc( ctxt ); retval = (Bool_t)Cppyy::CallB( method, self, &ctxt->fArgs ); }
   PyObject* result = retval ? Py_True : Py_False;
   Py_INCREF( result );
   return result;
}

//____________________________________________________________________________
PyObject* PyROOT::TBoolConstRefExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt )
{
// execute <method> with argument <self, ctxt>, construct python bool return value
   return PyROOT_PyBool_FromInt( *((Bool_t*)GILCallL( method, self, ctxt )) );
}

//____________________________________________________________________________
PyObject* PyROOT::TCharExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt )
{
// execute <method with argument <self, ctxt>, construct python string return value
// with the single char
   return PyROOT_PyUnicode_FromInt( (Int_t)GILCallL( method, self, ctxt ) );
}

//____________________________________________________________________________
PyObject* PyROOT::TCharConstRefExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt )
{
// execute <method> with argument <self, ctxt>, construct python string return value
// with the single char
   return PyROOT_PyUnicode_FromInt( *((Char_t*)GILCallL( method, self, ctxt )) );
}

//____________________________________________________________________________
PyObject* PyROOT::TUCharExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt )
{
// execute <method> with argument <self, args>, construct python string return value
// with the single char
   return PyROOT_PyUnicode_FromInt( (UChar_t)GILCallL( method, self, ctxt ) );
}

//____________________________________________________________________________
PyObject* PyROOT::TUCharConstRefExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt )
{
// execute <method> with argument <self, ctxt>, construct python string return value
//  with the single char from the pointer return
   return PyROOT_PyUnicode_FromInt( *((UChar_t*)GILCallL( method, self, ctxt )) );
}

//____________________________________________________________________________
PyObject* PyROOT::TIntExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt )
{
// execute <method> with argument <self, ctxt>, construct python int return value
   return PyInt_FromLong( (Int_t)GILCallL( method, self, ctxt ) );
}

//____________________________________________________________________________
PyObject* PyROOT::TLongExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt )
{
// execute <method> with argument <self, ctxt>, construct python long return value
   return PyLong_FromLong( (Long_t)GILCallL( method, self, ctxt ) );
}

//____________________________________________________________________________
PyObject* PyROOT::TULongExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt )
{
// execute <method> with argument <self, ctxt>, construct python unsigned long return value
   return PyLong_FromUnsignedLong( (ULong_t)GILCallL( method, self, ctxt ) );
}

//____________________________________________________________________________
PyObject* PyROOT::TLongLongExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt )
{
// execute <method> with argument <self, ctxt>, construct python long long return value
   Long64_t result;
   { GILControl gc( ctxt ); result = Cppyy::CallLL( method, self, &ctxt->fArgs ); }
   return PyLong_FromLongLong( result );
}

//____________________________________________________________________________
PyObject* PyROOT::TULongLongExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt )
{
// execute <method> with argument <self, ctxt>, construct python unsigned long long return value
   ULong64_t result;
   { GILControl gc( ctxt ); result = Cppyy::CallLL( method, self, &ctxt->fArgs ); }
   return PyLong_FromUnsignedLongLong( result );
}

//____________________________________________________________________________
PyObject* PyROOT::TDoubleExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt )
{
// execute <method> with argument <self, ctxt>, construct python float return value
   return PyFloat_FromDouble( (Double_t)GILCallD( method, self, ctxt ) );
}

//____________________________________________________________________________
Bool_t PyROOT::TRefExecutor::SetAssignable( PyObject* pyobject )
{
// prepare "buffer" for by-ref returns, used with __setitem__
   if ( pyobject != 0 ) {
      Py_INCREF( pyobject );
      fAssignable = pyobject;
      return kTRUE;
   }

   fAssignable = 0;
   return kFALSE;
}

//____________________________________________________________________________
#define PYROOT_IMPLEMENT_BASIC_REFEXECUTOR( name, type, stype, F1, F2 )      \
PyObject* PyROOT::T##name##RefExecutor::Execute(                             \
       Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt )\
{                                                                            \
   type* ref = (type*)GILCallL( method, self, ctxt );                        \
   if ( ! fAssignable )                                                      \
      return F1( (stype)*ref );                                              \
   else {                                                                    \
      *ref = (type)F2( fAssignable );                                        \
      Py_DECREF( fAssignable );                                              \
      fAssignable = 0;                                                       \
      Py_INCREF( Py_None );                                                  \
      return Py_None;                                                        \
   }                                                                         \
}

PYROOT_IMPLEMENT_BASIC_REFEXECUTOR( Bool,   Bool_t,   Long_t,   PyROOT_PyBool_FromInt,    PyLong_AsLong )
PYROOT_IMPLEMENT_BASIC_REFEXECUTOR( Char,   Char_t,   Long_t,   PyROOT_PyUnicode_FromInt, PyLong_AsLong )
PYROOT_IMPLEMENT_BASIC_REFEXECUTOR( UChar,  UChar_t,  ULong_t,  PyROOT_PyUnicode_FromInt, PyLongOrInt_AsULong )
PYROOT_IMPLEMENT_BASIC_REFEXECUTOR( Short,  Short_t,  Long_t,   PyInt_FromLong,     PyLong_AsLong )
PYROOT_IMPLEMENT_BASIC_REFEXECUTOR( UShort, UShort_t, ULong_t,  PyInt_FromLong,     PyLongOrInt_AsULong )
PYROOT_IMPLEMENT_BASIC_REFEXECUTOR( Int,    Int_t,    Long_t,   PyInt_FromLong,     PyLong_AsLong )
PYROOT_IMPLEMENT_BASIC_REFEXECUTOR( UInt,   UInt_t,   ULong_t,  PyLong_FromUnsignedLong, PyLongOrInt_AsULong )
PYROOT_IMPLEMENT_BASIC_REFEXECUTOR( Long,   Long_t,   Long_t,   PyLong_FromLong,    PyLong_AsLong )
PYROOT_IMPLEMENT_BASIC_REFEXECUTOR( ULong,  ULong_t,  ULong_t,  PyLong_FromUnsignedLong, PyLongOrInt_AsULong )
PYROOT_IMPLEMENT_BASIC_REFEXECUTOR(
   LongLong,  Long64_t,  Long64_t,   PyLong_FromLongLong,         PyLong_AsLongLong )
PYROOT_IMPLEMENT_BASIC_REFEXECUTOR(
   ULongLong, ULong64_t, ULong64_t,  PyLong_FromUnsignedLongLong, PyLongOrInt_AsULong64 )
PYROOT_IMPLEMENT_BASIC_REFEXECUTOR( Float,  Float_t,  Double_t, PyFloat_FromDouble, PyFloat_AsDouble )
PYROOT_IMPLEMENT_BASIC_REFEXECUTOR( Double, Double_t, Double_t, PyFloat_FromDouble, PyFloat_AsDouble )
PYROOT_IMPLEMENT_BASIC_REFEXECUTOR(
   LongDouble, LongDouble_t, LongDouble_t, PyFloat_FromDouble, PyFloat_AsDouble )

//____________________________________________________________________________
PyObject* PyROOT::TSTLStringRefExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt ) 
{
// execute <method> with argument <self, ctxt>, return python string return value
   if ( ! fAssignable ) {
      std::string* result = (std::string*)GILCallL( method, self, ctxt );
      return PyROOT_PyUnicode_FromStringAndSize( result->c_str(), result->size() );
   } else {
      std::string* result = (std::string*)GILCallL( method, self, ctxt );
      *result = std::string(
         PyROOT_PyUnicode_AsString( fAssignable ), PyROOT_PyUnicode_GET_SIZE( fAssignable ) );

      Py_DECREF( fAssignable );
      fAssignable = 0;

      Py_INCREF( Py_None );
      return Py_None;
   }
}

//____________________________________________________________________________
PyObject* PyROOT::TVoidExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt ) 
{
// execute <method> with argument <self, ctxt>, return None
   GILCallV( method, self, ctxt );
   Py_INCREF( Py_None );
   return Py_None;
}

//____________________________________________________________________________
PyObject* PyROOT::TCStringExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt ) 
{
// execute <method> with argument <self, ctxt>, construct python string return value
   char* result = (char*)GILCallS( method, self, ctxt );
   if ( ! result ) {
      Py_INCREF( PyStrings::gEmptyString );
      return PyStrings::gEmptyString;
   }

   return PyROOT_PyUnicode_FromString( result );
}


//- pointer/array executors ---------------------------------------------------
PyObject* PyROOT::TVoidArrayExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt ) 
{
// execute <method> with argument <self, ctxt>, construct python long return value
   Long_t* result = (Long_t*)GILCallL( method, self, ctxt );
   if ( ! result ) {
      Py_INCREF( gNullPtrObject );
      return gNullPtrObject;
   }
   return BufFac_t::Instance()->PyBuffer_FromMemory( result, 1 );
}

//____________________________________________________________________________
#define PYROOT_IMPLEMENT_ARRAY_EXECUTOR( name, type )                        \
PyObject* PyROOT::T##name##ArrayExecutor::Execute(                           \
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt )\
{                                                                            \
   return BufFac_t::Instance()->PyBuffer_FromMemory( (type*)GILCallL( method, self, ctxt ) );\
}

PYROOT_IMPLEMENT_ARRAY_EXECUTOR( Bool,   Bool_t )
PYROOT_IMPLEMENT_ARRAY_EXECUTOR( Short,  Short_t )
PYROOT_IMPLEMENT_ARRAY_EXECUTOR( UShort, UShort_t )
PYROOT_IMPLEMENT_ARRAY_EXECUTOR( Int,    Int_t )
PYROOT_IMPLEMENT_ARRAY_EXECUTOR( UInt,   UInt_t )
PYROOT_IMPLEMENT_ARRAY_EXECUTOR( Long,   Long_t )
PYROOT_IMPLEMENT_ARRAY_EXECUTOR( ULong,  ULong_t )
PYROOT_IMPLEMENT_ARRAY_EXECUTOR( Float,  Float_t )
PYROOT_IMPLEMENT_ARRAY_EXECUTOR( Double, Double_t )


//- special cases ------------------------------------------------------------
PyObject* PyROOT::TSTLStringExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt ) 
{
// execute <method> with argument <self, ctxt>, construct python string return value

   static Cppyy::TCppScope_t sSTLStringScope = Cppyy::GetScope( "std::string" );
   TInterpreterValue* value = (TInterpreterValue*)GILCallO( method, self, ctxt, sSTLStringScope );
   if ( ! value || ! value->GetAsPointer() ) {
      Py_INCREF( PyStrings::gEmptyString );
      return PyStrings::gEmptyString;
   }

   std::string* result = (std::string*)value->GetAsPointer();
   PyObject* pyresult =
      PyROOT_PyUnicode_FromStringAndSize( result->c_str(), result->size() );
   delete value;

// last time I checked, this was a no-op, but by convention it is required
   gInterpreter->ClearStack();

   return pyresult;
}

//____________________________________________________________________________
PyObject* PyROOT::TTGlobalExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt ) 
{
// execute <method> with argument <self, ctxt>, construct python proxy object return value
   return BindCppGlobal( (TGlobal*)GILCallL( method, self, ctxt ) );
}

//____________________________________________________________________________
PyObject* PyROOT::TCppObjectExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt ) 
{
// execute <method> with argument <self, ctxt>, construct python proxy object return value
   return BindCppObject( (void*)GILCallL( method, self, ctxt ), fClass );
}

//____________________________________________________________________________
PyObject* PyROOT::TCppObjectByValueExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt ) 
{
// execution will bring a temporary in existence
   Cppyy::TCppObject_t value = GILCallO( method, self, ctxt, fClass );

   if ( ! value ) {
      if ( ! PyErr_Occurred() )         // callee may have set a python error itself
         PyErr_SetString( PyExc_ValueError, "NULL result where temporary expected" );
      return 0;
   }

// the result can then be bound
   ObjectProxy* pyobj = (ObjectProxy*)BindCppObjectNoCast( value, fClass, kFALSE, kTRUE );
   if ( ! pyobj )
      return 0;

// python ref counting will now control this object's life span
   pyobj->HoldOn();
   return (PyObject*)pyobj;
}

//____________________________________________________________________________
PyObject* PyROOT::TCppObjectRefExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt ) 
{
// executor binds the result to the left-hand side, overwriting if an old object
   PyObject* result = BindCppObject( (void*)GILCallL( method, self, ctxt ), fClass );
   if ( ! result || ! fAssignable )
      return result;
   else {
   // this generic code is quite slow compared to its C++ equivalent ...
      PyObject* assign = PyObject_GetAttrString( result, const_cast< char* >( "__assign__" ) );
      if ( ! assign ) {
         PyErr_Clear();
         PyObject* descr = PyObject_Str( result );
         if ( descr && PyBytes_CheckExact( descr ) ) {
            PyErr_Format( PyExc_TypeError, "can not assign to return object (%s)",
                          PyBytes_AS_STRING( descr ) );
         } else {
            PyErr_SetString( PyExc_TypeError, "can not assign to result" );
         }
         Py_XDECREF( descr );
         Py_DECREF( result );
         Py_DECREF( fAssignable ); fAssignable = 0;
         return 0;
      }

      PyObject* res2 = PyObject_CallFunction( assign, const_cast< char* >( "O" ), fAssignable );

      Py_DECREF( assign );
      Py_DECREF( result );
      Py_DECREF( fAssignable ); fAssignable = 0;

      if ( res2 ) {
         Py_DECREF( res2 );             // typically, *this from operator=()
         Py_INCREF( Py_None );
         return Py_None;
      }

      return 0;
   }
}

//____________________________________________________________________________
PyObject* PyROOT::TCppObjectPtrPtrExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt ) 
{
// execute <method> with argument <self, ctxt>, construct python ROOT object return ptr value
   return BindCppObject( (void*)GILCallL( method, self, ctxt ), fClass, kTRUE );
}

//____________________________________________________________________________
PyObject* PyROOT::TCppObjectPtrRefExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt ) 
{
// execute <method> with argument <self, ctxt>, construct python ROOT object (ignoring ref) return ptr value
   return BindCppObject( *(void**)GILCallL( method, self, ctxt ), fClass, kFALSE );
}

//____________________________________________________________________________
PyObject* PyROOT::TCppObjectArrayExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt ) 
{
// execute <method> with argument <self, ctxt>, construct TTupleOfInstances from return value
   return BindCppObjectArray( (void*)GILCallL( method, self, ctxt ), fClass, fArraySize );
}


//____________________________________________________________________________
PyObject* PyROOT::TConstructorExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t klass, TCallContext* ctxt ) 
{
// package return address in PyObject* for caller to handle appropriately (see
// TConstructorHolder for the actual build of the PyObject)
   return (PyObject*)GILCallConstructor( method, (Cppyy::TCppType_t)klass, ctxt );
}

//____________________________________________________________________________
PyObject* PyROOT::TPyObjectExecutor::Execute(
      Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, TCallContext* ctxt ) 
{
// execute <method> with argument <self, ctxt>, return python object
   return (PyObject*)GILCallL( method, self, ctxt );
}


//- factories -----------------------------------------------------------------
PyROOT::TExecutor* PyROOT::CreateExecutor( const std::string& fullType )
{
// The matching of the fulltype to an executor factory goes through up to 4 levels:
//   1) full, qualified match
//   2) drop '&' as by ref/full type is often pretty much the same python-wise
//   3) ROOT classes, either by ref/ptr or by value
//   4) additional special case for enums
//
// If all fails, void is used, which will cause the return type to be ignored on use

// an exactly matching executor is best
   ExecFactories_t::iterator h = gExecFactories.find( fullType );
   if ( h != gExecFactories.end() )
      return (h->second)();

// resolve typedefs etc., and collect qualifiers
   std::string resolvedType = Cppyy::ResolveName( fullType );

// a full, qualified matching executor is preferred
   h = gExecFactories.find( resolvedType );
   if ( h != gExecFactories.end() )
      return (h->second)();

//-- nothing? ok, collect information about the type and possible qualifiers/decorators
   const std::string& cpd = Utility::Compound( resolvedType );
   std::string realType = TClassEdit::ShortType( resolvedType.c_str(), 1 );

// const-ness (dropped by TClassEdit::ShortType) is in general irrelevant
   h = gExecFactories.find( realType + cpd );
   if ( h != gExecFactories.end() )
      return (h->second)();

//-- still nothing? try pointer instead of array (for builtins)
   if ( cpd == "[]" ) {
   // CLING WORKAROUND -- if the type is a fixed-size array, it will have a funky
   // resolved type like MyClass(&)[N], which TClass::GetClass() fails on. So, strip
   // it down:
      realType = TClassEdit::CleanType( realType.substr( 0, realType.rfind("(") ).c_str(), 1 );
   // -- CLING WORKAROUND
      h = gExecFactories.find( realType + "*" );
      if ( h != gExecFactories.end() )
         return (h->second)();         // TODO: use array size
   }

// ROOT classes and special cases (enum)
   TExecutor* result = 0;
   if ( Cppyy::TCppType_t klass = Cppyy::GetScope( realType ) ) {
      if ( cpd == "" )
         result = new TCppObjectByValueExecutor( klass );
      else if ( cpd == "&" )
         result = new TCppObjectRefExecutor( klass );
      else if ( cpd == "**" )
         result = new TCppObjectPtrPtrExecutor( klass );
      else if ( cpd == "*&" || cpd == "&*" )
         result = new TCppObjectPtrRefExecutor( klass );
      else if ( cpd == "[]" ) {
         Py_ssize_t asize = Utility::ArraySize( resolvedType );
         if ( 0 < asize )
            result = new TCppObjectArrayExecutor( klass, asize );
         else
            result = new TCppObjectPtrRefExecutor( klass );
      }
      else
         result = new TCppObjectExecutor( klass );
   } else if ( Cppyy::IsEnum( realType ) ) {
   // enums don't resolve to unsigned ints, but that's what they are ...
      h = gExecFactories.find( "UInt_t" + cpd );
   } else {
   // handle (with warning) unknown types
      std::stringstream s;
      s << "creating executor for unknown type \"" << fullType << "\"" << std::ends;
      PyErr_Warn( PyExc_RuntimeWarning, (char*)s.str().c_str() );
   // void* may work ("user knows best"), void will fail on use of return value
      h = (cpd == "") ? gExecFactories.find( "void" ) : gExecFactories.find( "void*" );
   }

   if ( ! result && h != gExecFactories.end() )
   // executor factory available, use it to create executor
      result = (h->second)();

   return result;                  // may still be null
}

//____________________________________________________________________________
#define PYROOT_EXECUTOR_FACTORY( name )                \
TExecutor* Create##name##Executor()                    \
{                                                      \
   return new T##name##Executor;                       \
}

namespace {

   using namespace PyROOT;

// use macro rather than template for portability ...
   PYROOT_EXECUTOR_FACTORY( Bool )
   PYROOT_EXECUTOR_FACTORY( BoolRef )
   PYROOT_EXECUTOR_FACTORY( BoolConstRef )
   PYROOT_EXECUTOR_FACTORY( Char )
   PYROOT_EXECUTOR_FACTORY( CharRef )
   PYROOT_EXECUTOR_FACTORY( CharConstRef )
   PYROOT_EXECUTOR_FACTORY( UChar )
   PYROOT_EXECUTOR_FACTORY( UCharRef )
   PYROOT_EXECUTOR_FACTORY( UCharConstRef )
   PYROOT_EXECUTOR_FACTORY( ShortRef )
   PYROOT_EXECUTOR_FACTORY( UShortRef )
   PYROOT_EXECUTOR_FACTORY( Int )
   PYROOT_EXECUTOR_FACTORY( IntRef )
   PYROOT_EXECUTOR_FACTORY( UIntRef )
   PYROOT_EXECUTOR_FACTORY( ULong )
   PYROOT_EXECUTOR_FACTORY( ULongRef )
   PYROOT_EXECUTOR_FACTORY( Long )
   PYROOT_EXECUTOR_FACTORY( LongRef )
   PYROOT_EXECUTOR_FACTORY( FloatRef )
   PYROOT_EXECUTOR_FACTORY( Double )
   PYROOT_EXECUTOR_FACTORY( DoubleRef )
   PYROOT_EXECUTOR_FACTORY( LongDoubleRef )
   PYROOT_EXECUTOR_FACTORY( Void )
   PYROOT_EXECUTOR_FACTORY( LongLong )
   PYROOT_EXECUTOR_FACTORY( LongLongRef )
   PYROOT_EXECUTOR_FACTORY( ULongLong )
   PYROOT_EXECUTOR_FACTORY( ULongLongRef )
   PYROOT_EXECUTOR_FACTORY( CString )
   PYROOT_EXECUTOR_FACTORY( VoidArray )
   PYROOT_EXECUTOR_FACTORY( BoolArray )
   PYROOT_EXECUTOR_FACTORY( ShortArray )
   PYROOT_EXECUTOR_FACTORY( UShortArray )
   PYROOT_EXECUTOR_FACTORY( IntArray )
   PYROOT_EXECUTOR_FACTORY( UIntArray )
   PYROOT_EXECUTOR_FACTORY( LongArray )
   PYROOT_EXECUTOR_FACTORY( ULongArray )
   PYROOT_EXECUTOR_FACTORY( FloatArray )
   PYROOT_EXECUTOR_FACTORY( DoubleArray )
   PYROOT_EXECUTOR_FACTORY( STLString )
   PYROOT_EXECUTOR_FACTORY( STLStringRef )
   PYROOT_EXECUTOR_FACTORY( TGlobal )
   PYROOT_EXECUTOR_FACTORY( Constructor )
   PYROOT_EXECUTOR_FACTORY( PyObject )

// executor factories for ROOT types
   typedef std::pair< const char*, ExecutorFactory_t > NFp_t;

   NFp_t factories_[] = {
   // factories for built-ins
      NFp_t( "bool",               &CreateBoolExecutor                ),
      NFp_t( "bool&",              &CreateBoolRefExecutor             ),
      NFp_t( "const bool&",        &CreateBoolConstRefExecutor        ),
      NFp_t( "char",               &CreateCharExecutor                ),
      NFp_t( "signed char",        &CreateCharExecutor                ),
      NFp_t( "unsigned char",      &CreateUCharExecutor               ),
      NFp_t( "char&",              &CreateCharRefExecutor             ),
      NFp_t( "signed char&",       &CreateCharRefExecutor             ),
      NFp_t( "unsigned char&",     &CreateUCharRefExecutor            ),
      NFp_t( "const char&",        &CreateCharConstRefExecutor        ),
      NFp_t( "const signed char&", &CreateCharConstRefExecutor        ),
      NFp_t( "const unsigned char&", &CreateUCharConstRefExecutor     ),
      NFp_t( "short",              &CreateIntExecutor                 ),
      NFp_t( "short&",             &CreateShortRefExecutor            ),
      NFp_t( "unsigned short",     &CreateIntExecutor                 ),
      NFp_t( "unsigned short&",    &CreateUShortRefExecutor           ),
      NFp_t( "int",                &CreateIntExecutor                 ),
      NFp_t( "int&",               &CreateIntRefExecutor              ),
      NFp_t( "unsigned int",       &CreateULongExecutor               ),
      NFp_t( "unsigned int&",      &CreateUIntRefExecutor             ),
      NFp_t( "UInt_t",  /* enum */ &CreateULongExecutor               ),
      NFp_t( "UInt_t&", /* enum */ &CreateUIntRefExecutor             ),
      NFp_t( "long",               &CreateLongExecutor                ),
      NFp_t( "long&",              &CreateLongRefExecutor             ),
      NFp_t( "unsigned long",      &CreateULongExecutor               ),
      NFp_t( "unsigned long&",     &CreateULongRefExecutor            ),
      NFp_t( "long long",          &CreateLongLongExecutor            ),
      NFp_t( "Long64_t",           &CreateLongLongExecutor            ),
      NFp_t( "long long&",         &CreateLongLongRefExecutor         ),
      NFp_t( "Long64_t&",          &CreateLongLongRefExecutor         ),
      NFp_t( "unsigned long long", &CreateULongLongExecutor           ),
      NFp_t( "ULong64_t",          &CreateULongLongExecutor           ),
      NFp_t( "unsigned long long&", &CreateULongLongRefExecutor       ),
      NFp_t( "ULong64_t&",         &CreateULongLongRefExecutor        ),

      NFp_t( "float",              &CreateDoubleExecutor              ),
      NFp_t( "float&",             &CreateFloatRefExecutor            ),
      NFp_t( "double",             &CreateDoubleExecutor              ),
      NFp_t( "double&",            &CreateDoubleRefExecutor           ),
      NFp_t( "long double",        &CreateDoubleExecutor              ),   // TODO: lost precision
      NFp_t( "long double&",       &CreateLongDoubleRefExecutor       ),
      NFp_t( "void",               &CreateVoidExecutor                ),

   // pointer/array factories
      NFp_t( "void*",              &CreateVoidArrayExecutor           ),
      NFp_t( "bool*",              &CreateBoolArrayExecutor           ),
      NFp_t( "short*",             &CreateShortArrayExecutor          ),
      NFp_t( "unsigned short*",    &CreateUShortArrayExecutor         ),
      NFp_t( "int*",               &CreateIntArrayExecutor            ),
      NFp_t( "unsigned int*",      &CreateUIntArrayExecutor           ),
      NFp_t( "long*",              &CreateLongArrayExecutor           ),
      NFp_t( "unsigned long*",     &CreateULongArrayExecutor          ),
      NFp_t( "float*",             &CreateFloatArrayExecutor          ),
      NFp_t( "double*",            &CreateDoubleArrayExecutor         ),

   // factories for special cases
      NFp_t( "const char*",        &CreateCStringExecutor             ),
      NFp_t( "char*",              &CreateCStringExecutor             ),
      NFp_t( "std::string",        &CreateSTLStringExecutor           ),
      NFp_t( "string",             &CreateSTLStringExecutor           ),
      NFp_t( "std::string&",       &CreateSTLStringRefExecutor        ),
      NFp_t( "string&",            &CreateSTLStringRefExecutor        ),
      NFp_t( "TGlobal*",           &CreateTGlobalExecutor             ),
      NFp_t( "__init__",           &CreateConstructorExecutor         ),
      NFp_t( "PyObject*",          &CreatePyObjectExecutor            ),
      NFp_t( "_object*",           &CreatePyObjectExecutor            ),
      NFp_t( "FILE*",              &CreateVoidArrayExecutor           )
   };

   struct InitExecFactories_t {
   public:
      InitExecFactories_t()
      {
      // load all executor factories in the global map 'gExecFactories'
         int nf = sizeof( factories_ ) / sizeof( factories_[ 0 ] );
         for ( int i = 0; i < nf; ++i ) {
            gExecFactories[ factories_[ i ].first ] = factories_[ i ].second;
         }
      }
   } initExecvFactories_;

} // unnamed namespace
