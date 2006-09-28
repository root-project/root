// @(#)root/pyroot:$Name:  $:$Id: Executors.cxx,v 1.20 2006/09/28 19:59:12 brun Exp $
// Author: Wim Lavrijsen, Jan 2005

// Bindings
#include "PyROOT.h"
#include "Executors.h"
#include "ObjectProxy.h"
#include "PyBufferFactory.h"
#include "RootWrapper.h"
#include "Utility.h"

// ROOT
#include "TClass.h"
#include "TClassEdit.h"
#include "DllImport.h"

// CINT
#include "Api.h"

// Standard
#include <utility>
#include <Riostream.h>


//- data ______________________________________________________________________
PyROOT::ExecFactories_t PyROOT::gExecFactories;


//- executors for built-ins ---------------------------------------------------
PyObject* PyROOT::TLongExecutor::Execute( G__CallFunc* func, void* self )
{
// execute <func> with argument <self>, construct python long return value
   return PyLong_FromLong( (Long_t)func->ExecInt( self ) );
}

//____________________________________________________________________________
PyObject* PyROOT::TCharExecutor::Execute( G__CallFunc* func, void* self )
{
// execute <func> with argument <self>, construct python string return value
   return PyString_FromFormat( "%c", (int)func->ExecInt( self ) );
}

//____________________________________________________________________________
PyObject* PyROOT::TIntExecutor::Execute( G__CallFunc* func, void* self )
{
// execute <func> with argument <self>, construct python int return value
   return PyInt_FromLong( (Long_t)func->ExecInt( self ) );
}

//____________________________________________________________________________
PyObject* PyROOT::TULongExecutor::Execute( G__CallFunc* func, void* self )
{
// execute <func> with argument <self>, construct python unsigned long return value
   return PyLong_FromUnsignedLong( (ULong_t)func->ExecInt( self ) );
}

//____________________________________________________________________________
PyObject* PyROOT::TLongLongExecutor::Execute( G__CallFunc* func, void* self )
{
// execute <func> with argument <self>, construct python long long return value
   return PyLong_FromLongLong( (Long64_t)G__Longlong( func->Execute( self ) ) );
}

//____________________________________________________________________________
PyObject* PyROOT::TULongLongExecutor::Execute( G__CallFunc* func, void* self )
{
// execute <func> with argument <self>, construct python unsigned long long return value
   return PyLong_FromUnsignedLongLong( (ULong64_t)G__ULonglong( func->Execute( self ) ) );
}

//____________________________________________________________________________
PyObject* PyROOT::TDoubleExecutor::Execute( G__CallFunc* func, void* self )
{
// execute <func> with argument <self>, construct python float return value
   return PyFloat_FromDouble( (double)func->ExecDouble( self ) );
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
#define PYROOT_IMPLEMENT_BASIC_REFEXECUTOR( name, type, stype, F1, F2, CF )  \
PyObject* PyROOT::T##name##RefExecutor::Execute( G__CallFunc* func, void* self )\
{                                                                            \
   if ( ! fAssignable )                                                      \
      return F1( (stype)func->CF( self ) );                                  \
   else {                                                                    \
      const G__value& result = func->Execute( self );                        \
      *((type*)result.ref) = (type)F2( fAssignable );                        \
      Py_DECREF( fAssignable );                                              \
      fAssignable = 0;                                                       \
      Py_INCREF( Py_None );                                                  \
      return Py_None;                                                        \
   }                                                                         \
}

PYROOT_IMPLEMENT_BASIC_REFEXECUTOR( Int,    Int_t,    Long_t,   PyInt_FromLong,     PyLong_AsLong,    ExecInt )
PYROOT_IMPLEMENT_BASIC_REFEXECUTOR( Long,   Long_t,   Long_t,   PyLong_FromLong,    PyLong_AsLong,    ExecInt )
PYROOT_IMPLEMENT_BASIC_REFEXECUTOR( Float,  Float_t,  Double_t, PyFloat_FromDouble, PyFloat_AsDouble, ExecDouble )
PYROOT_IMPLEMENT_BASIC_REFEXECUTOR( Double, Double_t, Double_t, PyFloat_FromDouble, PyFloat_AsDouble, ExecDouble )

//____________________________________________________________________________
PyObject* PyROOT::TSTLStringRefExecutor::Execute( G__CallFunc* func, void* self )
{
// execute <func> with argument <self>, return python string return value
   if ( ! fAssignable ) {
      return PyString_FromString( ((std::string*)func->ExecInt( self ))->c_str() );
   } else {
      std::string* result = (std::string*)func->ExecInt( self );
      *result = std::string( PyString_AsString( fAssignable ) );

      Py_DECREF( fAssignable );
      fAssignable = 0;

      Py_INCREF( Py_None );
      return Py_None;
   }
}

//____________________________________________________________________________
PyObject* PyROOT::TVoidExecutor::Execute( G__CallFunc* func, void* self )
{
// execute <func> with argument <self>, return None
   func->Exec( self );
   Py_INCREF( Py_None );
   return Py_None;
}

//____________________________________________________________________________
PyObject* PyROOT::TCStringExecutor::Execute( G__CallFunc* func, void* self )
{
// execute <func> with argument <self>, construct python string return value
   char* result = (char*)func->ExecInt( self );
   if ( ! result )
      return PyString_FromString( "" );

   return PyString_FromString( result );
}


//- pointer/array executors ---------------------------------------------------
PyObject* PyROOT::TVoidArrayExecutor::Execute( G__CallFunc* func, void* self )
{
// execute <func> with argument <self>, construct python long return value
   return PyLong_FromVoidPtr( (void*)func->ExecInt( self ) );
}

//____________________________________________________________________________
#define PYROOT_IMPLEMENT_ARRAY_EXECUTOR( name, type )                        \
PyObject* PyROOT::T##name##ArrayExecutor::Execute( G__CallFunc* func, void* self )\
{                                                                            \
   return BufFac_t::Instance()->PyBuffer_FromMemory( (type*)func->ExecInt( self ) );\
}

PYROOT_IMPLEMENT_ARRAY_EXECUTOR( Short,  Short_t )
PYROOT_IMPLEMENT_ARRAY_EXECUTOR( UShort, UShort_t )
PYROOT_IMPLEMENT_ARRAY_EXECUTOR( Int,    Int_t )
PYROOT_IMPLEMENT_ARRAY_EXECUTOR( UInt,   UInt_t )
PYROOT_IMPLEMENT_ARRAY_EXECUTOR( Long,   Long_t )
PYROOT_IMPLEMENT_ARRAY_EXECUTOR( ULong,  ULong_t )
PYROOT_IMPLEMENT_ARRAY_EXECUTOR( Float,  Float_t )
PYROOT_IMPLEMENT_ARRAY_EXECUTOR( Double, Double_t )


//- special cases ------------------------------------------------------------
PyObject* PyROOT::TSTLStringExecutor::Execute( G__CallFunc* func, void* self )
{
// execute <func> with argument <self>, construct python string return value
   std::string* result = (std::string*)func->ExecInt( self );
   if ( ! result )
      return PyString_FromString( "" );

   return PyString_FromString( (*result).c_str() );
}

//____________________________________________________________________________
PyObject* PyROOT::TTGlobalExecutor::Execute( G__CallFunc* func, void* self )
{
// execute <func> with argument <self>, construct python ROOT object return value
   return BindRootGlobal( (TGlobal*)func->ExecInt( self ) );
}

//____________________________________________________________________________
PyObject* PyROOT::TRootObjectExecutor::Execute( G__CallFunc* func, void* self )
{
// execute <func> with argument <self>, construct python ROOT object return value
   return BindRootObject( (void*)func->ExecInt( self ), fClass );
}

//____________________________________________________________________________
PyObject* PyROOT::TRootObjectByValueExecutor::Execute( G__CallFunc* func, void* self )
{
// execution will bring a temporary in existence ...
   void* result1 = (void*)func->ExecInt( self );
   if ( ! result1 ) {
      if ( ! PyErr_Occurred() )         // callee may have set a python error itself
         PyErr_SetString( PyExc_ValueError, "NULL result where temporary expected" );
      return 0;
   }

// ... which must be copied to retain ownership, then released
   void* result2 = result1;
   if ( fClass->GetClassInfo() && fClass->GetClassInfo()->Linkage() != -1 ) {
      result2 = malloc( fClass->Size() );
      memcpy( result2, result1, fClass->Size() );
   }
   G__pop_tempobject();            // doesn't call dtor

// the final result can then be bound
   ObjectProxy* pyobj = (ObjectProxy*)BindRootObjectNoCast( result2, fClass );
   if ( ! pyobj )
      return 0;

// python ref counting will now control the object life span
   pyobj->fFlags |= ObjectProxy::kIsOwner;
   return (PyObject*)pyobj;
}

//____________________________________________________________________________
PyObject* PyROOT::TConstructorExecutor::Execute( G__CallFunc* func, void* klass )
{
// package return address in PyObject* for caller to handle appropriately
   return (PyObject*)func->ExecInt( klass );
}

//____________________________________________________________________________
PyObject* PyROOT::TPyObjectExecutor::Execute( G__CallFunc* func, void* self )
{
// execute <func> with argument <self>, return python object
   return (PyObject*)func->ExecInt( self );
}


//- factories -----------------------------------------------------------------
PyROOT::TExecutor* PyROOT::CreateExecutor( const std::string& fullType )
{
// The matching of the fulltype to an executor factory goes through up to 4 levels:
//   1) full, unqualified match
//   2) drop '&' as as by ref/full type is often pretty much the same python-wise
//   3) ROOT classes, either by ref/ptr or by value
//   4) additional special case for enums
//
// If all fails, void is used, which will cause the return type to be ignored on use

// resolve typedefs etc., and collect qualifiers
   G__TypeInfo ti( fullType.c_str() );
   std::string resolvedType = ti.TrueName();
   if ( ! ti.IsValid() )
      resolvedType = fullType;     // otherwise, resolvedType will be "(unknown)"
   const std::string& cpd = Utility::Compound( resolvedType );
   std::string realType = TClassEdit::ShortType( resolvedType.c_str(), 1 );

// a full, unqualified matching executor is preferred
   ExecFactories_t::iterator h = gExecFactories.find( realType + cpd );
   if ( h != gExecFactories.end() )
      return (h->second)();

// accept ref as by value
   if ( cpd == "&" ) {
      h = gExecFactories.find( realType );
      if ( h != gExecFactories.end() )
         return (h->second)();
   }

// ROOT classes and special cases (enum)
   TExecutor* result = 0;
   if ( TClass* klass = gROOT->GetClass( realType.c_str() ) ) {
      result = cpd != ""  ? \
         new TRootObjectExecutor( klass ) : new TRootObjectByValueExecutor( klass );
   } else {
   // could still be an enum ...
      if ( ti.Property() & G__BIT_ISENUM )
         h = gExecFactories.find( "UInt_t" );
      else {
         std::cerr << "return type not handled (using void): " << fullType << std::endl;
         h = gExecFactories.find( "void" );
      }
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
   PYROOT_EXECUTOR_FACTORY( Char )
   PYROOT_EXECUTOR_FACTORY( Int )
   PYROOT_EXECUTOR_FACTORY( IntRef )
   PYROOT_EXECUTOR_FACTORY( ULong )
   PYROOT_EXECUTOR_FACTORY( Long )
   PYROOT_EXECUTOR_FACTORY( LongRef )
   PYROOT_EXECUTOR_FACTORY( FloatRef )
   PYROOT_EXECUTOR_FACTORY( Double )
   PYROOT_EXECUTOR_FACTORY( DoubleRef )
   PYROOT_EXECUTOR_FACTORY( Void )
   PYROOT_EXECUTOR_FACTORY( LongLong )
   PYROOT_EXECUTOR_FACTORY( ULongLong )
   PYROOT_EXECUTOR_FACTORY( CString )
   PYROOT_EXECUTOR_FACTORY( VoidArray )
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
      NFp_t( "char",               &CreateCharExecutor                ),
      NFp_t( "unsigned char",      &CreateCharExecutor                ),
      NFp_t( "short",              &CreateIntExecutor                 ),
      NFp_t( "unsigned short",     &CreateIntExecutor                 ),
      NFp_t( "int",                &CreateIntExecutor                 ),
      NFp_t( "int&",               &CreateIntRefExecutor              ),
      NFp_t( "unsigned int",       &CreateULongExecutor               ),
      NFp_t( "UInt_t", /* enum */  &CreateULongExecutor               ),
      NFp_t( "long",               &CreateLongExecutor                ),
      NFp_t( "long&",              &CreateLongRefExecutor             ),
      NFp_t( "unsigned long",      &CreateULongExecutor               ),
      NFp_t( "long long",          &CreateLongLongExecutor            ),
      NFp_t( "unsigned long long", &CreateULongLongExecutor           ),
      NFp_t( "float",              &CreateDoubleExecutor              ),
      NFp_t( "float&",             &CreateFloatRefExecutor            ),
      NFp_t( "double",             &CreateDoubleExecutor              ),
      NFp_t( "double&",            &CreateDoubleRefExecutor           ),
      NFp_t( "void",               &CreateVoidExecutor                ),
      NFp_t( "bool",               &CreateIntExecutor                 ),
      NFp_t( "const char*",        &CreateCStringExecutor             ),
      NFp_t( "char*",              &CreateCStringExecutor             ),

   // pointer/array factories
      NFp_t( "void*",              &CreateVoidArrayExecutor           ),
      NFp_t( "short*",             &CreateShortArrayExecutor          ),
      NFp_t( "unsigned short*",    &CreateUShortArrayExecutor         ),
      NFp_t( "int*",               &CreateIntArrayExecutor            ),
      NFp_t( "unsigned int*",      &CreateUIntArrayExecutor           ),
      NFp_t( "long*",              &CreateLongArrayExecutor           ),
      NFp_t( "unsigned long*",     &CreateULongArrayExecutor          ),
      NFp_t( "float*",             &CreateFloatArrayExecutor          ),
      NFp_t( "double*",            &CreateDoubleArrayExecutor         ),

   // factories for special cases
      NFp_t( "std::string",        &CreateSTLStringExecutor           ),
      NFp_t( "string",             &CreateSTLStringExecutor           ),
      NFp_t( "std::string&",       &CreateSTLStringRefExecutor        ),
      NFp_t( "string&",            &CreateSTLStringRefExecutor        ),
      NFp_t( "TGlobal*",           &CreateTGlobalExecutor             ),
      NFp_t( "__init__",           &CreateConstructorExecutor         ),
      NFp_t( "PyObject*",          &CreatePyObjectExecutor            ),
      NFp_t( "_object*",           &CreatePyObjectExecutor            )
   };

   struct InitExecFactories_t {
   public:
      InitExecFactories_t()
      {
         int nf = sizeof( factories_ ) / sizeof( factories_[ 0 ] );
         for ( int i = 0; i < nf; ++i ) {
            gExecFactories[ factories_[ i ].first ] = factories_[ i ].second;
         }
      }
   } initExecvFactories_;

} // unnamed namespace
