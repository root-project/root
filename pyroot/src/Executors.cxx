// @(#)root/pyroot:$Name:  $:$Id: Executors.cxx,v 1.11 2005/08/10 05:25:41 brun Exp $
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
   return PyLong_FromLong( (Long_t)func->ExecInt( self ) );
}

//____________________________________________________________________________
PyObject* PyROOT::TCharExecutor::Execute( G__CallFunc* func, void* self )
{
   Long_t result = func->ExecInt( self );
   char c[2]; c[1] = '\0';
   c[0] = (char) result;
   return PyString_FromString( c );
}

//____________________________________________________________________________
PyObject* PyROOT::TIntExecutor::Execute( G__CallFunc* func, void* self )
{
   return PyInt_FromLong( (Long_t)func->ExecInt( self ) );
}

//____________________________________________________________________________
PyObject* PyROOT::TULongExecutor::Execute( G__CallFunc* func, void* self )
{
   return PyLong_FromUnsignedLong( (ULong_t)func->ExecInt( self ) );
}

//____________________________________________________________________________
PyObject* PyROOT::TLongLongExecutor::Execute( G__CallFunc* func, void* self )
{
   return PyLong_FromLongLong( (Long64_t)G__Longlong( func->Execute( self ) ) );
}

//____________________________________________________________________________
PyObject* PyROOT::TDoubleExecutor::Execute( G__CallFunc* func, void* self )
{
   return PyFloat_FromDouble( (double)func->ExecDouble( self ) );
}

//____________________________________________________________________________
PyObject* PyROOT::TVoidExecutor::Execute( G__CallFunc* func, void* self )
{
   func->Exec( self );
   Py_INCREF( Py_None );
   return Py_None;
}

//____________________________________________________________________________
PyObject* PyROOT::TCStringExecutor::Execute( G__CallFunc* func, void* self )
{
   char* result = (char*)func->ExecInt( self );
   if ( ! result )
      return PyString_FromString( "" );

   return PyString_FromString( result );
}


//- pointer/array executors ---------------------------------------------------
PyObject* PyROOT::TVoidArrayExecutor::Execute( G__CallFunc* func, void* self )
{  
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
   std::string* result = (std::string*)func->ExecInt( self );
   if ( ! result )
      return PyString_FromString( "" );

   return PyString_FromString( (*result).c_str() );
}

//____________________________________________________________________________
PyObject* PyROOT::TTGlobalExecutor::Execute( G__CallFunc* func, void* self )
{
   return BindRootGlobal( (TGlobal*)func->ExecInt( self ) );
}

//____________________________________________________________________________
PyObject* PyROOT::TRootObjectExecutor::Execute( G__CallFunc* func, void* self )
{
   return BindRootObject( (void*)func->ExecInt( self ), fClass );
}

//____________________________________________________________________________
PyObject* PyROOT::TRootObjectByValueExecutor::Execute( G__CallFunc* func, void* self )
{
// execution will bring a temporary in existence ...
   void* result1 = (void*)func->ExecInt( self );
   if ( ! result1 )
      return 0;

// ... which must be copied to retain ownership, then released
   void* result2 = result1;
   if ( fClass->GetClassInfo() && fClass->GetClassInfo()->Linkage() != -1 ) {
      result2 = new char[ fClass->Size() ];
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


//- factories -----------------------------------------------------------------
PyROOT::TExecutor* PyROOT::CreateExecutor( const std::string& fullType )
{
   TExecutor* result = 0;
   std::string realType = TClassEdit::ShortType( G__TypeInfo( fullType.c_str() ).TrueName(), 1 );

// select and set executor
   const std::string& cpd = Utility::Compound( fullType );
   const char* q = cpd == "*" ? "*" : "";

   ExecFactories_t::iterator h = gExecFactories.find( realType + q );
   if ( h == gExecFactories.end() ) {
      TClass* klass = gROOT->GetClass( realType.c_str() );
      if ( klass != 0 ) {
         result = cpd != ""  ? \
            new TRootObjectExecutor( klass ) : new TRootObjectByValueExecutor( klass );
      } else {
      // could still be an enum ...
         G__TypeInfo ti( fullType.c_str() );
         if ( ti.Property() & G__BIT_ISENUM )
            h = gExecFactories.find( "UInt_t" );
         else {
            std::cerr << "return type in not handled (using void): " << fullType << std::endl;
            h = gExecFactories.find( "void" );
         }
      }
   }

   if ( ! result && h != gExecFactories.end() )
   // executor factory available, use it to create executor
      result = (h->second)();

   return result;                  // may still be null
}

//____________________________________________________________________________
#define PYROOT_EXECUTOR_FACTORY( name )                \
TExecutor* Create##name()                              \
{                                                      \
   return new T##name;                                 \
}

namespace {

   using namespace PyROOT;

// us macro rather than template for portability ...
   PYROOT_EXECUTOR_FACTORY( CharExecutor )
   PYROOT_EXECUTOR_FACTORY( IntExecutor )
   PYROOT_EXECUTOR_FACTORY( ULongExecutor )
   PYROOT_EXECUTOR_FACTORY( LongExecutor )
   PYROOT_EXECUTOR_FACTORY( DoubleExecutor )
   PYROOT_EXECUTOR_FACTORY( VoidExecutor )
   PYROOT_EXECUTOR_FACTORY( LongLongExecutor )
   PYROOT_EXECUTOR_FACTORY( CStringExecutor )
   PYROOT_EXECUTOR_FACTORY( VoidArrayExecutor )
   PYROOT_EXECUTOR_FACTORY( ShortArrayExecutor )
   PYROOT_EXECUTOR_FACTORY( UShortArrayExecutor )
   PYROOT_EXECUTOR_FACTORY( IntArrayExecutor )
   PYROOT_EXECUTOR_FACTORY( UIntArrayExecutor )
   PYROOT_EXECUTOR_FACTORY( LongArrayExecutor )
   PYROOT_EXECUTOR_FACTORY( ULongArrayExecutor )
   PYROOT_EXECUTOR_FACTORY( FloatArrayExecutor )
   PYROOT_EXECUTOR_FACTORY( DoubleArrayExecutor )
   PYROOT_EXECUTOR_FACTORY( STLStringExecutor )
   PYROOT_EXECUTOR_FACTORY( TGlobalExecutor )
   PYROOT_EXECUTOR_FACTORY( ConstructorExecutor )

// executor factories for ROOT types
   typedef std::pair< const char*, ExecutorFactory_t > NFp_t;

   NFp_t factories_[] = {
   // factories for built-ins
      NFp_t( "char",               &CreateCharExecutor                ),
      NFp_t( "unsigned char",      &CreateCharExecutor                ),
      NFp_t( "short",              &CreateIntExecutor                 ),
      NFp_t( "unsigned short",     &CreateIntExecutor                 ),
      NFp_t( "int",                &CreateIntExecutor                 ),
      NFp_t( "unsigned int",       &CreateULongExecutor               ),
      NFp_t( "UInt_t", /* enum */  &CreateULongExecutor               ),
      NFp_t( "long",               &CreateLongExecutor                ),
      NFp_t( "unsigned long",      &CreateULongExecutor               ),
      NFp_t( "long long",          &CreateLongLongExecutor            ),
      NFp_t( "float",              &CreateDoubleExecutor              ),
      NFp_t( "double",             &CreateDoubleExecutor              ),
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
      NFp_t( "TGlobal*",           &CreateTGlobalExecutor             ),
      NFp_t( "__init__",           &CreateConstructorExecutor         )
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
