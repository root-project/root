// @(#)root/pyroot:$Name:  $:$Id: Executors.cxx,v 1.1 2005/03/04 07:44:11 brun Exp $
// Author: Wim Lavrijsen, Jan 2005

// Bindings
#include "PyROOT.h"
#include "Executors.h"
#include "PyBufferFactory.h"
#include "RootWrapper.h"

// ROOT
#include "Rtypes.h"
#include "TClass.h"

// CINT
#include "Api.h"

// Standard
#include <utility>


//- data ______________________________________________________________________
PyROOT::ExecFactories_t PyROOT::gExecFactories;


//- executors for built-ins ---------------------------------------------------
PyObject* PyROOT::CharExecutor::Execute( G__CallFunc* func, void* self )
{
   long result = func->ExecInt( self );
   char c[2]; c[1] = '\0';
   c[0] = (char) result;
   return PyString_FromString( c );
}

//____________________________________________________________________________
PyObject* PyROOT::IntExecutor::Execute( G__CallFunc* func, void* self )
{
   return PyInt_FromLong( (long)func->ExecInt( self ) );
}

//____________________________________________________________________________
PyObject* PyROOT::UnsignedExecutor::Execute( G__CallFunc* func, void* self )
{
   return PyLong_FromUnsignedLong( (unsigned long)func->ExecInt( self ) );
}

//____________________________________________________________________________
PyObject* PyROOT::LongExecutor::Execute( G__CallFunc* func, void* self )
{
   return PyLong_FromLong( (long)func->ExecInt( self ) );
}

//____________________________________________________________________________
PyObject* PyROOT::DoubleExecutor::Execute( G__CallFunc* func, void* self )
{
   return PyFloat_FromDouble( (double)func->ExecDouble( self ) );
}

//____________________________________________________________________________
PyObject* PyROOT::VoidExecutor::Execute( G__CallFunc* func, void* self )
{
   func->Exec( self );
   Py_INCREF( Py_None );
   return Py_None;
}

//____________________________________________________________________________
PyObject* PyROOT::LongLongExecutor::Execute( G__CallFunc* func, void* self )
{
   return PyLong_FromLongLong( (Long64_t)func->ExecDouble( self ) );
}

//____________________________________________________________________________
PyObject* PyROOT::CStringExecutor::Execute( G__CallFunc* func, void* self )
{
   char* result = (char*)func->ExecInt( self );
   if ( ! result )
      return PyString_FromString( "" );

   return PyString_FromString( result );
}


//- pointer/array executors ---------------------------------------------------
PyObject* PyROOT::VoidArrayExecutor::Execute( G__CallFunc* func, void* self )
{
   return PyLong_FromVoidPtr( (void*)func->ExecInt( self ) );
}

//____________________________________________________________________________
PyObject* PyROOT::IntArrayExecutor::Execute( G__CallFunc* func, void* self )
{
   return PyBufferFactory::Instance()->PyBuffer_FromMemory( (int*)func->ExecInt( self ) );
}

//____________________________________________________________________________
PyObject* PyROOT::LongArrayExecutor::Execute( G__CallFunc* func, void* self )
{
   return PyBufferFactory::Instance()->PyBuffer_FromMemory( (long*)func->ExecInt( self ) );
}

//____________________________________________________________________________
PyObject* PyROOT::FloatArrayExecutor::Execute( G__CallFunc* func, void* self )
{
   return PyBufferFactory::Instance()->PyBuffer_FromMemory( (float*)func->ExecInt( self ) );
}

//____________________________________________________________________________
PyObject* PyROOT::DoubleArrayExecutor::Execute( G__CallFunc* func, void* self )
{
   return PyBufferFactory::Instance()->PyBuffer_FromMemory( (double*)func->ExecInt( self ) );
}


//- special cases ------------------------------------------------------------
PyObject* PyROOT::STLStringExecutor::Execute( G__CallFunc* func, void* self )
{
   std::string* result = (std::string*)func->ExecInt( self );
   if ( ! result )
      return PyString_FromString( "" );

   return PyString_FromString( (*result).c_str() );
}

//____________________________________________________________________________
PyObject* PyROOT::TGlobalExecutor::Execute( G__CallFunc* func, void* self )
{
   return BindRootGlobal( (TGlobal*)func->ExecInt( self ) );
}

//____________________________________________________________________________
PyObject* PyROOT::RootObjectExecutor::Execute( G__CallFunc* func, void* self )
{
   return BindRootObject( (void*)func->ExecInt( self ), fClass );
}

//____________________________________________________________________________
PyObject* PyROOT::ConstructorExecutor::Execute( G__CallFunc* func, void* klass )
{
// package return address in PyObject* for caller to handle appropriately
   return (PyObject*)func->ExecInt( klass );
}


//- factories -----------------------------------------------------------------
#define PYROOT_EXECUTOR_FACTORY( name )                \
Executor* Create##name()                               \
{                                                      \
   return new name ();                                 \
}

namespace {

   using namespace PyROOT;

// us macro rather than template for portability ...
   PYROOT_EXECUTOR_FACTORY( CharExecutor )
   PYROOT_EXECUTOR_FACTORY( IntExecutor )
   PYROOT_EXECUTOR_FACTORY( UnsignedExecutor )
   PYROOT_EXECUTOR_FACTORY( LongExecutor )
   PYROOT_EXECUTOR_FACTORY( DoubleExecutor )
   PYROOT_EXECUTOR_FACTORY( VoidExecutor )
   PYROOT_EXECUTOR_FACTORY( LongLongExecutor )
   PYROOT_EXECUTOR_FACTORY( CStringExecutor )
   PYROOT_EXECUTOR_FACTORY( VoidArrayExecutor )
   PYROOT_EXECUTOR_FACTORY( IntArrayExecutor )
   PYROOT_EXECUTOR_FACTORY( LongArrayExecutor )
   PYROOT_EXECUTOR_FACTORY( FloatArrayExecutor )
   PYROOT_EXECUTOR_FACTORY( DoubleArrayExecutor )
   PYROOT_EXECUTOR_FACTORY( STLStringExecutor )
   PYROOT_EXECUTOR_FACTORY( TGlobalExecutor )
   PYROOT_EXECUTOR_FACTORY( ConstructorExecutor )

// executor factories for ROOT types
   typedef std::pair< const char*, ExecutorFactory_t > ncp_t;

   ncp_t factories_[] = {
   // factories for built-ins
      ncp_t( "char",               &CreateCharExecutor                ),
      ncp_t( "unsigned char",      &CreateCharExecutor                ),
      ncp_t( "short",              &CreateIntExecutor                 ),
      ncp_t( "unsigned short",     &CreateIntExecutor                 ),
      ncp_t( "int",                &CreateIntExecutor                 ),
      ncp_t( "unsigned int",       &CreateUnsignedExecutor            ),
      ncp_t( "long",               &CreateLongExecutor                ),
      ncp_t( "unsigned long",      &CreateUnsignedExecutor            ),
      ncp_t( "long long",          &CreateLongLongExecutor            ),
      ncp_t( "float",              &CreateDoubleExecutor              ),
      ncp_t( "double",             &CreateDoubleExecutor              ),
      ncp_t( "void",               &CreateVoidExecutor                ),
      ncp_t( "bool",               &CreateIntExecutor                 ),
      ncp_t( "const char*",        &CreateCStringExecutor             ),
      ncp_t( "char*",              &CreateCStringExecutor             ),

   // pointer/array factories
      ncp_t( "void*",              &CreateVoidArrayExecutor           ),
      ncp_t( "int*",               &CreateIntArrayExecutor            ),
      ncp_t( "long*",              &CreateLongArrayExecutor           ),
      ncp_t( "float*",             &CreateFloatArrayExecutor          ),
      ncp_t( "double*",            &CreateDoubleArrayExecutor         ),

   // factories for special cases
      ncp_t( "std::string",        &CreateSTLStringExecutor           ),
      ncp_t( "TGlobal*",           &CreateTGlobalExecutor             ),
      ncp_t( "__init__",           &CreateConstructorExecutor         )
   };

   const int nFactories_ = sizeof( factories_ ) / sizeof( factories_[ 0 ] );

   class InitExecFactories_ {
   public:
      InitExecFactories_()
      {
         for ( int i = 0; i < nFactories_; ++i ) {
            gExecFactories[ factories_[ i ].first ] = factories_[ i ].second;
         }
      }
   } initExecvFactories_;

} // unnamed namespace
