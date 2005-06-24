// @(#)root/pyroot:$Name:  $:$Id: MethodHolder.cxx,v 1.37 2005/06/22 20:18:12 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "MethodHolder.h"
#include "Converters.h"
#include "Executors.h"
#include "ObjectProxy.h"
#include "RootWrapper.h"
#include "TPyException.h"

// ROOT
#include "TROOT.h"
#include "TClass.h"
#include "TString.h"
#include "TMethod.h"
#include "TMethodArg.h"
#include "TClassEdit.h"
#include "TVirtualMutex.h"

// CINT
#include "Api.h"
#include "TCint.h"

// Standard
#include <assert.h>
#include <string.h>
#include <exception>
#include <string>


//- local helpers ------------------------------------------------------------
namespace {

// CINT temp level guard
   struct TempLevelGuard {
      TempLevelGuard() { G__settemplevel( 1 ); }
      ~TempLevelGuard() { G__settemplevel( -1 ); }
   };

   TClassRef GetGlobalNamespace() {
      static TClass c;
      return &c;
   }

   G__ClassInfo* GetGlobalNamespaceInfo() {
      static G__ClassInfo gcl;
      return &gcl;
   }

} // unnamed namespace


//- private helpers ----------------------------------------------------------
inline void PyROOT::MethodHolder::Copy_( const MethodHolder& other )
{
// yes, these pointer copy semantics are proper
   fClass  = other.fClass;
   fMethod = other.fMethod;

// do not copy caches
   fMethodCall = 0;
   fExecutor   = 0;

   fArgsRequired = -1;
   fOffset       =  0;
   fTagnum       = -1;

// being uninitialized will trigger setting up caches as appropriate
   fIsInitialized  = false;
}

//____________________________________________________________________________
inline void PyROOT::MethodHolder::Destroy_() const
{
// no deletion of fMethod (ROOT responsibility)
   delete fMethodCall;

// destroy executor and argument converters
   delete fExecutor;

   for ( int i = 0; i < (int)fConverters.size(); ++i )
      delete fConverters[ i ];
}

//____________________________________________________________________________
bool PyROOT::MethodHolder::InitCallFunc_( std::string& callString )
{
// buffers for argument dispatching
   const int nArgs = fMethod ? fMethod->GetNargs() : 0;
   if ( nArgs == 0 )
      return true;

   fConverters.resize( nArgs );    // id.

// setup the dispatch cache
   int iarg = 0;
   TIter nextarg( fMethod->GetListOfMethodArgs() );
   while ( TMethodArg* arg = (TMethodArg*)nextarg() ) {
      std::string fullType = arg->GetFullTypeName();
      fConverters[ iarg ] = CreateConverter( fullType );

      if ( ! fConverters[ iarg ] ) {
         PyErr_Format( PyExc_TypeError, "argument type %s not handled", fullType.c_str() );
         return false;
      }

   // setup call string
      if ( callString.length() == 0 )
         callString = fullType;
      else
         callString += "," + fullType;

   // advance argument counter
      iarg += 1;
   }

   return true;
}

//____________________________________________________________________________
bool PyROOT::MethodHolder::InitExecutor_( Executor*& executor )
{
   executor = CreateExecutor( fMethod ? fMethod->GetReturnTypeName() : fClass->GetName() );
   if ( ! executor )
      return false;

   return true;
}

//____________________________________________________________________________
inline void PyROOT::MethodHolder::CalcOffset_( void* obj, TClass* klass )
{
// actual offset calculation, as needed
   long derivedtagnum = klass->GetClassInfo() ? klass->GetClassInfo()->Tagnum() : -1;

   if ( derivedtagnum != fTagnum ) {
      fOffset = G__isanybase(
         fClass->GetClassInfo() ? fClass->GetClassInfo()->Tagnum() : -1,
         derivedtagnum, (long)obj );
      fTagnum = derivedtagnum;
   }
}

//____________________________________________________________________________
inline void PyROOT::MethodHolder::SetPyError_( PyObject* msg )
{
// helper to report errors in a consistent format (derefs msg)
   PyObject* doc = GetDocString();
   PyErr_Format( PyExc_TypeError,
      "%s =>\n    %s", PyString_AS_STRING( doc ), PyString_AS_STRING( msg ) );

   Py_DECREF( doc );
   Py_DECREF( msg );
}

//- constructors and destructor ----------------------------------------------
PyROOT::MethodHolder::MethodHolder( TClass* klass, TMethod* method ) :
      fClass( klass ), fMethod( method )
{
   fMethodCall    =  0;
   fExecutor      =  0;
   fArgsRequired  = -1;
   fOffset        =  0;
   fTagnum        = -1;

   fIsInitialized = false;
}

//____________________________________________________________________________
PyROOT::MethodHolder::MethodHolder( TFunction* function ) :
      fClass( GetGlobalNamespace() ), fMethod( function )
{
   fMethodCall    =  0;
   fExecutor      =  0;
   fArgsRequired  = -1;
   fOffset        =  0;
   fTagnum        = -1;

   fIsInitialized = false;
}

//____________________________________________________________________________
PyROOT::MethodHolder::MethodHolder( const MethodHolder& other ) : PyCallable( other )
{
   Copy_( other );
}

//____________________________________________________________________________
PyROOT::MethodHolder& PyROOT::MethodHolder::operator=( const MethodHolder& other )
{
   if ( this != &other ) {
      Destroy_();
      Copy_( other );
   }

   return *this;
}

//____________________________________________________________________________
PyROOT::MethodHolder::~MethodHolder()
{
   Destroy_();
}


//- public members -----------------------------------------------------------
PyObject* PyROOT::MethodHolder::GetDocString()
{
   return PyString_FromFormat( "%s%s",
      ( fMethod->Property() & G__BIT_ISSTATIC ) ? "static " : "", fMethod->GetPrototype() );
}

//____________________________________________________________________________
bool PyROOT::MethodHolder::Initialize()
{
// done if cache is already setup
   if ( fIsInitialized == true )
      return true;

   std::string callString = "";
   if ( ! InitCallFunc_( callString ) )
      return false;

   if ( ! InitExecutor_( fExecutor ) )
      return false;

// setup call func
   assert( fMethodCall == 0 );

   fMethodCall = new G__CallFunc();
   fMethodCall->Init();

   G__ClassInfo* gcl = fClass->GetClassInfo();
   if ( ! gcl )
      gcl = GetGlobalNamespaceInfo();
   
   fMethodCall->SetFunc( gcl->GetMethod(
      fMethod ? fMethod->GetName() : fClass->GetName(), callString.c_str(), &fOffset ) );

// minimum number of arguments when calling
   fArgsRequired = fMethod ? fMethod->GetNargs() - fMethod->GetNargsOpt() : 0;

// init done
   fIsInitialized = true;

   return true;
}

//____________________________________________________________________________
PyObject* PyROOT::MethodHolder::FilterArgs( ObjectProxy*& self, PyObject* args, PyObject* )
{
// verify existence of self, return if ok
   if ( self != 0 ) {
      Py_INCREF( args );
      return args;
   }

// otherwise, check for a suitable 'self' in args and update accordingly
   if ( PyTuple_GET_SIZE( args ) != 0 ) {
      ObjectProxy* pyobj = (ObjectProxy*)PyTuple_GET_ITEM( args, 0 );

   // demand PyROOT object, and either free global or matching class instance
      if ( ObjectProxy_Check( pyobj ) && ( strlen( fClass->GetName() ) == 0 ||
           ( pyobj->ObjectIsA() && pyobj->ObjectIsA()->GetBaseClass( fClass ) ) ) ) {
      // reset self (will live for the life time of args; i.e. call of function)
         self = pyobj;

      // offset args by 1 (new ref)
         return PyTuple_GetSlice( args, 1, PyTuple_GET_SIZE( args ) );
      }
   }

// no self, set error and lament
   SetPyError_( PyString_FromFormat(
      "unbound method %s::%s must be called with a %s instance as first argument",
      fClass->GetName(), fMethod->GetName(), fClass->GetName() ) );
   return 0;
}

//____________________________________________________________________________
bool PyROOT::MethodHolder::SetMethodArgs( PyObject* args )
{
// clean slate
   fMethodCall->ResetArg();

   int argc = PyTuple_GET_SIZE( args );
   int argMax = fConverters.size();

// argc must be between min and max number of arguments
   if ( argc < fArgsRequired ) {
      SetPyError_( PyString_FromFormat(
         "takes at least %d arguments (%d given)", fArgsRequired, argc ) );
      return false;
   } else if ( argMax < argc ) {
      SetPyError_( PyString_FromFormat(
         "takes at most %d arguments (%d given)", argMax, argc ) );
      return false;
   }

// convert the arguments to the method call array
   for ( int i = 0; i < argc; i++ ) {
      if ( ! fConverters[ i ]->SetArg( PyTuple_GET_ITEM( args, i ), fMethodCall ) ) {
         SetPyError_( PyString_FromFormat( "could not convert argument %d", i+1 ) );
         return false;
      }
   }

   return true;
}

//____________________________________________________________________________
PyObject* PyROOT::MethodHolder::Execute( void* self )
{
   R__LOCKGUARD2( gCINTMutex );
   TempLevelGuard g;

   PyObject* result = 0;

   try {
      result = fExecutor->Execute( fMethodCall, (void*)((long)self + fOffset) );
   } catch ( TPyException& ) {
      result = TPyExceptionMagic;
   } catch ( std::exception& e ) {
      PyErr_Format( PyExc_Exception, "%s (C++ exception)", e.what() );
      result = 0;
   }

   if ( result && PyErr_Occurred() ) {
   // can happen in the case of a CINT error: trigger exception processing
      Py_DECREF( result );
      result = 0;
   }

   return result;
}

//____________________________________________________________________________
PyObject* PyROOT::MethodHolder::operator()( ObjectProxy* self, PyObject* args, PyObject* kwds )
{
// setup as necessary
   if ( ! Initialize() )
      return 0;                              // important: 0, not Py_None

// fetch self, verify, and put the arguments in usable order
   if ( ! ( args = FilterArgs( self, args, kwds ) ) )
      return 0;

// translate the arguments
   bool bConvertOk = SetMethodArgs( args );
   Py_DECREF( args );

   if ( bConvertOk == false )
      return 0;

// get the ROOT object that this object proxy is a handle for
   void* object = self->GetObject();

// validity check that should not fail
   if ( ! object ) {
      PyErr_SetString( PyExc_ReferenceError, "attempt to access a null-pointer" );
      return 0;
   }

// reset this method's offset for the object as appropriate
   CalcOffset_( object, self->ObjectIsA() );

// actual call; recycle self instead of new object for same address objects
   PyObject* pyobject = Execute( self->GetObject() );
   if ( ObjectProxy_Check( pyobject ) &&
        ((ObjectProxy*)pyobject)->GetObject() == self->GetObject() &&
        ((ObjectProxy*)pyobject)->ObjectIsA() == self->ObjectIsA() ) {
      Py_INCREF( (PyObject*)self );
      Py_DECREF( pyobject );
      return (PyObject*)self;
   }

   return pyobject;
}
