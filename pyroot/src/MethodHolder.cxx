// @(#)root/pyroot:$Name:  $:$Id: MethodHolder.cxx,v 1.49 2006/07/01 21:19:55 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "MethodHolder.h"
#include "Converters.h"
#include "Executors.h"
#include "ObjectProxy.h"
#include "RootWrapper.h"
#include "TPyException.h"
#include "Utility.h"

// ROOT
#include "TROOT.h"
#include "TClass.h"
#include "TString.h"
#include "TFunction.h"
#include "TMethodArg.h"
#include "TClassEdit.h"
#include "TVirtualMutex.h"
#include "TException.h"

// CINT
#include "Api.h"
#include "TCint.h"

// Standard
#include <assert.h>
#include <string.h>
#include <exception>
#include <string>


//- data and local helpers ---------------------------------------------------
namespace {

// CINT temp level guard
   struct TempLevelGuard_t {
      TempLevelGuard_t() { G__settemplevel( 1 ); }
      ~TempLevelGuard_t() { G__settemplevel( -1 ); }
   };

   G__ClassInfo* GetGlobalNamespaceInfo() {
      static G__ClassInfo gcl;
      return &gcl;
   }

} // unnamed namespace


//- private helpers ----------------------------------------------------------
inline void PyROOT::TMethodHolder::Copy_( const TMethodHolder& other )
{
// yes, these pointer copy semantics are proper
   fClass  = other.fClass;
   fMethod = other.fMethod;

// do not copy caches
   fMethodCall = 0;
   fExecutor   = 0;

   fArgsRequired = -1;
   fOffset       =  0;

// being uninitialized will trigger setting up caches as appropriate
   fIsInitialized  = kFALSE;
}

//____________________________________________________________________________
inline void PyROOT::TMethodHolder::Destroy_() const
{
// no deletion of fMethod (ROOT responsibility)
   delete fMethodCall;

// destroy executor and argument converters
   delete fExecutor;

   for ( int i = 0; i < (int)fConverters.size(); ++i )
      delete fConverters[ i ];
}

//____________________________________________________________________________
inline PyObject* PyROOT::TMethodHolder::CallFast( void* self )
{
// helper code to prevent some duplication; this is called from CallSafe() as well
// as directly from TMethodHolder::Execute in fast mode

   PyObject* result = 0;

   try {       // C++ try block
      result = fExecutor->Execute( fMethodCall, (void*)((Long_t)self + fOffset) );
   } catch ( TPyException& ) {
      result = (PyObject*)TPyExceptionMagic;
   } catch ( std::exception& e ) {
      PyErr_Format( PyExc_Exception, "%s (C++ exception)", e.what() );
      result = 0;
   } catch ( ... ) {
      PyErr_SetString( PyExc_Exception, "unhandled, unknown C++ exception" );
      result = 0;
   }

   return result;
}

//____________________________________________________________________________
inline PyObject* PyROOT::TMethodHolder::CallSafe( void* self )
{
// helper code to prevent some code duplication; this code embeds a ROOT "try/catch"
// block that saves the stack for restoration in case of an otherwise fatal signal

   PyObject* result = 0;

   TRY {       // ROOT "try block"
      result = CallFast( self );
   } CATCH( excode ) {
      PyErr_SetString( PyExc_SystemError, "problem in C++; program state has been reset" );
      result = 0;
      Throw( excode );
   } ENDTRY;

   return result;
}

//____________________________________________________________________________
Bool_t PyROOT::TMethodHolder::InitCallFunc_( std::string& callString )
{
// buffers for argument dispatching
   const int nArgs = fMethod ? fMethod->GetNargs() : 0;
   if ( nArgs == 0 )
      return kTRUE;

   fConverters.resize( nArgs );    // id.

// setup the dispatch cache
   int iarg = 0;
   TIter nextarg( fMethod->GetListOfMethodArgs() );
   while ( TMethodArg* arg = (TMethodArg*)nextarg() ) {
      std::string fullType = arg->GetFullTypeName();
      fConverters[ iarg ] = CreateConverter( fullType );

      if ( ! fConverters[ iarg ] ) {
         PyErr_Format( PyExc_TypeError, "argument type %s not handled", fullType.c_str() );
         return kFALSE;
      }

   // setup call string
      if ( callString.length() == 0 )
         callString = fullType;
      else
         callString += "," + fullType;

   // advance argument counter
      iarg += 1;
   }

   return kTRUE;
}

//____________________________________________________________________________
Bool_t PyROOT::TMethodHolder::InitExecutor_( TExecutor*& executor )
{
// install executor conform to the return type
   executor = CreateExecutor( fMethod ? fMethod->GetReturnTypeName() : fClass->GetName() );
   if ( ! executor )
      return kFALSE;

   return kTRUE;
}

//____________________________________________________________________________
void PyROOT::TMethodHolder::SetPyError_( PyObject* msg )
{
// helper to report errors in a consistent format (derefs msg)
   PyObject *etype, *evalue, *etrace;
   PyErr_Fetch( &etype, &evalue, &etrace );

   std::string details = "";
   if ( evalue ) {
      PyObject* s = PyObject_Str( evalue );
      details = PyString_AS_STRING( s );
      Py_DECREF( s );
   }

   Py_XDECREF( etype ); Py_XDECREF( evalue ); Py_XDECREF( etrace );

   PyObject* doc = GetDocString();

   if ( details != "" ) {
      PyErr_Format( PyExc_TypeError, "%s =>\n    %s (%s)",
          PyString_AS_STRING( doc ), PyString_AS_STRING( msg ), details.c_str() );
   } else {
      PyErr_Format( PyExc_TypeError, "%s =>\n    %s",
          PyString_AS_STRING( doc ), PyString_AS_STRING( msg ) );
   }

   Py_DECREF( doc );
   Py_DECREF( msg );
}

//- constructors and destructor ----------------------------------------------
PyROOT::TMethodHolder::TMethodHolder( TClass* klass, TFunction* method ) :
      fClass( klass ), fMethod( method )
{
// constructor; initialization is deferred
   fMethodCall    =  0;
   fExecutor      =  0;
   fArgsRequired  = -1;
   fOffset        =  0;

   fIsInitialized = kFALSE;
}

//____________________________________________________________________________
PyROOT::TMethodHolder::TMethodHolder( const TMethodHolder& other ) : PyCallable( other )
{
// copy constructor
   Copy_( other );
}

//____________________________________________________________________________
PyROOT::TMethodHolder& PyROOT::TMethodHolder::operator=( const TMethodHolder& other )
{
// assignment operator
   if ( this != &other ) {
      Destroy_();
      Copy_( other );
   }

   return *this;
}

//____________________________________________________________________________
PyROOT::TMethodHolder::~TMethodHolder()
{
// destructor
   Destroy_();
}


//- public members -----------------------------------------------------------
PyObject* PyROOT::TMethodHolder::GetSignature()
{
// construct python string from the method's signature
   return PyString_FromFormat( "%s", fMethod->GetSignature() );
}

//____________________________________________________________________________
PyObject* PyROOT::TMethodHolder::GetPrototype()
{
// construct python string from the method's prototype
   return PyString_FromFormat( "%s%s",
      ( fMethod->Property() & G__BIT_ISSTATIC ) ? "static " : "", fMethod->GetPrototype() );
}

//____________________________________________________________________________
Int_t PyROOT::TMethodHolder::GetPriority()
{
// Method priorities exist (in lieu of true overloading) there to prevent
// void* or <unknown>* from usurping otherwise valid calls. TODO: extend this
// to favour classes that are not bases.

   Int_t priority = 0;

   TIter nextarg( fMethod->GetListOfMethodArgs() );
   while ( TMethodArg* arg = (TMethodArg*)nextarg() ) {
      G__TypeInfo ti( arg->GetFullTypeName() );

   // the following numbers are made up and may cause problems in specific
   // situations: use <obj>.<meth>.disp() for choice of exact dispatch
      if ( ! ti.IsValid() )
         priority -= 10000;   // class is gibberish
      else if ( (ti.Property() & (kIsClass|kIsStruct)) && ! ti.IsLoaded() )
         priority -= 1000;    // class is known, but no dictionary available
      else if ( TClassEdit::CleanType( ti.TrueName(), 1 ) == "void*" )
         priority -= 100;     // void* shouldn't be too greedy
      else if ( TClassEdit::CleanType( ti.TrueName(), 1 ) == "float" )
         priority -= 30;      // double preferred over float (no float in python)
      else if ( TClassEdit::CleanType( ti.TrueName(), 1 ) == "double" )
         priority -= 10;      // char, int, long preferred over double
   }

   return priority;
}

//____________________________________________________________________________
Bool_t PyROOT::TMethodHolder::Initialize()
{
// done if cache is already setup
   if ( fIsInitialized == kTRUE )
      return kTRUE;

   std::string callString = "";
   if ( ! InitCallFunc_( callString ) )
      return kFALSE;

   if ( ! InitExecutor_( fExecutor ) )
      return kFALSE;

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
   fIsInitialized = kTRUE;

   return kTRUE;
}

//____________________________________________________________________________
PyObject* PyROOT::TMethodHolder::FilterArgs( ObjectProxy*& self, PyObject* args, PyObject* )
{
// verify existence of self, return if ok
   if ( self != 0 ) {
      Py_INCREF( args );
      return args;
   }

// otherwise, check for a suitable 'self' in args and update accordingly
   if ( PyTuple_GET_SIZE( args ) != 0 ) {
      ObjectProxy* pyobj = (ObjectProxy*)PyTuple_GET_ITEM( args, 0 );

   // demand PyROOT object, and an argument that may match down the road
      if ( ObjectProxy_Check( pyobj ) &&
           ( strlen( fClass->GetName() ) == 0 ||            // free global
           ( pyobj->ObjectIsA() == 0 ) ||                   // null pointer or ctor call
           ( pyobj->ObjectIsA()->GetBaseClass( fClass ) ) ) // matching types
         ) {
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
Bool_t PyROOT::TMethodHolder::SetMethodArgs( PyObject* args )
{
// clean slate
   fMethodCall->ResetArg();

   int argc = PyTuple_GET_SIZE( args );
   int argMax = fConverters.size();

// argc must be between min and max number of arguments
   if ( argc < fArgsRequired ) {
      SetPyError_( PyString_FromFormat(
         "takes at least %d arguments (%d given)", fArgsRequired, argc ) );
      return kFALSE;
   } else if ( argMax < argc ) {
      SetPyError_( PyString_FromFormat(
         "takes at most %d arguments (%d given)", argMax, argc ) );
      return kFALSE;
   }

// convert the arguments to the method call array
   for ( int i = 0; i < argc; i++ ) {
      if ( ! fConverters[ i ]->SetArg( PyTuple_GET_ITEM( args, i ), fMethodCall ) ) {
         SetPyError_( PyString_FromFormat( "could not convert argument %d", i+1 ) );
         return kFALSE;
      }
   }

   return kTRUE;
}

//____________________________________________________________________________
PyObject* PyROOT::TMethodHolder::Execute( void* self )
{
// call the interface method
   R__LOCKGUARD2( gCINTMutex );
   TempLevelGuard_t g;

   PyObject* result = 0;

#ifndef R__WIN32              // G__return isn't for API use on Windows
// TODO: get this into an API to cleanup/init on fresh call
   G__return = G__RETURN_NON;
#endif

   if ( Utility::gSignalPolicy == Utility::kFast ) {
   // bypasses ROOT try block (i.e. segfaults will abort)
      result = CallFast( self );
   } else {
   // at the cost of ~10% performance, don't abort the interpreter on any signal
      result = CallSafe( self );
   }

   if ( result && PyErr_Occurred() ) {
   // can happen in the case of a CINT error: trigger exception processing
      Py_DECREF( result );
      result = 0;
   }

   return result;
}

//____________________________________________________________________________
PyObject* PyROOT::TMethodHolder::operator()( ObjectProxy* self, PyObject* args, PyObject* kwds )
{
// setup as necessary
   if ( ! Initialize() )
      return 0;                              // important: 0, not Py_None

// fetch self, verify, and put the arguments in usable order
   if ( ! ( args = FilterArgs( self, args, kwds ) ) )
      return 0;

// translate the arguments
   Bool_t bConvertOk = SetMethodArgs( args );
   Py_DECREF( args );

   if ( bConvertOk == kFALSE )
      return 0;                              // important: 0, not Py_None

// get the ROOT object that this object proxy is a handle for
   void* object = self->GetObject();

// validity check that should not fail
   if ( ! object ) {
      PyErr_SetString( PyExc_ReferenceError, "attempt to access a null-pointer" );
      return 0;
   }

// get its class
   TClass* klass = self->ObjectIsA();

// reset this method's offset for the object as appropriate
   int objTag  = klass->GetClassInfo()  ? klass->GetClassInfo()->Tagnum()  : -1;   // derived
   int methTag = fClass->GetClassInfo() ? fClass->GetClassInfo()->Tagnum() : -1;   // base
   fOffset = objTag == methTag ? 0 : G__isanybase( methTag, objTag, (Long_t)object );

// actual call; recycle self instead of new object for same address objects
   ObjectProxy* pyobj = (ObjectProxy*)Execute( object );
   if ( ObjectProxy_Check( pyobj ) &&
        pyobj->GetObject() == object && pyobj->ObjectIsA() == klass ) {
      Py_INCREF( (PyObject*)self );
      Py_DECREF( pyobj );
      return (PyObject*)self;
   }

   return (PyObject*)pyobj;
}
