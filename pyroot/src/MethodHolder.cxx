// @(#)root/pyroot:$Name:  $:$Id: MethodHolder.cxx,v 1.28 2005/03/04 07:44:11 brun Exp $
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

// CINT
#include "Api.h"
#include "TVirtualMutex.h"

// Standard
#include <assert.h>
#include <exception>
#include <string>
#include <Riostream.h>


//- local helpers ------------------------------------------------------------
namespace {

// CINT temp level guard
   struct TempLevelGuard {
      TempLevelGuard() { G__settemplevel( 1 ); }
      ~TempLevelGuard() { G__settemplevel( -1 ); }
   };

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
      G__TypeInfo argType = arg->GetTypeName();

      std::string fullType = arg->GetFullTypeName();
      std::string realType = argType.TrueName();

      if ( Utility::isPointer( fullType ) ) {
         ConvFactories_t::iterator h = gConvFactories.find( realType + "*" );
         if ( h == gConvFactories.end() ) {
            if ( fullType.find( "const" ) != std::string::npos )
               h = gConvFactories.find( "const void*" );
            else
               h = gConvFactories.find( "void*" );
         }
         fConverters[ iarg ] = (h->second)();
      } else if ( argType.Property() & G__BIT_ISENUM ) {
         fConverters[ iarg ] = (gConvFactories.find( "UInt_t" )->second)();
      } else {
         ConvFactories_t::iterator h = gConvFactories.find( realType );
         if ( h != gConvFactories.end() ) {
            fConverters[ iarg ] = (h->second)();
         } else {
            PyErr_Format( PyExc_TypeError, "argument type %s not handled", fullType.c_str() );
            return false;
         }
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
// determine effective return type
   std::string longName = fMethod ? fMethod->GetReturnTypeName() : fClass->GetName();
   std::string shortName = TClassEdit::ShortType( G__TypeInfo( longName.c_str() ).TrueName(), 1 );

// select and set executor
   const char* q = "";
   if ( Utility::isPointer( longName ) == 1 )
      q = "*";

   ExecFactories_t::iterator h = gExecFactories.find( shortName + q );
   if ( h != gExecFactories.end() )
      executor = (h->second)();
   else {
      TClass* klass = gROOT->GetClass( shortName.c_str() );
      if ( klass != 0 )
         executor = new RootObjectExecutor( klass );
      else {
         std::cerr << "return type in method not handled! " << shortName << std::endl;
         executor = (gExecFactories[ "void" ])();
      }
   }

   return true;
}

//____________________________________________________________________________
inline void PyROOT::MethodHolder::CalcOffset_( void* obj, TClass* klass )
{
// actual offset calculation, as needed
   long derivedtagnum = klass->GetClassInfo()->Tagnum();

   if ( derivedtagnum != fTagnum ) {
      fOffset = G__isanybase( fClass->GetClassInfo()->Tagnum(), derivedtagnum, (long) obj );
      fTagnum = derivedtagnum;
   }
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
   return PyString_FromFormat( "%s%s %s::%s%s",
      ( fMethod->Property() & G__BIT_ISSTATIC ) ? "static " : "", fMethod->GetReturnTypeName(),
      fClass->GetName(), fMethod->GetName(), fMethod->GetSignature() );
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
   fMethodCall->SetFuncProto( fClass->GetClassInfo(),
      fMethod ? fMethod->GetName() : fClass->GetName(), callString.c_str(), &fOffset );

// minimum number of arguments when calling
   fArgsRequired = fMethod ? fMethod->GetNargs() - fMethod->GetNargsOpt() : 0;

// init done
   fIsInitialized = true;

   return true;
}


//____________________________________________________________________________
bool PyROOT::MethodHolder::FilterArgs( ObjectProxy*& self, PyObject*& args, PyObject*& )
{
// verify self
   if ( self != 0 )
      return true;

// otherwise, check for a suitable 'self' in args and update accordingly
   if ( PyTuple_GET_SIZE( args ) != 0 ) {
      ObjectProxy* pyobj = (ObjectProxy*)PyTuple_GET_ITEM( args, 0 );
      if ( ObjectProxy_Check( pyobj ) ) {
      // reset self
         self = pyobj;

      // offset args by 1
         args = PyTuple_GetSlice( args, 1, PyTuple_GET_SIZE( args ) );

      // declare success
         return true;
      }
   }

// no self, set error and lament
   PyErr_Format( PyExc_TypeError,
      "unbound method %s::%s must be called with a %s instance as first argument",
      fClass->GetName(), fMethod->GetName(), fClass->GetName() );
   return false;
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
      PyErr_Format( PyExc_TypeError, "%s() takes at least %d arguments (%d given)",
         fMethod ? fMethod->GetName() : fClass->GetName(), fArgsRequired, argc );
      return false;
   } else if ( argMax < argc ) {
      PyErr_Format( PyExc_TypeError, "%s() takes at most %d arguments (%d given)",
         fMethod ? fMethod->GetName() : fClass->GetName(), argMax, argc );
      return false;
   }

// convert the arguments to the method call array
   for ( int i = 0; i < argc; i++ ) {
      if ( ! fConverters[ i ]->SetArg( PyTuple_GET_ITEM( args, i ), fMethodCall ) ) {
         PyErr_Format( PyExc_TypeError, "could not convert argument %d", i );
         return false;
      }
   }

   return true;
}

//____________________________________________________________________________
PyObject* PyROOT::MethodHolder::Execute( void* self )
{
   R__LOCKGUARD( gCINTMutex );
   TempLevelGuard g;

   PyObject* result = 0;

   try {
      result = fExecutor->Execute( fMethodCall, (void*)((long)self + fOffset) );
   } catch ( TPyException& ) {
      result = TPyExceptionMagic;
   } catch ( std::exception& e ) {
      std::cout << "C++ exception caught: " << e.what() << std::endl;
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

// verify and put the arguments in usable order
   if ( ! FilterArgs( self, args, kwds ) )
      return 0;

// translate the arguments
   if ( ! SetMethodArgs( args ) )
      return 0;                              // important: 0, not Py_None

// get the ROOT object that this object proxy is a handle for
   void* object = self->GetObject();

// validity check that should not fail
   if ( ! object ) {
      PyErr_SetString( PyExc_ReferenceError, "attempt to access a null-pointer" );
      return 0;
   }

// reset this method's offset for the object as appropriate
   CalcOffset_( object, self->ObjectIsA() );

// actual call
   return Execute( self->GetObject() );
}
