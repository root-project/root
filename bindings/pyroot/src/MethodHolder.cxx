// @(#)root/pyroot:$Id$
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
#include "Adapters.h"

// ROOT
#include "TROOT.h"
#include "TClass.h"
#include "TString.h"
#include "TClassEdit.h"
#include "TVirtualMutex.h"
#include "TException.h"

// CINT
#include "Api.h"
#include "TInterpreter.h"

// Standard
#include <assert.h>
#include <string.h>
#include <exception>
#include <string>


//- data and local helpers ---------------------------------------------------
R__EXTERN PyObject* gRootModule;

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
template< class T, class M >
inline void PyROOT::TMethodHolder< T, M >::Copy_( const TMethodHolder& other )
{
// do not copy caches
   fMethodCall = 0;
   fExecutor   = 0;

   fArgsRequired = -1;
   fOffset       =  0;

   fSignature = other.fSignature;

// being uninitialized will trigger setting up caches as appropriate
   fIsInitialized  = kFALSE;
}

//____________________________________________________________________________
template< class T, class M >
inline void PyROOT::TMethodHolder< T, M >::Destroy_() const
{
// no deletion of fMethod (ROOT responsibility)
   delete fMethodCall;

// destroy executor and argument converters
   delete fExecutor;

   for ( int i = 0; i < (int)fConverters.size(); ++i )
      delete fConverters[ i ];
}

//____________________________________________________________________________
template< class T, class M >
inline PyObject* PyROOT::TMethodHolder< T, M >::CallFast( void* self )
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
template< class T, class M >
inline PyObject* PyROOT::TMethodHolder< T, M >::CallSafe( void* self )
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
namespace PyROOT {

#ifdef PYROOT_USE_REFLEX
template<>
Bool_t TMethodHolder< ROOT::Reflex::Scope, ROOT::Reflex::Member >::InitCallFunc_()
{
// build buffers for argument dispatching
   const size_t nArgs = fMethod.FunctionParameterSize();
   fConverters.resize( nArgs );
   fParameters.resize( nArgs );
   fParamPtrs.resize( nArgs );

// setup the dispatch cache
   for ( size_t iarg = 0; iarg < nArgs; ++iarg ) {
      std::string fullType =
         fMethod.TypeOf().FunctionParameterAt( iarg ).Name( ROOT::Reflex::QUALIFIED | ROOT::Reflex::SCOPED );
      fConverters[ iarg ] = CreateConverter( fullType );

      if ( ! fConverters[ iarg ] ) {
         PyErr_Format( PyExc_TypeError, "argument type %s not handled", fullType.c_str() );
         return kFALSE;
      }

   }

   return kTRUE;
}
#endif

} // namespace PyROOT

template< class T, class M >
Bool_t PyROOT::TMethodHolder< T, M >::InitCallFunc_()
{
// build buffers for argument dispatching
   const size_t nArgs = fMethod.FunctionParameterSize();
   fConverters.resize( nArgs );
   fParameters.resize( nArgs );
   fParamPtrs.resize( nArgs );

// setup the dispatch cache
   std::string callString = "";
   for ( size_t iarg = 0; iarg < nArgs; ++iarg ) {
      std::string fullType =
         fMethod.TypeOf().FunctionParameterAt( iarg ).Name( ROOT::Reflex::QUALIFIED | ROOT::Reflex::SCOPED );
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
   }

// setup call func
   assert( fMethodCall == 0 );

   G__ClassInfo* gcl = (G__ClassInfo*)((TClass*)fClass.Id())->GetClassInfo();
   if ( ! gcl )
      gcl = GetGlobalNamespaceInfo();

   G__MethodInfo gmi = gcl->GetMethod(
      (Bool_t)fMethod == true ? fMethod.Name().c_str() : fClass.Name().c_str(), callString.c_str(),
      &fOffset, G__ClassInfo::ExactMatch );

   if ( ! gmi.IsValid() && (Bool_t)fMethod == true ) {
      PyErr_Format( PyExc_RuntimeError, "could not resolve %s::%s(%s)",
         fClass.Name().c_str(), fMethod.Name().c_str(), callString.c_str() );
      return kFALSE;
   }

   fMethodCall = new G__CallFunc();
   fMethodCall->Init();
   fMethodCall->SetFunc( gmi );

   return kTRUE;
}

//____________________________________________________________________________
template< class T, class M >
Bool_t PyROOT::TMethodHolder< T, M >::InitExecutor_( TExecutor*& executor )
{
// install executor conform to the return type
   executor = CreateExecutor( (Bool_t)fMethod == true ?
      fMethod.TypeOf().ReturnType().Name( ROOT::Reflex::Q | ROOT::Reflex::S | ROOT::Reflex::F )
      : fClass.Name( ROOT::Reflex::S | ROOT::Reflex::F ) );
   if ( ! executor )
      return kFALSE;

   return kTRUE;
}

//____________________________________________________________________________
template< class T, class M >
void PyROOT::TMethodHolder< T, M >::CreateSignature_()
{
// built a signature a la TFunction::GetSignature as python string, using Adapters
   Int_t ifirst = 0;
   fSignature = "(";
   const size_t nArgs = fMethod.FunctionParameterSize();
   for ( size_t iarg = 0; iarg < nArgs; ++iarg ) {
      if ( ifirst ) fSignature += ", ";

      fSignature += fMethod.TypeOf().FunctionParameterAt( iarg ).Name( ROOT::Reflex::QUALIFIED );

      const std::string& parname = fMethod.FunctionParameterNameAt( iarg );
      if ( ! parname.empty() ) {
         fSignature += " ";
         fSignature += parname;
      }

      const std::string& defvalue = fMethod.FunctionParameterDefaultAt( iarg );
      if ( ! defvalue.empty() ) {
         fSignature += " = ";
         fSignature += defvalue;
      }
      ifirst++;
   }
   fSignature += ")";
}

//____________________________________________________________________________
template< class T, class M >
const std::string& PyROOT::TMethodHolder< T, M >::GetSignatureString()
{
// construct python string from the method's signature
   if ( fSignature.empty() )
      CreateSignature_();

   return fSignature;
}

//____________________________________________________________________________
template< class T, class M >
void PyROOT::TMethodHolder< T, M >::SetPyError_( PyObject* msg )
{
// helper to report errors in a consistent format (derefs msg)
   PyObject *etype, *evalue, *etrace;
   PyErr_Fetch( &etype, &evalue, &etrace );

   std::string details = "";
   if ( evalue ) {
      PyObject* s = PyObject_Str( evalue );
      details = PyROOT_PyUnicode_AsString( s );
      Py_DECREF( s );
   }

   Py_XDECREF( etype ); Py_XDECREF( evalue ); Py_XDECREF( etrace );

   PyObject* doc = GetDocString();

   if ( details != "" ) {
      PyErr_Format( PyExc_TypeError, "%s =>\n    %s (%s)",
          PyROOT_PyUnicode_AsString( doc ), PyROOT_PyUnicode_AsString( msg ), details.c_str() );
   } else {
      PyErr_Format( PyExc_TypeError, "%s =>\n    %s",
          PyROOT_PyUnicode_AsString( doc ), PyROOT_PyUnicode_AsString( msg ) );
   }

   Py_DECREF( doc );
   Py_DECREF( msg );
}

//- constructors and destructor ----------------------------------------------
template< class T, class M >
PyROOT::TMethodHolder< T, M >::TMethodHolder( const T& klass, const M& method ) :
      fMethod( method ), fClass( klass )
{
// constructor; initialization is deferred
   fMethodCall    =  0;
   fExecutor      =  0;
   fArgsRequired  = -1;
   fOffset        =  0;

   fIsInitialized = kFALSE;
}

//____________________________________________________________________________
template< class T, class M >
PyROOT::TMethodHolder< T, M >::TMethodHolder( const TMethodHolder< T, M >& other ) :
       PyCallable( other ), fMethod( other.fMethod ), fClass( other.fClass )
{
// copy constructor
   Copy_( other );
}

//____________________________________________________________________________
template< class T, class M >
PyROOT::TMethodHolder< T, M >& PyROOT::TMethodHolder< T, M >::operator=(
      const TMethodHolder< T, M >& other )
{
// assignment operator
   if ( this != &other ) {
      Destroy_();
      Copy_( other );
      fClass  = other.fClass;
      fMethod = other.fMethod;
   }

   return *this;
}

//____________________________________________________________________________
template< class T, class M >
PyROOT::TMethodHolder< T, M >::~TMethodHolder()
{
// destructor
   Destroy_();
}


//- public members -----------------------------------------------------------
template< class T, class M >
PyObject* PyROOT::TMethodHolder< T, M >::GetSignature()
{
// construct python string from the method's signature
   return PyROOT_PyUnicode_FromString( GetSignatureString().c_str() );
}

//____________________________________________________________________________
template< class T, class M >
PyObject* PyROOT::TMethodHolder< T, M >::GetPrototype()
{
// construct python string from the method's prototype
   return PyROOT_PyUnicode_FromFormat( "%s%s %s::%s%s",
      ( fMethod.IsStatic() ? "static " : "" ),
      fMethod.TypeOf().ReturnType().Name( ROOT::Reflex::Q | ROOT::Reflex::S ).c_str(),
      fMethod.DeclaringScope().Name().c_str(), fMethod.Name().c_str(),
      GetSignatureString().c_str() );
}

//____________________________________________________________________________
namespace PyROOT {

#ifdef PYROOT_USE_REFLEX
template<>
Int_t TMethodHolder< ROOT::Reflex::Scope, ROOT::Reflex::Member >::GetPriority();
#endif

} // namespace PyROOT

template< class T, class M >
Int_t PyROOT::TMethodHolder< T, M >::GetPriority()
{
// Method priorities exist (in lieu of true overloading) there to prevent
// void* or <unknown>* from usurping otherwise valid calls. TODO: extend this
// to favour classes that are not bases.

   Int_t priority = 0;

   const size_t nArgs = fMethod.FunctionParameterSize();
   for ( size_t iarg = 0; iarg < nArgs; ++iarg ) {
      const T& arg = fMethod.TypeOf().FunctionParameterAt( iarg );

   // the following numbers are made up and may cause problems in specific
   // situations: use <obj>.<meth>.disp() for choice of exact dispatch
      if ( ! (Bool_t)arg ) {
         priority -= 10000;   // class is gibberish
      } else if ( (arg.IsClass() || arg.IsStruct()) && ! arg.IsComplete() ) {
      // class is known, but no dictionary available, 2 more cases: * and &
         const std::string aname = arg.Name( ROOT::Reflex::Q );
         if ( aname[ aname.size() - 1 ] == '&' )
            priority -= 3000;
         else
            priority -= 1000; // prefer pointer passing over reference
      } else {
         const std::string aname = arg.Name( ROOT::Reflex::F | ROOT::Reflex::Q );
         if ( aname == "void*" )
            priority -= 100;  // void* shouldn't be too greedy
         else if ( aname == "float" )
            priority -= 30;   // double preferred over float (no float in python)
         else if ( aname == "double" )
            priority -= 10;   // char, int, long preferred over double

      // resolve a few special cases
         else if ( aname == "IBaseFunctionMultiDim")
            priority -= 1;
         else if ( aname == "RooAbsReal" )
            priority -= 1;
      }

   }

   return priority;
}

namespace PyROOT {

#ifdef PYROOT_USE_REFLEX
template<>
Int_t TMethodHolder< ROOT::Reflex::Scope, ROOT::Reflex::Member >::GetPriority()
{
// Scope, Type, what's in a name ...
   return ((TMethodHolder< ROOT::Reflex::Type, ROOT::Reflex::Member >*)this)->
      TMethodHolder< ROOT::Reflex::Type, ROOT::Reflex::Member >::GetPriority();
}
#endif

} // endif

//____________________________________________________________________________
template< class T, class M >
Int_t PyROOT::TMethodHolder< T, M >::GetMaxArgs()
{
   return fMethod.FunctionParameterSize();
}

//____________________________________________________________________________
template< class T, class M>
PyObject* PyROOT::TMethodHolder< T, M >::GetArgSpec( Int_t iarg )
{
// Build a string representation of the arguments list.
   if ( iarg >= (int)fMethod.FunctionParameterSize() )
      return 0;

   std::string argrep = fMethod.TypeOf().FunctionParameterAt( iarg ).Name( ROOT::Reflex::Q );

   const std::string& parname = fMethod.FunctionParameterNameAt( iarg );
   if ( ! parname.empty() ) {
      argrep += " ";
      argrep += parname;
   }

   return PyROOT_PyUnicode_FromString( argrep.c_str() );
}

//____________________________________________________________________________
template< class T, class M>
PyObject* PyROOT::TMethodHolder< T, M >::GetArgDefault( Int_t iarg )
{
// get the default value (if any) of argument iarg of this method
   if ( iarg >= (int)fMethod.FunctionParameterSize() )
      return 0;

   const std::string& defvalue = fMethod.FunctionParameterDefaultAt( iarg ).c_str();
   if ( ! defvalue.empty() ) {

   // attempt to evaluate the string representation (will work for all builtin types)
      PyObject* pyval = (PyObject*)PyRun_String(
          (char*)defvalue.c_str(), Py_eval_input, gRootModule, gRootModule );
      if ( ! pyval && PyErr_Occurred() ) {
         PyErr_Clear();
         return PyROOT_PyUnicode_FromString( defvalue.c_str() );
      }

      return pyval;
   }

   return 0;
}

//____________________________________________________________________________
template< class T, class M>
PyObject* PyROOT::TMethodHolder< T, M >::GetScope()
{
// Get or build the scope of this method.
   return MakeRootClassFromString< TScopeAdapter, TBaseAdapter, TMemberAdapter >(
      fMethod.DeclaringScope().Name( ROOT::Reflex::SCOPED | ROOT::Reflex::FINAL ) );
}

//____________________________________________________________________________
template< class T, class M >
Bool_t PyROOT::TMethodHolder< T, M >::Initialize()
{
// done if cache is already setup
   if ( fIsInitialized == kTRUE )
      return kTRUE;

   if ( ! InitCallFunc_() )
      return kFALSE;

   if ( ! InitExecutor_( fExecutor ) )
      return kFALSE;

// minimum number of arguments when calling
   fArgsRequired = (Bool_t)fMethod == true ? fMethod.FunctionParameterSize( true ) : 0;

// init done
   fIsInitialized = kTRUE;

   return kTRUE;
}

//____________________________________________________________________________
template< class T, class M >
PyObject* PyROOT::TMethodHolder< T, M >::FilterArgs( ObjectProxy*& self, PyObject* args, PyObject* )
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
           ( fClass.Name().size() == 0 ||                   // free global
           ( pyobj->ObjectIsA() == 0 ) ||                   // null pointer or ctor call
           ( pyobj->ObjectIsA()->GetBaseClass( (TClass*)fClass.Id() ) ) ) // matching types
         ) {
      // reset self (will live for the life time of args; i.e. call of function)
         self = pyobj;

      // offset args by 1 (new ref)
         return PyTuple_GetSlice( args, 1, PyTuple_GET_SIZE( args ) );
      }
   }

// no self, set error and lament
   SetPyError_( PyROOT_PyUnicode_FromFormat(
      "unbound method %s::%s must be called with a %s instance as first argument",
      fClass.Name().c_str(), fMethod.Name().c_str(), fClass.Name().c_str() ) );
   return 0;
}

//____________________________________________________________________________
template< class T, class M >
Bool_t PyROOT::TMethodHolder< T, M >::SetMethodArgs( PyObject* args, Long_t user )
{
// clean slate
   if ( fMethodCall )
      fMethodCall->ResetArg();

   int argc = PyTuple_GET_SIZE( args );
   int argMax = fConverters.size();

// argc must be between min and max number of arguments
   if ( argc < fArgsRequired ) {
      SetPyError_( PyROOT_PyUnicode_FromFormat(
         "takes at least %d arguments (%d given)", fArgsRequired, argc ) );
      return kFALSE;
   } else if ( argMax < argc ) {
      SetPyError_( PyROOT_PyUnicode_FromFormat(
         "takes at most %d arguments (%d given)", argMax, argc ) );
      return kFALSE;
   }

// convert the arguments to the method call array
   for ( int i = 0; i < argc; ++i ) {
      if ( ! fConverters[ i ]->SetArg(
              PyTuple_GET_ITEM( args, i ), fParameters[i], fMethodCall, user ) ) {
         SetPyError_( PyROOT_PyUnicode_FromFormat( "could not convert argument %d", i+1 ) );
         return kFALSE;
      }
      fParamPtrs[i] = &fParameters[i];
   }

   return kTRUE;
}

//____________________________________________________________________________
namespace PyROOT {

#ifdef PYROOT_USE_REFLEX
template<>
PyObject* TMethodHolder< ROOT::Reflex::Scope, ROOT::Reflex::Member >::Execute( void* self )
{
// my first Reflex call ... this is all pretty wrong, but hey, it's a proto-prototype :)

   if ( fMethod.IsConstructor() )
      return (PyObject*)((ROOT::Reflex::Type)fClass).Construct( fMethod.TypeOf(), fParamPtrs ).Address();

   Reflex::Object obj( fClass, (void*)((Long_t)self + fOffset) );
   Reflex::Object result;
   static Reflex::Type tVoid = Reflex::Type::ByName("void");
   bool returnsVoid = (fMethod.TypeOf().ReturnType() == tVoid);
   if ( !returnsVoid ) {
      result = fMethod.TypeOf().ReturnType().Construct();
   }
   fMethod.Invoke( obj, &result, fParamPtrs );
   if ( !returnsVoid ) {
      TConverter* converter = CreateConverter( fMethod.TypeOf().ReturnType().Name(Reflex::S|Reflex::Q|Reflex::F) );
      if ( converter ) {
         PyObject* pyresult = converter->FromMemory( result.Address() );
         delete converter;
         return pyresult;
      }
   }

   Py_INCREF( Py_None );
   return Py_None;
}
#endif

} // namespace PyROOT

template< class T, class M >
PyObject* PyROOT::TMethodHolder< T, M >::Execute( void* self )
{
// call the interface method
   R__LOCKGUARD2( gCINTMutex );
   TempLevelGuard_t g;

   PyObject* result = 0;

   if ( Utility::gSignalPolicy == Utility::kFast ) {
   // bypasses ROOT try block (i.e. segfaults will abort)
      result = CallFast( self );
   } else {
   // at the cost of ~10% performance, don't abort the interpreter on any signal
      result = CallSafe( self );
   }

   if ( result && result != (PyObject*)TPyExceptionMagic
           && Utility::PyErr_Occurred_WithGIL() ) {
   // can happen in the case of a CINT error: trigger exception processing
      Py_DECREF( result );
      result = 0;
   }

// recover from error, if not cleared (needed as we're returning to prompt)
   if ( G__get_return( 0 ) > G__RETURN_NORMAL )
      G__security_recover( 0 );    // 0 ensures silence

   return result;
}

//____________________________________________________________________________
template< class T, class M >
PyObject* PyROOT::TMethodHolder< T, M >::operator()(
      ObjectProxy* self, PyObject* args, PyObject* kwds, Long_t user )
{
// preliminary check in case keywords are accidently used (they are ignored otherwise)
   if ( kwds != 0 && PyDict_Size( kwds ) ) {
      PyErr_SetString( PyExc_TypeError, "keyword arguments are not yet supported" );
      return 0;
   }

// setup as necessary
   if ( ! Initialize() )
      return 0;                              // important: 0, not Py_None

// fetch self, verify, and put the arguments in usable order
   if ( ! ( args = FilterArgs( self, args, kwds ) ) )
      return 0;

// translate the arguments
   Bool_t bConvertOk = SetMethodArgs( args, user );
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
   if ( klass ) {             // TODO: check should not be needed; only for Reflex, which'll fail
   // reset this method's offset for the object as appropriate
      int objTag  = (G__ClassInfo*)klass->GetClassInfo()  ? ((G__ClassInfo*)klass->GetClassInfo())->Tagnum()  : -1;   // derived
      G__ClassInfo* cli = (G__ClassInfo*)((TClass*)fClass.Id())->GetClassInfo();
      int methTag = cli ? cli->Tagnum() : -1;                                         // base
      fOffset = objTag == methTag ? 0 : G__isanybase( methTag, objTag, (Long_t)object );
   }

// actual call; recycle self instead of returning new object for same address objects
   ObjectProxy* pyobj = (ObjectProxy*)Execute( object );
   if ( pyobj != (ObjectProxy*)TPyExceptionMagic &&
        ObjectProxy_Check( pyobj ) &&
        pyobj->GetObject() == object &&
        klass && pyobj->ObjectIsA() == klass ) {
      Py_INCREF( (PyObject*)self );
      Py_DECREF( pyobj );
      return (PyObject*)self;
   }

   return (PyObject*)pyobj;
}

//____________________________________________________________________________
template class PyROOT::TMethodHolder< PyROOT::TScopeAdapter, PyROOT::TMemberAdapter >;
#ifdef PYROOT_USE_REFLEX
template class PyROOT::TMethodHolder< ROOT::Reflex::Scope, ROOT::Reflex::Member >;
#endif
