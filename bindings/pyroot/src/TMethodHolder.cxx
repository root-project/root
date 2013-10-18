// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "TMethodHolder.h"
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
#include "TInterpreter.h"

// Standard
#include <assert.h>
#include <string.h>
#include <exception>
#include <string>


//- data and local helpers ---------------------------------------------------
R__EXTERN PyObject* gRootModule;

namespace {

   ClassInfo_t* GetGlobalNamespaceInfo() {
      return gInterpreter->ClassInfo_Factory();
   }

} // unnamed namespace


//- private helpers ----------------------------------------------------------
inline void PyROOT::TMethodHolder::Copy_( const TMethodHolder& other )
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
inline void PyROOT::TMethodHolder::Destroy_() const
{
// no deletion of fMethod (ROOT responsibility)
   gInterpreter->CallFunc_Delete( fMethodCall );

// destroy executor and argument converters
   delete fExecutor;

   for ( int i = 0; i < (int)fConverters.size(); ++i )
      delete fConverters[ i ];
}

//____________________________________________________________________________
inline PyObject* PyROOT::TMethodHolder::CallFast( void* self, Bool_t release_gil )
{
// Helper code to prevent some duplication; this is called from CallSafe() as well
// as directly from TMethodHolder::Execute in fast mode.

   PyObject* result = 0;

// CLING WORKAROUND -- this happens for various cases, so allow them to fail
// here (return 0) rather than blocked from being called at all
   if ( ! gInterpreter->CallFunc_IsValid( fMethodCall ) )
      return 0;
// -- CLING WORKAROUND

   try {       // C++ try block
      result = fExecutor->Execute( fMethodCall, (void*)((Long_t)self + fOffset), release_gil );
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
inline PyObject* PyROOT::TMethodHolder::CallSafe( void* self, Bool_t release_gil )
{
// Helper code to prevent some code duplication; this code embeds a ROOT "try/catch"
// block that saves the stack for restoration in case of an otherwise fatal signal.

   PyObject* result = 0;

   TRY {       // ROOT "try block"
      result = CallFast( self, release_gil );
   } CATCH( excode ) {
      PyErr_SetString( PyExc_SystemError, "problem in C++; program state has been reset" );
      result = 0;
      Throw( excode );
   } ENDTRY;

   return result;
}

//____________________________________________________________________________
Bool_t PyROOT::TMethodHolder::InitCallFunc_()
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
         fMethod.TypeOf().FunctionParameterAt( iarg ).Name( Rflx::QUALIFIED | Rflx::SCOPED );
   // CLING WORKAROUND -- 
      fullType = Utility::ResolveTypedef( fullType, (TClass*)fClass.Id() );
   // -- END CLING WORKAROUND

   // CLING WORKAROUND -- std::string can not use kExactMatch as that will
   //                     fail, but if no exact match is used, the const-ref
   //                     std::string arguments will mask the const char* ones,
   //                     even though the extra default arguments differ
      if ( fClass.Name() == "string" && fMethod.Name() == "string" && fullType == "const std::string &" ) {
         fConverters[ iarg ] = new TStrictRootObjectConverter( "string", kFALSE ); // TODO: this is sooo wrong
   // -- CLING WORKAROUND
      } else
         fConverters[ iarg ] = CreateConverter( fullType );

      if ( ! fConverters[ iarg ] ) {
         PyErr_Format( PyExc_TypeError, "argument type %s not handled", fullType.c_str() );
         return kFALSE;
      }

   // setup call string
      if ( callString.length() == 0 )
         callString = fullType;
      else
         callString += ", " + fullType;
   }

// setup call func
   assert( fMethodCall == 0 );

   ClassInfo_t* gcl = ((TClass*)fClass.Id())->GetClassInfo();
   if ( ! gcl )
      gcl = GetGlobalNamespaceInfo();

   fMethodCall = gInterpreter->CallFunc_Factory();
   gInterpreter->CallFunc_SetFuncProto(
      fMethodCall,
      gcl,
      (Bool_t)fMethod == true ? fMethod.Name().c_str() : fClass.Name().c_str(),
      callString.c_str(),
      fMethod.IsConstant(),
      &fOffset,
      ROOT::kExactMatch );

// CLING WORKAROUND -- ROOT/meta thinks string is string, but gInterpreter disagrees
   if ( ! gInterpreter->CallFunc_IsValid( fMethodCall ) && fClass.Name() == "string" ) {
      gInterpreter->CallFunc_SetFuncProto(
         fMethodCall,
         gcl,
         "basic_string<char,char_traits<char>,allocator<char> >",
         callString.c_str(),
         fMethod.IsConstant(),
         &fOffset);      // <- no kExactMatch as that will fail
   }
// -- CLING WORKAROUND

// CLING WORKAROUND -- The number of arguments is not always correct (e.g. when there
//                     are default parameters, causing the callString to be wrong and
//                     the exact match to fail); or the method may have been inline or
//                     be compiler generated. In all those cases the exact match fails,
//                     whereas the conversion match sometimes works.
   if ( ! gInterpreter->CallFunc_IsValid( fMethodCall ) ) {
      gInterpreter->CallFunc_SetFuncProto(
         fMethodCall,
         gcl,
         (Bool_t)fMethod == true ? fMethod.Name().c_str() : fClass.Name().c_str(),
         callString.c_str(),
         fMethod.IsConstant(),
         &fOffset );     // <- no kExactMatch as that will fail
   }
// -- CLING WORKAROUND

// CLING WORKAROUND -- (Actually, this may be a feature rather than workaround: the
//                     question is whether this should live in TClass or here.)
//                     For some reason this code does not work (crashes) for several
//                     vector types (but not all) from "cintdlls", so skip on int.
//                     Note that vector<double> is instantiated at application startup.
   if ( ! gInterpreter->CallFunc_IsValid( fMethodCall ) &&
        fClass.Name().find( '<' ) != std::string::npos &&
        fClass.Name().find( "int" ) == std::string::npos ) {
      const std::string& cName = fClass.Name();
      if ( TClassEdit::IsSTLCont( cName.c_str() ) ) {
         gROOT->ProcessLine( (std::string("template class ") +
            (cName.find( "std::", 0, 5 ) == std::string::npos ? "std::" : "") +
            fClass.Name() + ";").c_str() );
      } else {
         gROOT->ProcessLine( ("template class " + fClass.Name() + ";").c_str() );
      }
   }
// -- CLING WORKAROUND

   if ( ! gInterpreter->CallFunc_IsValid( fMethodCall ) ) {
   // CLING WORKAROUND -- checking (Bool_t)fMethod (i.e. whether this is a method
   //                     rather than a ctor), remains necessary (for ctors, there
   //                     is another workaround later using TClass::New())
      if ( (Bool_t)fMethod == false || fMethod.Name() == fClass.Name() )
         return kTRUE;
   // -- CLING WORKAROUND
      PyErr_Format( PyExc_RuntimeError, "could not resolve %s::%s(%s)",
         fClass.Name().c_str(), fMethod.Name().c_str(), callString.c_str() );
      gInterpreter->CallFunc_Delete( fMethodCall );
      fMethodCall = 0;
      return kFALSE;
   }

   return kTRUE;
}

//____________________________________________________________________________
Bool_t PyROOT::TMethodHolder::InitExecutor_( TExecutor*& executor )
{
// install executor conform to the return type

// CLING WORKAROUND -- #100728: can have received the wrong overload
   MethodInfo_t* mi = gInterpreter->CallFunc_FactoryMethod(fMethodCall);
   if ( gInterpreter->MethodInfo_IsValid( mi ) &&
     /* beats me why void needs to be filtered, but it's always the wrong answer AFAICS */
        gInterpreter->MethodInfo_TypeNormalizedName( mi ) != "void" ) {
      executor = CreateExecutor( gInterpreter->MethodInfo_TypeNormalizedName( mi ) );
   } else {
//-- CLING WORKAROUND
   executor = CreateExecutor( (Bool_t)fMethod == true ?
      fMethod.TypeOf().ReturnType().Name( Rflx::QUALIFIED | Rflx::SCOPED | Rflx::FINAL )
      : fClass.Name( Rflx::SCOPED | Rflx::FINAL ) );
// CLING WORKAROUND -- #100728:
   }
   gInterpreter->MethodInfo_Delete( mi );
//-- CLING WORKAROUND

   if ( ! executor )
      return kFALSE;

   return kTRUE;
}

//____________________________________________________________________________
void PyROOT::TMethodHolder::CreateSignature_()
{
// built a signature a la TFunction::GetSignature as python string, using Adapters
   Int_t ifirst = 0;
   fSignature = "(";
   const size_t nArgs = fMethod.FunctionParameterSize();
   for ( size_t iarg = 0; iarg < nArgs; ++iarg ) {
      if ( ifirst ) fSignature += ", ";

      fSignature += fMethod.TypeOf().FunctionParameterAt( iarg ).Name( Rflx::QUALIFIED );

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
const std::string& PyROOT::TMethodHolder::GetSignatureString()
{
// construct python string from the method's signature
   if ( fSignature.empty() )
      CreateSignature_();

   return fSignature;
}

//____________________________________________________________________________
void PyROOT::TMethodHolder::SetPyError_( PyObject* msg )
{
// helper to report errors in a consistent format (derefs msg)
   PyObject *etype, *evalue, *etrace;
   PyErr_Fetch( &etype, &evalue, &etrace );

   std::string details = "";
   if ( evalue ) {
      PyObject* descr = PyObject_Str( evalue );
      if ( descr ) {
         details = PyROOT_PyUnicode_AsString( descr );
         Py_DECREF( descr );
      }
   }

   Py_XDECREF( etype ); Py_XDECREF( evalue ); Py_XDECREF( etrace );

   PyObject* doc = GetDocString();

   if ( details.empty() ) {
      PyErr_Format( PyExc_TypeError, "%s =>\n    %s", PyROOT_PyUnicode_AsString( doc ),
         msg ? PyROOT_PyUnicode_AsString( msg ) : ""  );
   } else if ( msg ) {
      PyErr_Format( PyExc_TypeError, "%s =>\n    %s (%s)",
         PyROOT_PyUnicode_AsString( doc ), PyROOT_PyUnicode_AsString( msg ), details.c_str() );
   } else {
      PyErr_Format( PyExc_TypeError, "%s =>\n    %s",
         PyROOT_PyUnicode_AsString( doc ), details.c_str() );
   }

   Py_DECREF( doc );
   Py_XDECREF( msg );
}

//- constructors and destructor ----------------------------------------------
PyROOT::TMethodHolder::TMethodHolder( const TScopeAdapter& klass, const TMemberAdapter& method ) :
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
PyROOT::TMethodHolder::TMethodHolder( const TMethodHolder& other ) :
       PyCallable( other ), fMethod( other.fMethod ), fClass( other.fClass )
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
      fClass  = other.fClass;
      fMethod = other.fMethod;
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
   return PyROOT_PyUnicode_FromString( GetSignatureString().c_str() );
}

//____________________________________________________________________________
PyObject* PyROOT::TMethodHolder::GetPrototype()
{
// construct python string from the method's prototype
   return PyROOT_PyUnicode_FromFormat( "%s%s %s::%s%s",
      ( fMethod.IsStatic() ? "static " : "" ),
      fMethod.TypeOf().ReturnType().Name( Rflx::QUALIFIED | Rflx::SCOPED ).c_str(),
      fMethod.DeclaringScope().Name().c_str(), fMethod.Name().c_str(),
      GetSignatureString().c_str() );
}

//____________________________________________________________________________
Int_t PyROOT::TMethodHolder::GetPriority()
{
// Method priorities exist (in lieu of true overloading) there to prevent
// void* or <unknown>* from usurping otherwise valid calls. TODO: extend this
// to favour classes that are not bases.

   Int_t priority = 0;

   const size_t nArgs = fMethod.FunctionParameterSize();
   for ( size_t iarg = 0; iarg < nArgs; ++iarg ) {
      const TScopeAdapter& arg = fMethod.TypeOf().FunctionParameterAt( iarg );

   // the following numbers are made up and may cause problems in specific
   // situations: use <obj>.<meth>.disp() for choice of exact dispatch
      if ( arg.IsClass() || arg.IsStruct() ) {
         if ( ! (Bool_t)arg ) {
            priority -= 10000;   // class is gibberish
         } else if ( ! arg.IsComplete() ) {
         // class is known, but no dictionary available, 2 more cases: * and &
            const std::string aname = arg.Name( Rflx::QUALIFIED );
            if ( aname[ aname.size() - 1 ] == '&' )
               priority -= 3000;
            else
               priority -= 1000; // prefer pointer passing over reference
         }
      } else {
         const std::string aname = arg.Name( Rflx::FINAL | Rflx::QUALIFIED );
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

//____________________________________________________________________________
Int_t PyROOT::TMethodHolder::GetMaxArgs()
{
   return fMethod.FunctionParameterSize();
}

//____________________________________________________________________________
PyObject* PyROOT::TMethodHolder::GetArgSpec( Int_t iarg )
{
// Build a string representation of the arguments list.
   if ( iarg >= (int)fMethod.FunctionParameterSize() )
      return 0;

   std::string argrep = fMethod.TypeOf().FunctionParameterAt( iarg ).Name( Rflx::QUALIFIED );

   const std::string& parname = fMethod.FunctionParameterNameAt( iarg );
   if ( ! parname.empty() ) {
      argrep += " ";
      argrep += parname;
   }

   return PyROOT_PyUnicode_FromString( argrep.c_str() );
}

//____________________________________________________________________________
PyObject* PyROOT::TMethodHolder::GetArgDefault( Int_t iarg )
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
PyObject* PyROOT::TMethodHolder::GetScope()
{
// Get or build the scope of this method.
   return MakeRootClassFromString( fMethod.DeclaringScope().Name( Rflx::SCOPED | Rflx::FINAL ) );
}

//____________________________________________________________________________
Bool_t PyROOT::TMethodHolder::Initialize()
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
Bool_t PyROOT::TMethodHolder::SetMethodArgs( PyObject* args, Long_t user )
{
// clean slate
   if ( fMethodCall )
      gInterpreter->CallFunc_ResetArg( fMethodCall );

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
PyObject* PyROOT::TMethodHolder::Execute( void* self, Bool_t release_gil )
{
// call the interface method
   R__LOCKGUARD2( gGlobalMutex );
   PyObject* result = 0;

   if ( Utility::gSignalPolicy == Utility::kFast ) {
   // bypasses ROOT try block (i.e. segfaults will abort)
      result = CallFast( self, release_gil );
   } else {
   // at the cost of ~10% performance, don't abort the interpreter on any signal
      result = CallSafe( self, release_gil );
   }

   if ( result && result != (PyObject*)TPyExceptionMagic
           && Utility::PyErr_Occurred_WithGIL() ) {
   // can happen in the case of a CINT error: trigger exception processing
      Py_DECREF( result );
      result = 0;
   } else if ( ! result && PyErr_Occurred() )
      SetPyError_( 0 );

   return result;
}

//____________________________________________________________________________
PyObject* PyROOT::TMethodHolder::operator()(
      ObjectProxy* self, PyObject* args, PyObject* kwds, Long_t user, Bool_t release_gil )
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
   TClass* derived = self->ObjectIsA();
   if ( derived ) {
   // reset this method's offset for the object as appropriate
      TClass* base = (TClass*)fClass.Id();
      if ( derived != base )
         fOffset = Utility::GetObjectOffset( derived->GetName(), base->GetClassInfo(), object );
      else fOffset = 0;
   }

// actual call; recycle self instead of returning new object for same address objects
   ObjectProxy* pyobj = (ObjectProxy*)Execute( object, release_gil );
   if ( pyobj != (ObjectProxy*)TPyExceptionMagic &&
        ObjectProxy_Check( pyobj ) &&
        pyobj->GetObject() == object &&
        derived && pyobj->ObjectIsA() == derived ) {
      Py_INCREF( (PyObject*)self );
      Py_DECREF( pyobj );
      return (PyObject*)self;
   }

   return (PyObject*)pyobj;
}
