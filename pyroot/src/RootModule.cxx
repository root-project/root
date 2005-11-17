// @(#)root/pyroot:$Name:  $:$Id: RootModule.cxx,v 1.20 2005/10/25 05:13:15 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "PyRootType.h"
#include "ObjectProxy.h"
#include "MethodProxy.h"
#include "PropertyProxy.h"
#include "PyBufferFactory.h"
#include "RootWrapper.h"
#include "Utility.h"

// ROOT
#include "TROOT.h"
#include "TSystem.h"
#include "TInterpreter.h"
#include "TApplication.h"
#include "TBenchmark.h"
#include "TStyle.h"

// CINT
#include "Api.h"

// Standard
#include <string>


//- data -----------------------------------------------------------------------
PyObject* gRootModule = 0;


//- private helpers ------------------------------------------------------------
namespace {

   using namespace PyROOT;

//______________________________________________________________________________
   class TPyROOTApplication : public TApplication {
   public:
      TPyROOTApplication( const char* acn, int* argc, char** argv ) :
            TApplication( acn, argc, argv )
      {
      // follow TRint to minimize differences with CINT
         ProcessLine( "#include <iostream>", kTRUE );
         ProcessLine( "#include <_string>",  kTRUE ); // for std::string iostream.
         ProcessLine( "#include <vector>",   kTRUE ); // needed because they're used within the
         ProcessLine( "#include <pair>",     kTRUE ); //  core ROOT dicts and CINT won't be able
                                                      //  to properly unload these files

      // allow the usage of ClassDef and ClassImp in interpreted macros
         ProcessLine( "#include <RtypesCint.h>", kTRUE );

      // disallow the interpretation of Rtypes.h, TError.h and TGenericClassInfo.h
         ProcessLine( "#define ROOT_Rtypes 0", kTRUE );
         ProcessLine( "#define ROOT_TError 0", kTRUE );
         ProcessLine( "#define ROOT_TGenericClassInfo 0", kTRUE );

      // the following libs are also useful to have, make sure they are loaded...
         gROOT->LoadClass("TMinuit",     "Minuit");
         gROOT->LoadClass("TPostScript", "Postscript");
         gROOT->LoadClass("THtml",       "Html");

      // save current interpreter context
         gInterpreter->SaveContext();
         gInterpreter->SaveGlobalsContext();

      // prevent ROOT from exiting python
         SetReturnFromRun( kTRUE );
      }
   };

//____________________________________________________________________________
   PyDictEntry* RootLookDictString( PyDictObject* mp, PyObject* key, Long_t hash )
   {
   // first search dictionary itself
      PyDictEntry* ep = (*gDictLookupOrg)( mp, key, hash );
      if ( ! ep || ep->me_value != 0 )
         return ep;

   // filter for builtins
      if ( PyDict_GetItem( PyEval_GetBuiltins(), key ) != 0 ) {
         return ep;
      }

   // filter python private variables (C++ discourages __ as variable start)
      if ( ! PyString_CheckExact( key ) )
         return ep;

      std::string strkey = PyString_AS_STRING( key );
      if ( 2 < strkey.size() && strkey.substr( 0, 2 ) == "__" )
         return ep;

   // all failed, start calling into ROOT
      gDictLookupActive = kTRUE;

   // attempt to get ROOT enum/global/class
      PyObject* val = PyObject_GetAttr( gRootModule, key );

      if ( ! val ) {
         PyErr_Clear();
         val = MakeRootClassFromString( strkey );
      }

      if ( ! val ) {
         PyErr_Clear();
         val = GetRootGlobalFromString( strkey );
      }

      if ( ! val ) {
         PyErr_Clear();
         TObject* object = gROOT->FindObject( strkey.c_str() );
         if ( object != 0 )
            val = BindRootObject( object, object->IsA() );
      }

      if ( val != 0 ) {
      // success ...
         if ( PropertyProxy_Check( val ) ) {
         // pretend something was actually found, but don't add to dictionary
            ep->me_key   = key;
            ep->me_hash  = hash;
            ep->me_value = val->ob_type->tp_descr_get( val, NULL, NULL );
         } else {
         // add reference to ROOT entity in the module dictionary
            Py_INCREF( key );

            if ( ! ep->me_key )
               mp->ma_fill++;
            else
               Py_DECREF( ep->me_key );
            ep->me_key   = key;
            ep->me_hash  = hash;
            ep->me_value = val;
            mp->ma_used++;
         }
      }

   // stopped calling into ROOT
      gDictLookupActive = kFALSE;

      return ep;
   }

//____________________________________________________________________________
   PyObject* InitRootApplication()
   {
      if ( ! gApplication ) {
      // retrieve arg list from python, translate to raw C, pass on
         PyObject* argl = PySys_GetObject( const_cast< char* >( "argv" ) );

         int argc = argl ? PyList_Size( argl ) : 1;
         char** argv = new char*[ argc ];
         for ( int i = 1; i < argc; ++i )
            argv[ i ] = PyString_AS_STRING( PyList_GET_ITEM( argl, i ) );
         argv[ 0 ] = Py_GetProgramName();

         gApplication = new TPyROOTApplication( "PyROOT", &argc, argv );

      // CINT message callback (only if loaded from python, i.e. !gApplication)
         G__set_errmsgcallback( (void*)&PyROOT::Utility::ErrMsgCallback );
      }

   // setup some more handy ROOT globals
      if ( ! gBenchmark ) gBenchmark = new TBenchmark();
      if ( ! gStyle ) gStyle = new TStyle();

      if ( ! gProgName )              // should be set by TApplication
         gSystem->SetProgname( Py_GetProgramName() );

      Py_INCREF( Py_None );
      return Py_None;
   }

//____________________________________________________________________________
   PyObject* SetRootLazyLookup( PyObject*, PyObject* args )
   {
      PyObject* dict = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "O!" ), &PyDict_Type, &dict ) )
         return 0;

      ((DictLookup_t&)((PyDictObject*)dict)->ma_lookup) = RootLookDictString;

      Py_INCREF( Py_None );
      return Py_None;
   }

//____________________________________________________________________________
   PyObject* MakeRootTemplateClass( PyObject*, PyObject* args )
   {
   // args is class name + template arguments, build full instantiation
      int nArgs = (int)PyTuple_GET_SIZE( args );
      if ( nArgs < 2 ) {
         PyErr_Format( PyExc_TypeError, "too few arguments for template instantiation" );
         return 0;
      }

   // copy initial argument (no check, comes from internal class)
      PyObject* pyname = PyString_FromString(
         PyString_AS_STRING( PyTuple_GET_ITEM( args, 0 ) ) );

   // build "< type, type, ... >" part of class name
      PyString_ConcatAndDel( &pyname, PyString_FromString( "<" ) );
      for ( int i = 1; i < nArgs; ++i ) {
      // add type as string to name
         PyObject* tn = PyTuple_GET_ITEM( args, i );
         if ( PyString_Check( tn ) )
            PyString_Concat( &pyname, tn );
         else {
            PyObject* tpName = PyObject_GetAttrString( tn, const_cast< char* >( "__name__" ) );
            if ( PyErr_Occurred() ) {
               Py_DECREF( pyname );
               return 0;
            }

         // special case for strings
            if ( strcmp( PyString_AS_STRING( tpName ), "str" ) == 0 ) {
               Py_DECREF( tpName );
               tpName = PyString_FromString( "std::string" );
            }

            PyString_ConcatAndDel( &pyname, tpName );
         }

      // add a comma, as needed
         if ( i != nArgs - 1 )
            PyString_ConcatAndDel( &pyname, PyString_FromString( "," ) );
      }

   // close template name; prevent '>>', which should be '> >'
      if ( PyString_AsString( pyname )[ PyString_Size( pyname ) - 1 ] == '>' )
         PyString_ConcatAndDel( &pyname, PyString_FromString( " >" ) );
      else
         PyString_ConcatAndDel( &pyname, PyString_FromString( ">" ) );

      std::string name = PyString_AS_STRING( pyname );
      Py_DECREF( pyname );

      return MakeRootClassFromString( name );
   }

//____________________________________________________________________________
   PyObject* AddressOf( PyObject*, PyObject* args )
   {
      ObjectProxy* pyobj = 0;
      PyObject* pyname = 0;
      if ( PyArg_ParseTuple( args, const_cast< char* >( "O|S" ), &pyobj, &pyname ) &&
           ObjectProxy_Check( pyobj ) && pyobj->fObject ) {

         if ( pyname != 0 ) {
         // locate property proxy for offset info
            PropertyProxy* pyprop = 0;

            PyObject* pyclass = PyObject_GetAttrString(
               (PyObject*)pyobj, const_cast< char* >( "__class__" ) );

            if ( pyclass ) {
               PyObject* dict = PyObject_GetAttrString( pyclass, const_cast< char* >( "__dict__" ) );
               pyprop = (PropertyProxy*)PyObject_GetItem( dict, pyname );
               Py_DECREF( dict );
            }
            Py_XDECREF( pyclass );

            if ( PropertyProxy_Check( pyprop ) ) {
            // this is an address of a value (i.e. &myobj->prop)
               PyObject* pybuf = BufFac_t::Instance()->PyBuffer_FromMemory(
                  (Long_t*)pyprop->GetAddress( pyobj ), 1 );
               Py_DECREF( pyprop );
               return pybuf;
            }

            Py_XDECREF( pyprop );

            PyErr_Format( PyExc_TypeError,
               "%s is not a valid data member", PyString_AS_STRING( pyname ) );
            return 0;
         }

      // this is an address of an address (i.e. &myobj, with myobj of type MyObj*)
         return BufFac_t::Instance()->PyBuffer_FromMemory( (Long_t*)&pyobj->fObject, 1 );
      }

      PyErr_SetString( PyExc_ValueError, "invalid argument for AddressOf()" );
      return 0;
   }

//____________________________________________________________________________
   PyObject* MakeNullPointer( PyObject*, PyObject* args )
   {
      int argc = PyTuple_GET_SIZE( args );
      if ( argc != 0 && argc != 1 ) {
         PyErr_Format( PyExc_TypeError,
            "MakeNullPointer takes at most 1 argument (%d given)", argc );
         return 0;
      }

   // no class given, use None as generic
      if ( argc == 0 ) {
         Py_INCREF( Py_None );
         return Py_None;
      }

   // check argument for either string name, or named python object
      PyObject* pyname = PyTuple_GET_ITEM( args, 0 );
      if ( ! PyString_Check( pyname ) ) {
         PyObject* nattr = PyObject_GetAttrString( pyname, "__name__" );
         if ( nattr )
            pyname = nattr;
         pyname = PyObject_Str(	pyname );
         Py_XDECREF( nattr );
      } else {
         Py_INCREF( pyname );
      }

      TClass* klass = gROOT->GetClass( PyString_AS_STRING( pyname ) );
      Py_DECREF( pyname );

      if ( ! klass ) {
         PyErr_SetString( PyExc_TypeError,
            "MakeNullPointer expects a valid class or class name as an argument" );
         return 0;
      }

      return BindRootObjectNoCast( 0, klass, kFALSE );
   }

//____________________________________________________________________________
   PyObject* SetMemoryPolicy( PyObject*, PyObject* args )
   {
      PyObject* policy = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "O!" ), &PyInt_Type, &policy ) )
         return 0;

      Long_t l = PyInt_AS_LONG( policy );
      if ( Utility::SetMemoryPolicy( (Utility::EMemoryPolicy)l ) ) {
         Py_INCREF( Py_None );
         return Py_None;
      }

      PyErr_Format( PyExc_ValueError, "Unknown policy %ld", l );
      return 0;
  }

//____________________________________________________________________________
   PyObject* SetSignalPolicy( PyObject*, PyObject* args )
   {
      PyObject* policy = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "O!" ), &PyInt_Type, &policy ) )
         return 0;

      Long_t l = PyInt_AS_LONG( policy );
      if ( Utility::SetSignalPolicy( (Utility::ESignalPolicy)l ) ) {
         Py_INCREF( Py_None );
         return Py_None;
      }

      PyErr_Format( PyExc_ValueError, "Unknown policy %ld", l );
      return 0;
  }

//____________________________________________________________________________
   PyObject* SetOwnership( PyObject*, PyObject* args )
   {
      ObjectProxy* pyobj = 0; PyObject* pykeep = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "O!O!" ),
                &ObjectProxy_Type, (PyObject*)&pyobj, &PyInt_Type, &pykeep ) )
         return 0;

      (Bool_t)PyLong_AsLong( pykeep ) ? pyobj->HoldOn() : pyobj->Release();

      Py_INCREF( Py_None );
      return Py_None;
   }

} // unnamed namespace


//- data -----------------------------------------------------------------------
static PyMethodDef gPyROOTMethods[] = {
   { (char*) "makeRootClass", (PyCFunction)PyROOT::MakeRootClass,
     METH_VARARGS, (char*) "PyROOT internal function" },
   { (char*) "getRootGlobal", (PyCFunction)PyROOT::GetRootGlobal,
     METH_VARARGS, (char*) "PyROOT internal function" },
   { (char*) "InitRootApplication", (PyCFunction)InitRootApplication,
     METH_NOARGS,  (char*) "PyROOT internal function" },
   { (char*) "setRootLazyLookup", (PyCFunction)SetRootLazyLookup,
     METH_VARARGS, (char*) "PyROOT internal function" },
   { (char*) "MakeRootTemplateClass", (PyCFunction)MakeRootTemplateClass,
     METH_VARARGS, (char*) "PyROOT internal function" },
   { (char*) "AddressOf", (PyCFunction)AddressOf,
     METH_VARARGS, (char*) "Retrieve address of held object" },
   { (char*) "MakeNullPointer", (PyCFunction)MakeNullPointer,
     METH_VARARGS, (char*) "Create a NULL pointer of the given type" },
   { (char*) "SetMemoryPolicy", (PyCFunction)SetMemoryPolicy,
     METH_VARARGS, (char*) "Determines object ownership model" },
   { (char*) "SetSignalPolicy", (PyCFunction)SetSignalPolicy,
     METH_VARARGS, (char*) "Trap signals in safe mode to prevent interpreter abort" },
   { (char*) "SetOwnership", (PyCFunction)SetOwnership,
     METH_VARARGS, (char*) "Modify held C++ object ownership" },
   { NULL, NULL, 0, NULL }
};


//____________________________________________________________________________
extern "C" void initlibPyROOT()
{
   using namespace PyROOT;

// prepare for lazyness
   PyObject* dict = PyDict_New();
   gDictLookupOrg = (DictLookup_t)((PyDictObject*)dict)->ma_lookup;
   Py_DECREF( dict );

// setup PyROOT
   gRootModule = Py_InitModule( const_cast< char* >( "libPyROOT" ), gPyROOTMethods );
   if ( ! gRootModule )
      return;
   Py_INCREF( gRootModule );

// inject meta type
   if ( ! Utility::InitProxy( gRootModule, &PyRootType_Type, "PyRootType" ) )
      return;

// inject object proxy type
   if ( ! Utility::InitProxy( gRootModule, &ObjectProxy_Type, "ObjectProxy" ) )
      return;

// inject method proxy type
   if ( ! Utility::InitProxy( gRootModule, &MethodProxy_Type, "MethodProxy" ) )
      return;

// inject property proxy type
   if ( ! Utility::InitProxy( gRootModule, &PropertyProxy_Type, "PropertyProxy" ) )
      return;

// setup ROOT
   PyROOT::InitRoot();

// signal policy: don't abort interpreter in interactive mode
   Utility::SetSignalPolicy( gROOT->IsBatch() ? Utility::kFast : Utility::kSafe );
}
