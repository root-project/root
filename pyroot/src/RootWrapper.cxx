// @(#)root/pyroot:$Name:  $:$Id: RootWrapper.cxx,v 1.1 2004/04/27 06:28:48 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "RootWrapper.h"
#include "ObjectHolder.h"
#include "MethodDispatcher.h"
#include "ConstructorDispatcher.h"
#include "MethodHolder.h"
#include "MemoryRegulator.h"
#include "Utility.h"

// ROOT
#include "TROOT.h"
#include "TSystem.h"
#include "TBenchmark.h"
#include "TRandom.h"
#include "TApplication.h"
#include "TStyle.h"
#include "TGeometry.h"
#include "TMethod.h"
#include "TMethodArg.h"

// Standard
#include <cassert>
#include <map>
#include <string>
#ifndef NDEBUG
#include <cstring>
#endif

// Special for Unixes
#if defined(linux) || defined(sun)
  #include "dlfcn.h"
#endif

#include <iostream>


//- data _______________________________________________________________________
namespace {

   PyObject* g_modroot = 0;

   std::map< std::string, std::string > c2p_operators_;

   class InitOperators_ {
   public:
      InitOperators_() {
         c2p_operators_[ "[]" ]  = "__getitem__";
         c2p_operators_[ "()" ]  = "__call__";
         c2p_operators_[ "+" ]   = "__add__";
         c2p_operators_[ "-" ]   = "__sub__";
         c2p_operators_[ "*" ]   = "__mul__";
         c2p_operators_[ "/" ]   = "__div__";
         c2p_operators_[ "%" ]   = "__mod__";
         c2p_operators_[ "**" ]  = "__pow__";
         c2p_operators_[ "<<" ]  = "__lshift__";
         c2p_operators_[ ">>" ]  = "__rshift__";
         c2p_operators_[ "&" ]   = "__and__";
         c2p_operators_[ "|" ]   = "__or__";
         c2p_operators_[ "^" ]   = "__xor__";
         c2p_operators_[ "+=" ]  = "__iadd__";
         c2p_operators_[ "-=" ]  = "__isub__";
         c2p_operators_[ "*=" ]  = "__imul__";
         c2p_operators_[ "/=" ]  = "__idiv__";
         c2p_operators_[ "/=" ]  = "__imod__";
         c2p_operators_[ "**=" ] = "__ipow__";
         c2p_operators_[ "<<=" ] = "__ilshift__";
         c2p_operators_[ ">>=" ] = "__irshift__";
         c2p_operators_[ "&=" ]  = "__iand__";
         c2p_operators_[ "|=" ]  = "__ior__";
         c2p_operators_[ "^=" ]  = "__ixor__";
         c2p_operators_[ "==" ]  = "__eq__";
         c2p_operators_[ "!=" ]  = "__ne__";
         c2p_operators_[ ">" ]   = "__gt__";
         c2p_operators_[ "<" ]   = "__lt__";
         c2p_operators_[ ">=" ]  = "__ge__";
         c2p_operators_[ "<=" ]  = "__le__";
      }
   } initOperators_;

} // unnamed namespace


//- helpers --------------------------------------------------------------------
namespace {

   class PyROOTApplication : public TApplication {
   public:
      PyROOTApplication( const char* acn, int* argc, char** argv ) :
            TApplication( acn, argc, argv ) {
         SetReturnFromRun( kTRUE );          // prevents ROOT from exiting python
      }
   };

   inline void addToScope( const char* label, TObject* obj, TClass* cls ) {
      PyModule_AddObject(
         PyImport_AddModule( const_cast< char* >( "PyROOT" ) ),
         const_cast< char* >( label ),
         PyROOT::bindRootObject( new PyROOT::ObjectHolder( obj, cls, false ) )
      );
   }

   void addToClass( const char* label, PyCFunction cfunc, PyObject* cls, PyObject* dct ) {
      PyMethodDef* pdef = new PyMethodDef;
      pdef->ml_name  = const_cast< char* >( label );
      pdef->ml_meth  = cfunc;
      pdef->ml_flags = METH_VARARGS;
      pdef->ml_doc   = NULL;

      PyObject* func = PyCFunction_New( pdef, NULL );
      PyObject* method = PyMethod_New( func, NULL, cls );
      PyDict_SetItemString( dct, pdef->ml_name, method );
      Py_DECREF( func );
      Py_DECREF( method );
   }

   void destroyObjectHolder( void* oh ) {
      delete reinterpret_cast< PyROOT::ObjectHolder* >( oh );
   }

} // unnamed namespace


//------------------------------------------------------------------------------
void PyROOT::initRoot() {
// setup interpreter locks to allow for threading in ROOT
   PyEval_InitThreads();

// The following libs are also useful to have, make sure they are loaded...
#if defined(linux) || defined(sun)
   dlopen( "libCint.so",   RTLD_GLOBAL | RTLD_LAZY );
   dlopen( "libCore.so",   RTLD_GLOBAL | RTLD_LAZY );
   dlopen( "libGpad.so",   RTLD_GLOBAL | RTLD_LAZY );
   dlopen( "libGraf.so",   RTLD_GLOBAL | RTLD_LAZY );
   dlopen( "libMatrix.so", RTLD_GLOBAL | RTLD_LAZY );
   dlopen( "libHist.so",   RTLD_GLOBAL | RTLD_LAZY );
   dlopen( "libTree.so",   RTLD_GLOBAL | RTLD_LAZY );
   dlopen( "libGraf3d.so", RTLD_GLOBAL | RTLD_LAZY );
   dlopen( "libGeom.so",   RTLD_GLOBAL | RTLD_LAZY );
#endif

// setup root globals (bind later)
   if ( !gBenchmark ) gBenchmark = new TBenchmark();
   if ( !gStyle ) gStyle = new TStyle();
   if ( !gApplication ) {
      char* argv[1];
      int argc = 0;
      gApplication = new PyROOTApplication( "PyROOT", &argc, argv );
   }

// bind ROOT globals (ObjectHolder instances will be properly destroyed)
   addToScope( "gROOT", gROOT, gROOT->IsA() );
   addToScope( "gSystem", gSystem, gSystem->IsA() );
   addToScope( "gRandom", gRandom, gRandom->IsA() );

// the following are safer, done this way
   addToScope( "gBenchmark", gBenchmark, TBenchmark::Class() );
   addToScope( "gStyle", gStyle, TStyle::Class() );

// memory management
   gROOT->GetListOfCleanups()->Add( new MemoryRegulator() );
}


int PyROOT::buildRootClassDict( TClass* cls, PyObject* pyclass, PyObject* dct ) {
   assert( cls != 0 );
   std::string className = cls->GetName();

// load all public methods
   typedef std::map< std::string, MethodDispatcher > DispatcherCache_t;
   DispatcherCache_t dispatcherCache;

   TIter nextmethod( cls->GetListOfAllPublicMethods() );
   while ( TMethod* mt = (TMethod*)nextmethod() ) {
      std::string mtName = mt->GetName();

   // filter C++ destructors
      if ( mtName[0] == '~' )
         continue;

   // operators
      if ( 8 < mtName.size() && mtName.substr( 0, 8 ) == "operator" ) {
         std::string op = mtName.substr( 8, std::string::npos );

         if ( op == "=" || op == " new" || op == " new[]" ||
              op == " delete" || op == " delete[]" )
            continue;

      // map C++ operator to python equivalent
         std::map< std::string, std::string >::iterator pop = c2p_operators_.find( op );
         if ( pop != c2p_operators_.end() )
            mtName = pop->second;
      }

   // allowable range in number of arguments
      unsigned maxArgs = mt->GetNargs() + 1;
      unsigned minArgs = maxArgs - mt->GetNargsOpt();

   // some more filtering
      if ( maxArgs == 2 && minArgs == 2 &&
           mtName == ((TMethodArg*)( TIter( mt->GetListOfMethodArgs() )() ))->GetTypeName() )
         continue;                             // don't expose copy constructor

   // use full signature as a doc string
      std::string doc( mt->GetReturnTypeName() );
      (((((doc += ' ') += className) += "::") += mtName) += mt->GetSignature());

   // construct method dispatchers
      if ( mtName == className ) {        // found a constructor
      // lookup dispatcher and store method
         static const std::string initname("__init__");
         MethodDispatcher& md = (*(dispatcherCache.insert(
            std::make_pair( initname, MethodDispatcher( "__init__" ) ) ).first)).second;
         md.addMethod( new ConstructorDispatcher( cls, mt ) );
      }
      else {                              // found a member function
      // lookup dispatcher and store method
         MethodDispatcher& md = (*(dispatcherCache.insert(
            std::make_pair( mtName, MethodDispatcher( mtName ) ) ).first)).second;
         md.addMethod( new MethodHolder( cls, mt ) );
      }

   }

// add the methods to the class dictionary
   for ( DispatcherCache_t::iterator imd = dispatcherCache.begin();
         imd != dispatcherCache.end(); ++imd ) {
      MethodDispatcher::addToClass(
         new MethodDispatcher( imd->second ), pyclass, dct );
   }

// add null/non-null testing
   addToClass( "__zero__", IsZero, pyclass, dct );
   addToClass( "__nonzero__", IsNotZero, pyclass, dct );

// all ok, done
   return 0;
}


PyObject* PyROOT::makeRootClass( PyObject*, PyObject* args ) {
   const char* className = PyString_AsString( PyTuple_GetItem( args, 0 ) );

   if ( PyErr_Occurred() )
      return 0;

// retrieve ROOT class (this verifies className)
   TClass* cls = gROOT->GetClass( className );
   if ( cls == 0 ) {
      PyErr_SetString( PyExc_TypeError,
         ( "requested class " + std::string( className ) + " does not exist" ).c_str() );
      return 0;
   }

// get the ROOT module dictionary, if available
   if ( !g_modroot ) {
      g_modroot = PyImport_AddModule( const_cast< char* >( "ROOT" ) );
      Py_XINCREF( g_modroot );
   }

   PyObject* pyclass = 0;
   if ( g_modroot != 0 ) {
   // attempt to retrieve the class representation from the ROOT module
      pyclass = PyObject_GetAttrString( g_modroot, const_cast< char* >( className ) );
   }

   if ( pyclass == 0 ) {
   // ignore error generated from the failed lookup
      PyErr_Clear();

   // start with an empty dictionary
      PyObject* dct = PyDict_New();

   // create a fresh Python class, given the name and the empty dictionary
      pyclass = PyClass_New( 0, dct, PyString_FromString( className ) );

   // fill the dictionary, if successful
      if ( pyclass != 0 ) {
         buildRootClassDict( cls, pyclass, dct );
         if ( g_modroot != 0 ) {
            Py_INCREF( pyclass );            // PyModule_AddObject steals reference
            PyModule_AddObject( g_modroot, const_cast< char* >( className ), pyclass );
         }
      }
      else
         PyErr_SetString( PyExc_TypeError, "could not allocate new class" );

      Py_DECREF( dct );
   }

   return pyclass;
}


PyObject* PyROOT::bindRootObject( ObjectHolder* obj ) {
   TClass* cls = obj->objectIsA();

// only known objects will be bound
   if ( cls != 0 ) {
      PyObject* args = PyTuple_New( 1 );
      PyTuple_SetItem( args, 0, PyString_FromString( cls->GetName() ) );
      PyObject* pyclass = makeRootClass( Py_None, args );
      Py_DECREF( args );

      if ( pyclass ) {
      // private to the object is the instance holder
         PyObject* dct = PyDict_New();
         PyObject* cobj = PyCObject_FromVoidPtr( obj, destroyObjectHolder );
         PyDict_SetItem( dct, Utility::theObjectString_, cobj );
         Py_DECREF( cobj );

      // instantiate an object of this class, with the holder added to it
         PyObject* newObject = PyInstance_NewRaw( pyclass, dct );
         Py_DECREF( pyclass );
         Py_DECREF( dct );

      // register and return if successful
         if ( newObject != 0 ) {
            MemoryRegulator::RegisterObject( newObject, obj->getObject() );
            return newObject;
         }
      }
   }

// confused ...
   PyErr_SetString( PyExc_TypeError, "failed to bind ROOT object" );
   return 0;
}
