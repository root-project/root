// @(#)root/pyroot:$Name:  $:$Id: RootWrapper.cxx,v 1.9 2004/07/30 06:31:18 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "RootWrapper.h"
#include "Pythonize.h"
#include "ObjectHolder.h"
#include "MethodDispatcher.h"
#include "ConstructorDispatcher.h"
#include "MethodHolder.h"
#include "ClassMethodHolder.h"
#include "PropertyHolder.h"
#include "MemoryRegulator.h"
#include "TPyClassGenerator.h"
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
#include "TBaseClass.h"
#include "TInterpreter.h"

// Standard
#include <assert.h>
#include <map>
#include <string>
#include <algorithm>
#include <vector>

// Special for Unixes
#if defined(linux) || defined(sun)
  #include "dlfcn.h"
#endif


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
         PyImport_AddModule( const_cast< char* >( "libPyROOT" ) ),
         const_cast< char* >( label ),
         PyROOT::bindRootObject( new PyROOT::ObjectHolder( obj, cls, false ) )
      );
   }

   void destroyObjectHolder( void* oh ) {
      delete reinterpret_cast< PyROOT::ObjectHolder* >( oh );
   }

} // unnamed namespace


//------------------------------------------------------------------------------
void PyROOT::initRoot() {
// setup interpreter locks to allow for threading in ROOT
   PyEval_InitThreads();

#if defined(linux) || defined(sun)
   dlopen( "libGpad.so",   RTLD_GLOBAL | RTLD_LAZY );
   dlopen( "libGraf.so",   RTLD_GLOBAL | RTLD_LAZY );
   dlopen( "libMatrix.so", RTLD_GLOBAL | RTLD_LAZY );
   dlopen( "libHist.so",   RTLD_GLOBAL | RTLD_LAZY );
   dlopen( "libTree.so",   RTLD_GLOBAL | RTLD_LAZY );
   dlopen( "libGraf3d.so", RTLD_GLOBAL | RTLD_LAZY );
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
   addToScope( "gInterpreter", gInterpreter, gInterpreter->IsA() );

// the following are safer, done this way
   addToScope( "gBenchmark", gBenchmark, TBenchmark::Class() );
   addToScope( "gStyle", gStyle, TStyle::Class() );

// memory management
   gROOT->GetListOfCleanups()->Add( new MemoryRegulator() );

// python side class construction, managed by ROOT
   gROOT->AddClassGenerator( new TPyClassGenerator() );
}


int PyROOT::buildRootClassDict( TClass* cls, PyObject* pyclass ) {
   assert( cls != 0 );
   std::string className = cls->GetName();

// load all public methods
   typedef std::map< std::string, MethodDispatcher > DispatcherCache_t;
   DispatcherCache_t dispCache;

   TIter nextmethod( cls->GetListOfMethods() );
   while ( TMethod* mt = (TMethod*)nextmethod() ) {
   // allow only public methods
      if ( !( mt->Property() & kIsPublic ) )
         continue;

   // retrieve method name
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

   // use full signature as a doc string
      std::string doc( mt->GetReturnTypeName() );
      (((((doc += ' ') += className) += "::") += mtName) += mt->GetSignature());

   // construct holder
      MethodHolder* pmh = 0;
      if ( mt->Property() & G__BIT_ISSTATIC )     // class method
         pmh = new ClassMethodHolder( cls, mt );
      else if ( mtName == className ) {           // constructor
         pmh = new ConstructorDispatcher( cls, mt );
         mtName = "__init__";
      }
      else                                        // member function
         pmh = new MethodHolder( cls, mt );

   // lookup method dispatcher and store method
      MethodDispatcher& md = (*(dispCache.insert( std::make_pair( mtName,
         MethodDispatcher( mtName, mt->Property() & G__BIT_ISSTATIC ) ) ).first)).second;
      md.addMethod( pmh );
   }

// add the methods to the class dictionary
   for ( DispatcherCache_t::iterator imd = dispCache.begin(); imd != dispCache.end(); ++imd ) {
      MethodDispatcher::addToClass( new MethodDispatcher( imd->second ), pyclass );
   }

// collect data members
   TIter nextmember( cls->GetListOfDataMembers() );
   while ( TDataMember* mb = (TDataMember*)nextmember() ) {
   // allow only public members
      if ( !( mb->Property() & kIsPublic ) )
         continue;

      PropertyHolder::addToClass( new PropertyHolder( mb ), pyclass );
   }

// all ok, done
   return 0;
}


PyObject* PyROOT::buildRootClassBases( TClass* cls ) {
   TList* allbases = cls->GetListOfBases();

   std::vector< std::string >::size_type nbases = 0;
   if ( allbases != 0 )
      nbases = allbases->GetSize();

// collect bases, remove duplicates
   std::vector< std::string > uqb;
   uqb.reserve( nbases );

   TIter nextbase( allbases );
   while ( TBaseClass* base = (TBaseClass*)nextbase() ) {
      std::string name = base->GetName();
      if ( std::find( uqb.begin(), uqb.end(), name ) == uqb.end() ) {
         uqb.push_back( name );
      }
   }

// allocate a tuple for the base classes, special case for no bases
   nbases = uqb.size();

   PyObject* pybases = PyTuple_New( nbases ? nbases : 1 );
   if ( ! pybases )
      return 0;

// build all the bases
   if ( nbases == 0 ) {
      Py_INCREF( &PyBaseObject_Type );
      PyTuple_SET_ITEM( pybases, 0, (PyObject*)&PyBaseObject_Type );
   }
   else {
      for ( std::vector< std::string >::size_type ibase = 0; ibase < nbases; ++ibase ) {
         PyObject* pyclass = makeRootClassFromString( uqb[ ibase ].c_str() );
         if ( ! pyclass ) {
            Py_DECREF( pybases );
            return 0;
         }

         PyTuple_SET_ITEM( pybases, ibase, pyclass );
      }
   }

   return pybases;
}


PyObject* PyROOT::makeRootClass( PyObject*, PyObject* args ) {
   const char* className = PyString_AsString( PyTuple_GetItem( args, 0 ) );

   if ( PyErr_Occurred() )
      return 0;

   return makeRootClassFromString( className );
}


PyObject* PyROOT::makeRootClassFromString( const char* className ) {
// retrieve ROOT class (this verifies className)
   TClass* cls = gROOT->GetClass( className, 0 );
   if ( cls == 0 && gInterpreter->AutoLoad( className ) != 0 )
      cls = gROOT->GetClass( className );

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

   // construct the base classes
      PyObject* pybases = buildRootClassBases( cls );
      if ( pybases != 0 ) {
      // create a fresh Python class, given bases, name and empty dictionary
         pyclass = PyObject_CallFunction( (PyObject*)&PyType_Type, "OOO",
            PyString_FromString( className ), pybases, dct );
         Py_DECREF( pybases );
      }

      Py_DECREF( dct );

   // fill the dictionary, if successful
      if ( pyclass != 0 ) {
         if ( buildRootClassDict( cls, pyclass ) != 0 ) {
         // something failed in building the dictionary
            Py_DECREF( pyclass );
            pyclass = 0;
         }
         else if ( g_modroot != 0 ) {
            Py_INCREF( pyclass );            // PyModule_AddObject steals reference
            PyModule_AddObject( g_modroot, const_cast< char* >( className ), pyclass );
         }
      }
   }

// add python-like features
   if ( ! pythonize( pyclass, className ) ) {
      Py_XDECREF( pyclass );
      pyclass = 0;
   }

// all done
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
      // use the old reference if the object already exists
         PyObject* oldObject = MemoryRegulator::RetrieveObject( obj->getObject() );
         if ( oldObject )
            return oldObject;

      // if not: instantiate an object of this class, with the holder added to it
         PyObject* newObject = PyType_GenericNew( (PyTypeObject*)pyclass, NULL, NULL );
         Py_DECREF( pyclass );

      // bind, register and return if successful
         if ( newObject != 0 ) {
         // private to the object is the instance holder
            PyObject* cobj = PyCObject_FromVoidPtr( obj, destroyObjectHolder );
            PyObject_SetAttr( newObject, Utility::theObjectString_, cobj );
            Py_DECREF( cobj );

         // memory management
            if ( obj->getObject() )
               MemoryRegulator::RegisterObject( newObject, obj->getObject() );

         // successful completion
            return newObject;
         }
      }
   }

// confused ...
   PyErr_SetString( PyExc_TypeError, "failed to bind ROOT object" );
   return 0;
}
