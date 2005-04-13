// @(#)root/pyroot:$Name:  $:$Id: RootWrapper.cxx,v 1.23 2005/04/05 05:20:44 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "ObjectProxy.h"
#include "MethodProxy.h"
#include "PropertyProxy.h"
#include "RootWrapper.h"
#include "Pythonize.h"
#include "MethodHolder.h"
#include "ConstructorHolder.h"
#include "ClassMethodHolder.h"
#include "MemoryRegulator.h"
#include "TPyClassGenerator.h"
#include "Utility.h"

// ROOT
#include "TROOT.h"
#include "TSystem.h"
#include "TBenchmark.h"
#include "TApplication.h"
#include "TStyle.h"
#include "TMethod.h"
#include "TMethodArg.h"
#include "TBaseClass.h"
#include "TInterpreter.h"
#include "TGlobal.h"
#include "DllImport.h"

// CINT
#include "Api.h"

// Standard
#include <assert.h>
#include <map>
#include <string>
#include <algorithm>
#include <vector>


//- data _______________________________________________________________________
R__EXTERN PyObject* gRootModule;


//- helpers --------------------------------------------------------------------
namespace {

   class PyROOTApplication : public TApplication {
   public:
      PyROOTApplication( const char* acn, int* argc, char** argv ) :
            TApplication( acn, argc, argv )
      {
         SetReturnFromRun( kTRUE );          // prevents ROOT from exiting python
      }
   };

   inline void addToScope( const char* label, TObject* obj, TClass* klass )
   {
      PyModule_AddObject( gRootModule, const_cast< char* >( label ),
         PyROOT::BindRootObject( obj, klass ) );
   }

} // unnamed namespace


//- public functions ---------------------------------------------------------
void PyROOT::InitRoot()
{
// setup interpreter locks to allow for threading in ROOT
   PyEval_InitThreads();

// setup ROOT globals (bind later)
   if ( !gBenchmark ) gBenchmark = new TBenchmark();
   if ( !gStyle ) gStyle = new TStyle();
   if ( !gApplication ) {
   // retrieve arg list from python, translate to raw C, pass on
      PyObject* argl = PySys_GetObject( "argv" );

      int argc = argl ? PyList_Size( argl ) : 1;
      char** argv = new char*[ argc ];
      for ( int i = 1; i < argc; ++i )
         argv[ i ] = PyString_AS_STRING( PyList_GET_ITEM( argl, i ) );
      argv[ 0 ] = Py_GetProgramName();

      gApplication = new PyROOTApplication( "PyROOT", &argc, argv );
   }

   if ( !gProgName )               // should be set by TApplication
      gSystem->SetProgname( Py_GetProgramName() );

// bind ROOT globals
   addToScope( "gROOT", gROOT, gROOT->IsA() );
   addToScope( "gSystem", gSystem, gSystem->IsA() );
   addToScope( "gInterpreter", gInterpreter, gInterpreter->IsA() );

// memory management
   gROOT->GetListOfCleanups()->Add( new MemoryRegulator() );

// python side class construction, managed by ROOT
   gROOT->AddClassGenerator( new TPyClassGenerator() );
}

//____________________________________________________________________________
int PyROOT::BuildRootClassDict( TClass* klass, PyObject* pyclass ) {
   assert( klass != 0 );

   std::string clName = klass->GetName();
   bool isNamespace = klass->Property() & G__BIT_ISNAMESPACE;
   bool hasConstructor = false;

// load all public methods and data members
   typedef std::vector< PyCallable* > Callables_t;
   typedef std::map< std::string, Callables_t > CallableCache_t;
   CallableCache_t cache;

   TIter nextmethod( klass->GetListOfMethods() );
   while ( TMethod* method = (TMethod*)nextmethod() ) {
   // retrieve method name
      std::string mtName = method->GetName();

   // filter C++ destructors
      if ( mtName[0] == '~' )
         continue;

   // translate operators
      if ( 8 < mtName.size() && mtName.substr( 0, 8 ) == "operator" ) {
         std::string op = mtName.substr( 8, std::string::npos );

      // filter memory operators
         if ( op == "=" || op == " new" || op == " new[]" ||
              op == " delete" || op == " delete[]" )
            continue;

      // map C++ operator to python equivalent
         Utility::TC2POperatorMapping_t::iterator pop = Utility::gC2POperatorMapping.find( op );
         if ( pop != Utility::gC2POperatorMapping.end() )
            mtName = pop->second;
      }

   // decide on method type: member or static (which includes globals)
      bool isStatic = isNamespace || ( method->Property() & G__BIT_ISSTATIC );

   // public methods are normally visible, private methods are mangled python-wise
   // note the overload implications which are name based
      if ( !( method->Property() & kIsPublic ) )
         if ( mtName == clName )             // don't expose private ctors
            continue;
         else                                // mangle private methods
            mtName = "__" + clName + "__" + mtName;

   // construct the holder
      PyCallable* pycall = 0;
      if ( isStatic == true )                // class method
         pycall = new ClassMethodHolder( klass, method );
      else if ( mtName == clName ) {         // constructor
         pycall = new ConstructorHolder( klass, method );
         mtName = "__init__";
         hasConstructor = true;
      } else                                 // member function
         pycall = new MethodHolder( klass, method );

   // lookup method dispatcher and store method
      Callables_t& md = (*(cache.insert(
         std::make_pair( mtName, Callables_t() ) ).first)).second;
      md.push_back( pycall );
   }

// special case if there's no constructor defined
   if ( ! hasConstructor )
      cache[ "__init__" ].push_back( new ConstructorHolder( klass, 0 ) );

// add the methods to the class dictionary
   for ( CallableCache_t::iterator imd = cache.begin(); imd != cache.end(); ++imd ) {
      MethodProxy* method = MethodProxy_New( imd->first, imd->second );
      PyObject_SetAttrString(
         pyclass, const_cast< char* >( method->GetName().c_str() ), (PyObject*)method );
      Py_DECREF( method );
   }

// collect data members
   TIter nextmember( klass->GetListOfDataMembers() );
   while ( TDataMember* mb = (TDataMember*)nextmember() ) {
   // allow only public members
      if ( !( mb->Property() & kIsPublic ) )
         continue;

   // enums
      if ( mb->IsEnum() ) {
         long offset = 0;
         G__DataMemberInfo dmi = klass->GetClassInfo()->GetDataMember( mb->GetName(), &offset );
         PyObject* val = PyInt_FromLong( *((int*)((G__var_array*)dmi.Handle())->p[dmi.Index()]) );
         PyObject_SetAttrString( pyclass, const_cast< char* >( mb->GetName() ), val );
         Py_DECREF( val );
      }

   // properties
      else {
         PropertyProxy* property = PropertyProxy_New( mb );
         PyObject_SetAttrString(
            pyclass, const_cast< char* >( property->GetName().c_str() ), (PyObject*)property );
         Py_DECREF( property );
      }
   }

// all ok, done
   return 0;
}

//____________________________________________________________________________
PyObject* PyROOT::BuildRootClassBases( TClass* klass )
{
   TList* allbases = klass->GetListOfBases();

   std::vector< std::string >::size_type nbases = 0;
   if ( allbases != 0 )
      nbases = allbases->GetSize();

// collect bases while removing duplicates
   std::vector< std::string > uqb;
   uqb.reserve( nbases );

   TIter nextbase( allbases );
   while ( TBaseClass* base = (TBaseClass*)nextbase() ) {
      std::string name = base->GetName();
      if ( std::find( uqb.begin(), uqb.end(), name ) == uqb.end() ) {
         uqb.push_back( name );
      }
   }

// allocate a tuple for the base classes, special case for first base
   nbases = uqb.size();

   PyObject* pybases = PyTuple_New( nbases ? nbases : 1 );
   if ( ! pybases )
      return 0;

// build all the bases
   if ( nbases == 0 ) {
      Py_INCREF( &ObjectProxy_Type );
      PyTuple_SET_ITEM( pybases, 0, (PyObject*)&ObjectProxy_Type );
   }
   else {
      for ( std::vector< std::string >::size_type ibase = 0; ibase < nbases; ++ibase ) {
         PyObject* pyclass = MakeRootClassFromString( uqb[ ibase ] );
         if ( ! pyclass ) {
            Py_DECREF( pybases );
            return 0;
         }

         PyTuple_SET_ITEM( pybases, ibase, pyclass );
      }
   }

   return pybases;
}

//____________________________________________________________________________
PyObject* PyROOT::MakeRootClass( PyObject*, PyObject* args )
{
   std::string cname = PyString_AsString( PyTuple_GetItem( args, 0 ) );

   if ( PyErr_Occurred() )
      return 0;

   return MakeRootClassFromString( cname );
}

//____________________________________________________________________________
PyObject* PyROOT::MakeRootClassFromString( const std::string& name )
{
// retrieve ROOT class (this verifies name)
   TClass* klass = gROOT->GetClass( name.c_str() );
   if ( klass == 0 ) {
      PyErr_Format( PyExc_TypeError, "requested class \'%s\' does not exist", name.c_str() );
      return 0;
   }

   PyObject* pyname = PyString_FromString( const_cast< char* >( name.c_str() ) );

// first try to retrieve the class representation from the ROOT module
   PyObject* pyclass = PyObject_GetAttr( gRootModule, pyname );

// build if the class does not yet exist
   if ( ! pyclass ) {
   // ignore error generated from the failed lookup
      PyErr_Clear();

   // construct the base classes
      PyObject* pybases = BuildRootClassBases( klass );
      if ( pybases != 0 ) {
      // create a fresh Python class, given bases, name, and empty dictionary
         PyObject* args = Py_BuildValue( const_cast< char* >( "OO{}" ), pyname, pybases );
         pyclass = PyType_Type.tp_new( &PyType_Type, args, NULL );

         Py_DECREF( args );
         Py_DECREF( pybases );
      }

   // fill the dictionary, if successful
      if ( pyclass != 0 ) {
         if ( BuildRootClassDict( klass, pyclass ) != 0 ) {
         // something failed in building the dictionary
            Py_DECREF( pyclass );
            pyclass = 0;
         }
         else {
            Py_INCREF( pyclass );            // PyModule_AddObject steals reference
            PyModule_AddObject( gRootModule, const_cast< char* >( name.c_str() ), pyclass );
         }
      }
   }

   Py_DECREF( pyname );

// add python-style features
   if ( ! Pythonize( pyclass, name ) ) {
      Py_XDECREF( pyclass );
      pyclass = 0;
   }

// all done
   return pyclass;
}

//____________________________________________________________________________
PyObject* PyROOT::GetRootGlobal( PyObject*, PyObject* args )
{
// get the requested name
   std::string ename = PyString_AsString( PyTuple_GetItem( args, 0 ) );

   if ( PyErr_Occurred() )
      return 0;

   return GetRootGlobalFromString( ename );
}

//____________________________________________________________________________
PyObject* PyROOT::GetRootGlobalFromString( const std::string& name )
{
// loop over globals to find this enum
   TIter nextGlobal( gROOT->GetListOfGlobals( kTRUE ) );
   while ( TGlobal* gb = (TGlobal*)nextGlobal() ) {
      if ( gb->GetName() == name && gb->GetAddress() ) {

         if ( G__TypeInfo( gb->GetTypeName() ).Property() & G__BIT_ISENUM )
         // enum, deref and return as long
            return PyInt_FromLong( *((int*)gb->GetAddress()) );

         else
         // TGlobal, attempt to get the actual class and cast as appropriate
            return BindRootGlobal( gb );
      }
   }

// still here ... try functions
   TFunction* func = gROOT->GetGlobalFunction( name.c_str(), 0, kTRUE );
   if ( func )
      return BindRootObject( func, TFunction::Class() );

// nothing found
   Py_INCREF( Py_None );
   return Py_None;
}

//____________________________________________________________________________
PyObject* PyROOT::BindRootObjectNoCast( void* address, TClass* klass, bool isRef ) {
// only known or knowable objects will be bound (null object is ok)
   if ( ! klass ) {
      PyErr_SetString( PyExc_TypeError, "attempt to bind ROOT object w/o class" );
      return 0;
   }

// retriev python class
   PyObject* pyclass = MakeRootClassFromString( klass->GetName() );
   if ( ! pyclass )
      return 0;                    // error set in make ROOT class

// instantiate an object of this class
   PyObject* args = PyTuple_New(0);
   ObjectProxy* pyobj =
      (ObjectProxy*)((PyTypeObject*)pyclass)->tp_new( (PyTypeObject*)pyclass, args, NULL );
   Py_DECREF( args );
   Py_DECREF( pyclass );

// bind, register and return if successful
   if ( pyobj != 0 ) {
   // fill proxy values
      if ( ! isRef )
         pyobj->Set( address, klass );
      else
         pyobj->Set( (void**)address, klass );
   }

// successful completion
   return (PyObject*)pyobj;
}

//____________________________________________________________________________
PyObject* PyROOT::BindRootObject( void* address, TClass* klass, bool isRef )
{
// for safety (should return "null pointer object")
   if ( ! address ) {
      Py_INCREF( Py_None );
      return Py_None;
   }

// only known or knowable objects will be bound
   if ( ! klass ) {
      PyErr_SetString( PyExc_TypeError, "attempt to bind ROOT object w/o class" );
      return 0;
   }

// upgrade to real class for object returns
   if ( ! isRef ) {
      TClass* clActual = klass->GetActualClass( (void*)address );
      if ( clActual && klass != clActual ) {
         int offset = clActual->GetBaseClassOffset( klass );
         (long&)address -= offset;
         klass = clActual;
      }
   }

// obtain pointer to TObject base class (if possible) for memory mgmt
   TObject* object = (TObject*)( isRef ? *((void**)address) : address );
   if ( klass != TObject::Class() )
      object = (TObject*) klass->DynamicCast( TObject::Class(), object );

   if ( ! isRef ) {
   // use the old reference if the object already exists
      PyObject* oldPyObject = MemoryRegulator::RetrieveObject( object );
      if ( oldPyObject )
         return oldPyObject;
   }

// actual binding
   ObjectProxy* pyobj = (ObjectProxy*)BindRootObjectNoCast( address, klass, isRef );

// memory management
   MemoryRegulator::RegisterObject( pyobj, object );

// completion (returned object may be zero w/ a python exception set)
   return (PyObject*)pyobj;
}

//____________________________________________________________________________
PyObject* PyROOT::BindRootGlobal( TGlobal* gbl )
{
// should return "null pointer" ... for now, None will do
   if ( ! gbl ) {
      Py_INCREF( Py_None );
      return Py_None;
   }

// determine type and cast as appropriate
   TClass* klass = gROOT->GetClass( gbl->GetTypeName() );
   if ( ! klass ) {
      switch ( Utility::effectiveType( gbl->GetFullTypeName() ) ) {
      case Utility::kBool:
         return PyInt_FromLong( (long) *(Bool_t*)gbl->GetAddress() );
      case Utility::kChar:
         return PyInt_FromLong( (long) *(Char_t*)gbl->GetAddress() );
      case Utility::kShort:
      case Utility::kEnum:
      case Utility::kInt:
         return PyInt_FromLong( (long) *(Int_t*)gbl->GetAddress() );
      case Utility::kLong:
         return PyLong_FromLong( (long) *(Long_t*)gbl->GetAddress() );
      case Utility::kFloat:
         return PyFloat_FromDouble( (double) *(Float_t*)gbl->GetAddress() );
      case Utility::kDouble:
         return PyFloat_FromDouble( (double) *(Double_t*)gbl->GetAddress() );
      default:
         klass = TGlobal::Class();
      }
   }

   if ( Utility::isPointer( gbl->GetFullTypeName() ) )
      return BindRootObject( (void*)gbl->GetAddress(), klass, true );

   return BindRootObject( (void*)gbl->GetAddress(), klass );
}

