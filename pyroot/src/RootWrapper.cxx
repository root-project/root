// @(#)root/pyroot:$Name:  $:$Id: RootWrapper.cxx,v 1.36 2005/09/09 05:19:10 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "PyRootType.h"
#include "ObjectProxy.h"
#include "MethodProxy.h"
#include "PropertyProxy.h"
#include "RootWrapper.h"
#include "Pythonize.h"
#include "MethodHolder.h"
#include "ConstructorHolder.h"
#include "ClassMethodHolder.h"
#include "TSetItemHolder.h"
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
#include "TSeqCollection.h"
#include "DllImport.h"

// CINT
#include "Api.h"

// Standard
#include <assert.h>
#include <map>
#include <set>
#include <string>
#include <algorithm>
#include <vector>


//- data _______________________________________________________________________
R__EXTERN PyObject* gRootModule;


//- helpers --------------------------------------------------------------------
namespace {

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

   inline void AddToScope( const char* label, TObject* obj, TClass* klass )
   {
      PyModule_AddObject( gRootModule, const_cast< char* >( label ),
         PyROOT::BindRootObject( obj, klass ) );
   }

   std::set< std::string > gSTLTypes;
   struct InitSTLTypes_t {
      InitSTLTypes_t()
      {
         const char* stlTypes[] = { "complex", "exception",
            "deque", "list", "queue", "stack", /* "vector", : preloaded */
            "map", "multimap", "set", "multiset" };
         std::string nss = "std::";
         for ( int i = 0; i < int(sizeof(stlTypes)/sizeof(stlTypes[0])); ++i ) {
            gSTLTypes.insert( stlTypes[ i ] );
            gSTLTypes.insert( nss + stlTypes[ i ] );
         }
      }
   } initSTLTypes_;

   void LoadDictionaryForSTLType( const std::string& tname )
   {
   // if name is of a known STL class, load the appropriate CINT dll(s)
       std::string sub = tname.substr( 0, tname.find( "<" ) );
       if ( gSTLTypes.find( sub ) != gSTLTypes.end() ) {
          if ( sub.substr( 0, 5 ) == "std::" )
             sub = sub.substr( 5, std::string::npos );

          if ( 0 <= G__loadfile( (sub+".dll").c_str() ) ) {
          // special case for map and multimap, which are spread over 2 files
             if ( sub == "map" || sub == "multimap" ) {
                G__loadfile( (sub+"2.dll").c_str() );
             }

          // success; prevent second attempt to load by erasing name
             gSTLTypes.erase( sub );
             gSTLTypes.erase( "std::" + sub );

          } else {
             PyErr_Warn( PyExc_RuntimeWarning,
                const_cast< char* >( ( "could not load dict lib for " + sub ).c_str() ) );
          }
       }
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

      gApplication = new TPyROOTApplication( "PyROOT", &argc, argv );

   // CINT message callback (only if loaded from python, i.e. !gApplication)
      G__set_errmsgcallback( (void*)&PyROOT::Utility::ErrMsgCallback );
   }

   if ( !gProgName )               // should be set by TApplication
      gSystem->SetProgname( Py_GetProgramName() );

// bind ROOT globals
   AddToScope( "gROOT", gROOT, gROOT->IsA() );
   AddToScope( "gSystem", gSystem, gSystem->IsA() );
   AddToScope( "gInterpreter", gInterpreter, gInterpreter->IsA() );

// memory management
   gROOT->GetListOfCleanups()->Add( new TMemoryRegulator );

// python side class construction, managed by ROOT
   gROOT->AddClassGenerator( new TPyClassGenerator );
}

//____________________________________________________________________________
int PyROOT::BuildRootClassDict( TClass* klass, PyObject* pyclass ) {
   assert( klass != 0 );

// in case of namespaces, get the unscoped name
   G__ClassInfo* clInfo = klass->GetClassInfo();
   std::string clName = clInfo ? clInfo->Name() : klass->GetName();

// some properties that'll affect building the dictionary
   Bool_t isNamespace = klass->Property() & G__BIT_ISNAMESPACE;
   Bool_t hasConstructor = kFALSE;

// special cases for C++ facilities that have no python equivalent
   TMethod* assign = 0;

// load all public methods and data members
   typedef std::vector< PyCallable* > Callables_t;
   typedef std::map< std::string, Callables_t > CallableCache_t;
   CallableCache_t cache;

   TIter nextmethod( klass->GetListOfMethods() );
   while ( TMethod* method = (TMethod*)nextmethod() ) {
   // special case tracker
      Bool_t setupSetItem = kFALSE;

   // retrieve method name
      std::string mtName = method->GetName();

   // filter empty names (happens for namespaces, is bug?)
      if ( mtName == "" )
         continue;

   // filter C++ destructors
      if ( mtName[0] == '~' )
         continue;

   // translate operators
      if ( 8 < mtName.size() && mtName.substr( 0, 8 ) == "operator" ) {
         std::string op = mtName.substr( 8, std::string::npos );

      // filter assignment operator for later use
         if ( op == "=" ) {
            assign = method;
            continue;
         }

      // map C++ operator to python equivalent, or made up name if no equivalent exists
         Utility::TC2POperatorMapping_t::iterator pop = Utility::gC2POperatorMapping.find( op );
         if ( pop != Utility::gC2POperatorMapping.end() ) {
            mtName = pop->second;
         } else if ( op == "[]" ) {
            mtName = "__getitem__";

         // operator[] returning a reference type will be used for __setitem__
            std::string cpd = Utility::Compound( method->GetReturnTypeName() );
            if ( cpd[ cpd.size() - 1 ] == '&' )
               setupSetItem = kTRUE;
         } else if ( op == "*" ) {
            if ( method->GetNargs() == 0 )   // dereference
               mtName = "__deref__";
            else                             // multiplier (is python equivalent)
               mtName = "__mul__";
         } else if ( op == "++" ) {
            if ( method->GetNargs() == 0 )   // prefix increment
               mtName = "__preinc__"; 
            else                             // postfix increment
               mtName = "__postinc__";
         } else if ( op == "--" ) {
            if ( method->GetNargs() == 0 )   // prefix decrement
               mtName = "__predec__";
            else                             // postfix decrement
               mtName = "__postdec__";
         } else if ( op == "->" ) {          // class member access
             mtName = "__follow__";
         } else {
            continue;                        // operator not handled (new, delete, etc.)
         }
      }

   // decide on method type: member or static (which includes globals)
      Bool_t isStatic = isNamespace || ( method->Property() & G__BIT_ISSTATIC );

   // public methods are normally visible, private methods are mangled python-wise
   // note the overload implications which are name based, and note that rootcint
   // does not create the interface methods for private/protected methods ...
      if ( !( method->Property() & kIsPublic ) )
         if ( mtName == clName )             // don't expose private ctors
            continue;
         else                                // mangle private methods
            mtName = "_" + clName + "__" + mtName;

   // construct the holder
      PyCallable* pycall = 0;
      if ( isStatic == kTRUE )               // class method
         pycall = new TClassMethodHolder( klass, method );
      else if ( mtName == clName ) {         // constructor
         pycall = new TConstructorHolder( klass, method );
         mtName = "__init__";
         hasConstructor = kTRUE;
      } else                                 // member function
         pycall = new TMethodHolder( klass, method );

   // lookup method dispatcher and store method
      Callables_t& md = (*(cache.insert(
         std::make_pair( mtName, Callables_t() ) ).first)).second;
      md.push_back( pycall );

   // special case for operator[] that returns by ref, use for getitem and setitem
      if ( setupSetItem ) {
         Callables_t& setitem = (*(cache.insert(
            std::make_pair( "__setitem__", Callables_t() ) ).first)).second;
         setitem.push_back( new TSetItemHolder( klass, method ) );
      }
   }

// add a pseudo-default ctor, if none defined
   if ( ! isNamespace && ! hasConstructor )
      cache[ "__init__" ].push_back( new TConstructorHolder( klass, 0 ) );

//   if ( assign != 0 )
//      std::cout << "found assignment operator for: " << klass->GetName() << std::endl;

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
         Long_t offset = 0;
         G__DataMemberInfo dmi = klass->GetClassInfo()->GetDataMember( mb->GetName(), &offset );
         PyObject* val = PyInt_FromLong( *((int*)((G__var_array*)dmi.Handle())->p[dmi.Index()]) );
         PyObject_SetAttrString( pyclass, const_cast< char* >( mb->GetName() ), val );
         Py_DECREF( val );
      }

   // properties
      else {
         PropertyProxy* property = PropertyProxy_New< TDataMember >( mb );
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
   } else {
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
PyObject* PyROOT::MakeRootClassFromString( std::string name, PyObject* scope )
{
// force building of the class if a scope is specified
   Bool_t force = scope != 0;

// determine scope name, if a python scope has been given
   std::string scName = "";
   if ( scope ) {
      PyObject* pyscope = PyObject_GetAttrString( scope, const_cast< char* >( "__name__" ) );
      if ( ! pyscope ) {
         PyErr_Format( PyExc_SystemError, "given scope has no name for %s", name.c_str() );
         return 0;
      }

   // should be a string
      scName = PyString_AsString( pyscope );
      Py_DECREF( pyscope );
      if ( PyErr_Occurred() )
         return 0;
   }

// retrieve ROOT class (this verifies name)
   const std::string& lookup = scope ? (scName+"::"+name) : name;
   TClass* klass = gROOT->GetClass( lookup.c_str() );
   if ( ! klass || ( klass != 0 && klass->GetNmethods() == 0 ) ) {
   // removal is required or the dictionary can't be updated properly
      if ( klass != 0 ) gROOT->RemoveClass( klass );

   // special action for STL classes to enforce loading dict lib
      LoadDictionaryForSTLType( name );

   // lookup again, if this was an STL class, we (may) now have a full dictionary
      klass = gROOT->GetClass( lookup.c_str() );
   }

   if ( ! klass ) {
   // in case a "naked" templated class is requested, return callable proxy for instantiations
      if ( G__defined_templateclass( const_cast< char* >( lookup.c_str() ) ) ) {
         PyObject* pytcl = PyObject_GetAttrString( gRootModule, const_cast< char* >( "Template" ) );
         PyObject* pytemplate = PyObject_CallFunction( pytcl, "s", lookup.c_str() );
         Py_DECREF( pytcl );
         return pytemplate;
      }

      PyErr_Format( PyExc_TypeError, "requested class \'%s\' does not exist", lookup.c_str() );
      return 0;
   }

// locate the scope for building the class if not specified
   if ( ! scope ) {
   // cut template part, if present
      std::string genName = name.substr( 0, name.find( "<" ) );

   // drop class name (i.e. only collect actual scopes here)
      std::string::size_type last = genName.rfind( "::" );
      if ( last != 0 && last != std::string::npos ) {
         scName = genName.substr( 0, last );
         name = name.substr( last + 2, std::string::npos );
      }

      if ( scName != "" ) {
         std::string::size_type pos = 0;
         do {
            last = scName.find( "::", pos );
            std::string part = scName.substr( pos, last == std::string::npos ? last : last - pos );
            PyObject* next = PyObject_GetAttrString(
               scope ? scope : gRootModule, const_cast< char* >( part.c_str() ) );

            if ( ! next ) {           // lookup failed, try to create it
               PyErr_Clear();
               next = MakeRootClassFromString( part, scope );
            }
            Py_XDECREF( scope );

            if ( ! next )             // create failed, give up
               return 0;

            scope = next;
            pos = last + 2;
         } while ( last != std::string::npos );
      }
   } else
     Py_INCREF( scope );

// use global scope if no inner scope found
   if ( ! scope ) {
      scope = gRootModule;
      Py_INCREF( scope );
   }

// use actual class name for binding
   G__ClassInfo* clInfo = klass->GetClassInfo();
   const std::string actual = clInfo ? clInfo->Name() : klass->GetName();

// first try to retrieve an existing class representation
   PyObject* pyactual = PyString_FromString( actual.c_str() );
   PyObject* pyclass = 0;
   if ( force == kFALSE )
      pyclass = PyObject_GetAttr( scope, pyactual );

// build if the class does not yet exist
   if ( ! pyclass ) {
   // ignore error generated from the failed lookup
      PyErr_Clear();

   // construct the base classes
      PyObject* pybases = BuildRootClassBases( klass );
      if ( pybases != 0 ) {
      // create a fresh Python class, given bases, name, and empty dictionary
         PyObject* pytrue = PyString_FromString( klass->GetName() );
         PyObject* args = Py_BuildValue( const_cast< char* >( "OO{}" ), pytrue, pybases );
         pyclass = PyType_Type.tp_new( &PyRootType_Type, args, NULL );

         Py_DECREF( args );
         Py_DECREF( pytrue );
         Py_DECREF( pybases );
      }

   // fill the dictionary, if successful
      if ( pyclass != 0 ) {
         if ( BuildRootClassDict( klass, pyclass ) != 0 ) {
         // something failed in building the dictionary
            Py_DECREF( pyclass );
            pyclass = 0;
         } else
            PyObject_SetAttr( scope, pyactual, pyclass );
      }
   }

   if ( pyclass && name != actual )     // class exists, but is typedef-ed: simply map reference
      PyObject_SetAttrString( scope, const_cast< char* >( name.c_str() ), pyclass );

   Py_DECREF( pyactual );
   Py_DECREF( scope );

// add python-style features
   if ( ! Pythonize( pyclass, klass->GetName() ) ) {
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
   if ( func ) {
      PyObject* pyobject = BindRootObject( new TFunction( *func ), TFunction::Class() );
      ((ObjectProxy*)pyobject)->fFlags |= ObjectProxy::kIsOwner;
      return pyobject;
   }

// nothing found
   PyErr_Format( PyExc_LookupError, "no such global: %s", name.c_str() );
   return 0;
}

//____________________________________________________________________________
PyObject* PyROOT::BindRootObjectNoCast( void* address, TClass* klass, Bool_t isRef ) {
// only known or knowable objects will be bound (null object is ok)
   if ( ! klass ) {
      PyErr_SetString( PyExc_TypeError, "attempt to bind ROOT object w/o class" );
      return 0;
   }

// retrieve python class
   PyObject* pyclass = MakeRootClassFromString( klass->GetName() );
   if ( ! pyclass )
      return 0;                    // error has been set in MakeRootClass

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
PyObject* PyROOT::BindRootObject( void* address, TClass* klass, Bool_t isRef )
{
// for safety (None can't be used as NULL pointer)
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
      TClass* clActual = klass->GetActualClass( address );
      if ( clActual && klass != clActual ) {
       // root/meta base class offset fails in the case of virtual inheritance
       //   Long_t offset = clActual->GetBaseClassOffset( klass );
         Long_t offset = G__isanybase(
            klass->GetClassInfo()->Tagnum(), clActual->GetClassInfo()->Tagnum(), (Long_t)address );
         (Long_t&)address -= offset;
         klass = clActual;
      }
   }

// obtain pointer to TObject base class (if possible) for memory mgmt
   TObject* object = (TObject*)( isRef ? *((void**)address) : address );
   if ( klass != TObject::Class() )
      object = (TObject*) klass->DynamicCast( TObject::Class(), object );

   if ( ! isRef ) {
   // use the old reference if the object already exists
      PyObject* oldPyObject = TMemoryRegulator::RetrieveObject( object );
      if ( oldPyObject )
         return oldPyObject;
   }

// actual binding
   ObjectProxy* pyobj = (ObjectProxy*)BindRootObjectNoCast( address, klass, isRef );

// memory management
   TMemoryRegulator::RegisterObject( pyobj, object );

// completion (returned object may be zero w/ a python exception set)
   return (PyObject*)pyobj;
}

//____________________________________________________________________________
PyObject* PyROOT::BindRootGlobal( TGlobal* gbl )
{
// gbl == 0 means global does not exist (rather than gbl is NULL pointer)
   if ( ! gbl ) {
      Py_INCREF( Py_None );
      return Py_None;
   }

// determine type and cast as appropriate
   TClass* klass = gROOT->GetClass( gbl->GetTypeName() );
   if ( klass != 0 ) {
      if ( Utility::Compound( gbl->GetFullTypeName() ) != "" )
         return BindRootObject( (void*)gbl->GetAddress(), klass, kTRUE );

      return BindRootObject( (void*)gbl->GetAddress(), klass );
   }

// for built-in types, to ensure setability
   return (PyObject*)PropertyProxy_New< TGlobal >( gbl );
}
