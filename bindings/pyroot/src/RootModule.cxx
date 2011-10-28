// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "PyStrings.h"
#include "PyRootType.h"
#include "ObjectProxy.h"
#include "MethodProxy.h"
#include "TemplateProxy.h"
#include "PropertyProxy.h"
#include "PyBufferFactory.h"
#include "TCustomPyTypes.h"
#include "RootWrapper.h"
#include "Utility.h"
#include "Adapters.h"

#ifdef PYROOT_USE_REFLEX
#include "TRflxCallback.h"
#endif

// ROOT
#include "TROOT.h"
#include "TClass.h"
#include "TObject.h"

#include "TBufferFile.h"

// Standard
#include <string>


//- data -----------------------------------------------------------------------
PyObject* gRootModule = 0;


//- private helpers ------------------------------------------------------------
namespace {

   using namespace PyROOT;

//____________________________________________________________________________
   PyObject* LookupRootEntity( PyObject* pyname, PyObject* args )
   {
      const char* cname = 0;
      if ( pyname && PyROOT_PyUnicode_CheckExact( pyname ) )
         cname = PyROOT_PyUnicode_AsString( pyname );
      else if ( ! ( args && PyArg_ParseTuple( args, const_cast< char* >( "s" ), &cname ) ) )
         return 0;

      std::string name = cname;

   // block search for privates
      if ( name.size() <= 2 || name.substr( 0, 2 ) != "__" ) {
      // 1st attempt: look in myself
         PyObject* attr = PyObject_GetAttrString( gRootModule, const_cast< char* >( cname ) );
         if ( attr != 0 )
            return attr;

      // 2nd attempt: construct name as a class
         PyErr_Clear();
         attr = MakeRootClassFromString< TScopeAdapter, TBaseAdapter, TMemberAdapter >( name );
         if ( attr != 0 )
            return attr;

      // 3rd attempt: lookup name as global variable
         PyErr_Clear();
         attr = GetRootGlobalFromString( name );
         if ( attr != 0 )
            return attr;

      // 4th attempt: find existing object (e.g. from file)
         PyErr_Clear();
         TObject* object = gROOT->FindObject( name.c_str() );
         if ( object != 0 )
            return BindRootObject( object, object->IsA() );
      }

   // still here? raise attribute error
      PyErr_Format( PyExc_AttributeError, "%s", name.c_str() );
      return 0;
   }

//____________________________________________________________________________
   PyDictEntry* RootLookDictString( PyDictObject* mp, PyObject* key, Long_t hash )
   {
   // first search dictionary itself
      PyDictEntry* ep = (*gDictLookupOrg)( mp, key, hash );
      if ( ! ep || ep->me_value != 0 || gDictLookupActive )
         return ep;

   // filter for builtins
      if ( PyDict_GetItem( PyEval_GetBuiltins(), key ) != 0 ) {
         return ep;
      }

   // all failed, start calling into ROOT
      gDictLookupActive = kTRUE;

   // attempt to get ROOT enum/global/class
      PyObject* val = LookupRootEntity( key, 0 );

      if ( val != 0 ) {
      // success ...
         if ( PropertyProxy_Check( val ) ) {
         // pretend something was actually found, but don't add to dictionary
            Py_INCREF( key );
            ep->me_key   = key;
            ep->me_hash  = hash;
            ep->me_value = Py_TYPE(val)->tp_descr_get( val, NULL, NULL );
         } else {
         // add reference to ROOT entity in the given dictionary
            ((DictLookup_t&)mp->ma_lookup) = gDictLookupOrg;     // prevent recursion
            if ( PyDict_SetItem( (PyObject*)mp, key, val ) == 0 ) {
               ep = (*gDictLookupOrg)( mp, key, hash );
            } else {
               ep->me_key   = 0;
               ep->me_value = 0;
            }
            ((DictLookup_t&)mp->ma_lookup) = RootLookDictString; // restore
         }

      // done with val
         Py_DECREF( val );
      } else
         PyErr_Clear();

   // stopped calling into ROOT
      gDictLookupActive = kFALSE;

      return ep;
   }

//____________________________________________________________________________
   PyObject* SetRootLazyLookup( PyObject*, PyObject* args )
   {
   // Modify the given dictionary to install the lookup function that also
   // tries the ROOT namespace before failing. Called on a module's dictionary,
   // this allows for lazy lookups.
      PyDictObject* dict = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "O!" ), &PyDict_Type, &dict ) )
         return 0;

      ((DictLookup_t&)dict->ma_lookup) = RootLookDictString;

      Py_INCREF( Py_None );
      return Py_None;
   }

//____________________________________________________________________________
   PyObject* MakeRootTemplateClass( PyObject*, PyObject* args )
   {
   // Create a binding for a templated class instantiation.

   // args is class name + template arguments; build full instantiation
      Py_ssize_t nArgs = PyTuple_GET_SIZE( args );
      if ( nArgs < 2 ) {
         PyErr_Format( PyExc_TypeError, "too few arguments for template instantiation" );
         return 0;
      }

   // copy initial argument (no check, comes from internal class)
      PyObject* pyname = PyROOT_PyUnicode_FromString(
         PyROOT_PyUnicode_AsString( PyTuple_GET_ITEM( args, 0 ) ) );

   // build "< type, type, ... >" part of class name (modifies pyname)
      if ( ! Utility::BuildTemplateName( pyname, args, 1 ) ) {
         Py_DECREF( pyname );
         return 0;
      }

      std::string name = PyROOT_PyUnicode_AsString( pyname );
      Py_DECREF( pyname );

      return MakeRootClassFromString< TScopeAdapter, TBaseAdapter, TMemberAdapter >( name );
   }

//____________________________________________________________________________
   void* GetObjectProxyAddress( PyObject*, PyObject* args )
   {
   // Helper to get the address (address-of-address) of various object proxy types.
      ObjectProxy* pyobj = 0;
      PyObject* pyname = 0;
      if ( PyArg_ParseTuple( args, const_cast< char* >( "O|O!" ), &pyobj,
             &PyROOT_PyUnicode_Type, &pyname ) &&
           ObjectProxy_Check( pyobj ) && pyobj->fObject ) {

         if ( pyname != 0 ) {
         // locate property proxy for offset info
            PropertyProxy* pyprop = 0;

            PyObject* pyclass = PyObject_GetAttr( (PyObject*)pyobj, PyStrings::gClass );

            if ( pyclass ) {
               PyObject* dict = PyObject_GetAttr( pyclass, PyStrings::gDict );
               pyprop = (PropertyProxy*)PyObject_GetItem( dict, pyname );
               Py_DECREF( dict );
            }
            Py_XDECREF( pyclass );

            if ( PropertyProxy_Check( pyprop ) ) {
            // this is an address of a value (i.e. &myobj->prop)
               void* addr = (void*)pyprop->GetAddress( pyobj ); 
               Py_DECREF( pyprop );
               return addr;
            }

            Py_XDECREF( pyprop );

            PyErr_Format( PyExc_TypeError,
               "%s is not a valid data member", PyROOT_PyUnicode_AsString( pyname ) );
            return 0;
         }

      // this is an address of an address (i.e. &myobj, with myobj of type MyObj*)
         return (void*)&pyobj->fObject;
      }

      PyErr_SetString( PyExc_ValueError, "invalid argument for AddressOf()" );
      return 0;
   }

   PyObject* AddressOf( PyObject* dummy, PyObject* args )
   {
   // Return object proxy address as an indexable buffer.
      void* addr = GetObjectProxyAddress( dummy, args );
      if ( addr )
         return BufFac_t::Instance()->PyBuffer_FromMemory( (Long_t*)addr, 1 );

      return 0;
   }

   PyObject* AsCObject( PyObject* dummy, PyObject* args )
   {
   // Return object proxy as an opaque CObject.
      void* addr = GetObjectProxyAddress( dummy, args );
      if ( addr )
         return PyROOT_PyCapsule_New( (void*)(*(Long_t*)addr), NULL, NULL );

      return 0;
   }

//____________________________________________________________________________
   PyObject* BindObject_( void* addr, PyObject* pyname )
   {
   // Helper to factorize the common code between MakeNullPointer and BindObject.

      if ( ! PyROOT_PyUnicode_Check( pyname ) ) {     // name given as string
         PyObject* nattr = PyObject_GetAttr( pyname, PyStrings::gName );
         if ( nattr )                        // object is actually a class
            pyname = nattr;
         pyname = PyObject_Str( pyname );
         Py_XDECREF( nattr );
      } else {
         Py_INCREF( pyname );
      }

      TClass* klass = TClass::GetClass( PyROOT_PyUnicode_AsString( pyname ) );
      Py_DECREF( pyname );

      if ( ! klass ) {
         PyErr_SetString( PyExc_TypeError,
            "BindObject expects a valid class or class name as an argument" );
         return 0;
      }

      return BindRootObjectNoCast( addr, klass, kFALSE );
   }

//____________________________________________________________________________
   PyObject* BindObject( PyObject*, PyObject* args )
   {
   // From a long representing an address or a PyCapsule/CObject, bind to a class.
      Py_ssize_t argc = PyTuple_GET_SIZE( args );
      if ( argc != 2 ) {
         PyErr_Format( PyExc_TypeError,
           "BindObject takes exactly 2 argumenst ("PY_SSIZE_T_FORMAT" given)", argc );
         return 0;
      }

   // try to convert first argument: either PyCapsule/CObject or long integer
      PyObject* pyaddr = PyTuple_GET_ITEM( args, 0 );
      void* addr = PyROOT_PyCapsule_GetPointer( pyaddr, NULL );
      if ( PyErr_Occurred() ) {
         PyErr_Clear();

         addr = PyLong_AsVoidPtr( pyaddr );

         if ( PyErr_Occurred() ) {
            PyErr_Clear();
            PyErr_SetString( PyExc_TypeError,
               "BindObject requires a CObject or long integer as first argument" );
            return 0;
         }
      }

      return BindObject_( addr, PyTuple_GET_ITEM( args, 1 ) );
   }

//____________________________________________________________________________
   PyObject* MakeNullPointer( PyObject*, PyObject* args )
   {
   // Create an object of the given type point to NULL (historic note: this
   // function is older than BindObject(), which can be used instead).
      Py_ssize_t argc = PyTuple_GET_SIZE( args );
      if ( argc != 0 && argc != 1 ) {
         PyErr_Format( PyExc_TypeError,
            "MakeNullPointer takes at most 1 argument ("PY_SSIZE_T_FORMAT" given)", argc );
         return 0;
      }

   // no class given, use None as generic
      if ( argc == 0 ) {
         Py_INCREF( Py_None );
         return Py_None;
      }

      return BindObject_( 0, PyTuple_GET_ITEM( args, 0 ) );
   }

//____________________________________________________________________________
   PyObject* ObjectProxyExpand( PyObject*, PyObject* args )
   {
   // This method is a helper for (un)pickling of ObjectProxy instances.
      PyObject* pybuf = 0, *pyname = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "O!O!:__expand__" ),
               &PyBytes_Type, &pybuf, &PyBytes_Type, &pyname ) )
         return 0;

      const char* clname = PyBytes_AS_STRING(pyname);

   // TBuffer and its derived classes can't write themselves, but can be created
   // directly from the buffer, so handle them in a special case
      void* newObj = 0;
      if ( strcmp( clname, "TBufferFile" ) == 0 ) {
         TBufferFile* buf = new TBufferFile( TBuffer::kWrite );
         buf->WriteFastArray( PyBytes_AS_STRING(pybuf), PyBytes_GET_SIZE( pybuf ) );
         newObj = buf;
      } else {
      // use the PyString macro's to by-pass error checking; do not adopt the buffer,
      // as the local TBufferFile can go out of scope (there is no copying)
         TBufferFile buf( TBuffer::kRead,
            PyBytes_GET_SIZE( pybuf ), PyBytes_AS_STRING( pybuf ), kFALSE );
         newObj = buf.ReadObjectAny( 0 );
      }

      PyObject* result = BindRootObject( newObj, TClass::GetClass( clname ) );
      if ( result ) {
      // this object is to be owned by the interpreter, assuming that the call
      // originated from there
         ((ObjectProxy*)result)->HoldOn();
      }

      return result;
   }

//____________________________________________________________________________
   PyObject* SetMemoryPolicy( PyObject*, PyObject* args )
   {
   // Set the global memory policy, which affects object ownership when objects
   // are passed as function arguments.
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
   // Set the global signal policy, which determines whether a jmp address
   // should be saved to return to after a C++ segfault.
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
   // Set the ownership (True is python-owns) for the given object.
      ObjectProxy* pyobj = 0; PyObject* pykeep = 0;
      if ( ! PyArg_ParseTuple( args, const_cast< char* >( "O!O!" ),
                &ObjectProxy_Type, (void*)&pyobj, &PyInt_Type, &pykeep ) )
         return 0;

      (Bool_t)PyLong_AsLong( pykeep ) ? pyobj->HoldOn() : pyobj->Release();

      Py_INCREF( Py_None );
      return Py_None;
   }

} // unnamed namespace


//- data -----------------------------------------------------------------------
static PyMethodDef gPyROOTMethods[] = {
   { (char*) "MakeRootClass", (PyCFunction)PyROOT::MakeRootClass,
     METH_VARARGS, (char*) "PyROOT internal function" },
   { (char*) "GetRootGlobal", (PyCFunction)PyROOT::GetRootGlobal,
     METH_VARARGS, (char*) "PyROOT internal function" },
   { (char*) "LookupRootEntity", (PyCFunction)LookupRootEntity,
     METH_VARARGS, (char*) "PyROOT internal function" },
   { (char*) "SetRootLazyLookup", (PyCFunction)SetRootLazyLookup,
     METH_VARARGS, (char*) "PyROOT internal function" },
   { (char*) "MakeRootTemplateClass", (PyCFunction)MakeRootTemplateClass,
     METH_VARARGS, (char*) "PyROOT internal function" },
   { (char*) "_DestroyPyStrings", (PyCFunction)PyROOT::DestroyPyStrings,
     METH_NOARGS, (char*) "PyROOT internal function" },
   { (char*) "AddressOf", (PyCFunction)AddressOf,
     METH_VARARGS, (char*) "Retrieve address of held object in a buffer" },
   { (char*) "AsCObject", (PyCFunction)AsCObject,
     METH_VARARGS, (char*) "Retrieve held object in a CObject" },
   { (char*) "BindObject", (PyCFunction)BindObject,
     METH_VARARGS, (char*) "Create an object of given type, from given address" },
   { (char*) "MakeNullPointer", (PyCFunction)MakeNullPointer,
     METH_VARARGS, (char*) "Create a NULL pointer of the given type" },
   { (char*) "_ObjectProxy__expand__", (PyCFunction)ObjectProxyExpand,
     METH_VARARGS, (char*) "Helper method for pickling" },
   { (char*) "SetMemoryPolicy", (PyCFunction)SetMemoryPolicy,
     METH_VARARGS, (char*) "Determines object ownership model" },
   { (char*) "SetSignalPolicy", (PyCFunction)SetSignalPolicy,
     METH_VARARGS, (char*) "Trap signals in safe mode to prevent interpreter abort" },
   { (char*) "SetOwnership", (PyCFunction)SetOwnership,
     METH_VARARGS, (char*) "Modify held C++ object ownership" },
#ifdef PYROOT_USE_REFLEX
   { (char*) "EnableReflex", (PyCFunction)PyROOT::TRflxCallback::Enable,
     METH_NOARGS, (char*) "Enable PyReflex notification of new types from Reflex" },
   { (char*) "DisableReflex", (PyCFunction)PyROOT::TRflxCallback::Disable,
     METH_NOARGS, (char*) "Disable PyReflex notification of new types from Reflex" },
#endif
   { NULL, NULL, 0, NULL }
};


#if PY_VERSION_HEX >= 0x03000000
struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static int rootmodule_traverse( PyObject* m, visitproc visit, void* arg )
{
    Py_VISIT( GETSTATE( m )->error );
    return 0;
}

static int rootmodule_clear( PyObject* m )
{
    Py_CLEAR( GETSTATE( m )->error );
    return 0;
}


static struct PyModuleDef moduledef = {
   PyModuleDef_HEAD_INIT,
   "libPyROOT",
   NULL,
   sizeof(struct module_state),
   gPyROOTMethods,
   NULL,
   rootmodule_traverse,
   rootmodule_clear,
   NULL
};

//____________________________________________________________________________
#define PYROOT_INIT_ERROR return NULL
extern "C" PyObject* PyInit_libPyROOT()
#else
#define PYROOT_INIT_ERROR return
extern "C" void initlibPyROOT()
#endif
{
   using namespace PyROOT;

// load commonly used python strings
   if ( ! PyROOT::CreatePyStrings() )
      PYROOT_INIT_ERROR;

// prepare for lazyness
   PyObject* dict = PyDict_New();
   gDictLookupOrg = (DictLookup_t)((PyDictObject*)dict)->ma_lookup;
   Py_DECREF( dict );

// setup PyROOT
#if PY_VERSION_HEX >= 0x03000000
   gRootModule = PyModule_Create( &moduledef );
#else
   gRootModule = Py_InitModule( const_cast< char* >( "libPyROOT" ), gPyROOTMethods );
#endif
   if ( ! gRootModule )
      PYROOT_INIT_ERROR;

// keep gRootModule, but do not increase its reference count even as it is borrowed,
// or a self-referencing cycle would be created

// inject meta type
   if ( ! Utility::InitProxy( gRootModule, &PyRootType_Type, "PyRootType" ) )
      PYROOT_INIT_ERROR;

// inject object proxy type
   if ( ! Utility::InitProxy( gRootModule, &ObjectProxy_Type, "ObjectProxy" ) )
      PYROOT_INIT_ERROR;

// inject method proxy type
   if ( ! Utility::InitProxy( gRootModule, &MethodProxy_Type, "MethodProxy" ) )
      PYROOT_INIT_ERROR;

// inject template proxy type
   if ( ! Utility::InitProxy( gRootModule, &TemplateProxy_Type, "TemplateProxy" ) )
      PYROOT_INIT_ERROR;

// inject property proxy type
   if ( ! Utility::InitProxy( gRootModule, &PropertyProxy_Type, "PropertyProxy" ) )
      PYROOT_INIT_ERROR;

// inject custom data types
   if ( ! Utility::InitProxy( gRootModule, &TCustomFloat_Type, "Double" ) )
      PYROOT_INIT_ERROR;

   if ( ! Utility::InitProxy( gRootModule, &TCustomInt_Type, "Long" ) )
      PYROOT_INIT_ERROR;

   if ( ! Utility::InitProxy( gRootModule, &TCustomInstanceMethod_Type, "InstanceMethod" ) )
      PYROOT_INIT_ERROR;

// policy labels
   PyModule_AddObject( gRootModule, (char*)"kMemoryHeuristics", PyInt_FromLong( 1l ) );
   PyModule_AddObject( gRootModule, (char*)"kMemoryStrict",     PyInt_FromLong( 2l ) );
   PyModule_AddObject( gRootModule, (char*)"kSignalFast",       PyInt_FromLong( 1l ) );
   PyModule_AddObject( gRootModule, (char*)"kSignalSafe",       PyInt_FromLong( 2l ) );

// setup ROOT
   PyROOT::InitRoot();

// signal policy: don't abort interpreter in interactive mode
   Utility::SetSignalPolicy( gROOT->IsBatch() ? Utility::kFast : Utility::kSafe );

// inject ROOT namespace for convenience
   PyModule_AddObject( gRootModule, (char*)"ROOT",
      MakeRootClassFromString< TScopeAdapter, TBaseAdapter, TMemberAdapter >( "ROOT" ) );

#if PY_VERSION_HEX >= 0x03000000
   Py_INCREF( gRootModule );
   return gRootModule;
#endif
}
