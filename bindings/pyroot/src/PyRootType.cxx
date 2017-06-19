// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Jan 2005

// Bindings
#include "PyROOT.h"
#include "PyRootType.h"
#include "MethodProxy.h"
#include "PropertyProxy.h"
#include "RootWrapper.h"
#include "TFunctionHolder.h"
#include "TemplateProxy.h"
#include "PyStrings.h"

// ROOT
#include "TClass.h"     // for method and enum finding
#include "TList.h"      // id.

// Standard
#include <string.h>
#include <string>
#include <vector>


namespace PyROOT {

namespace {

//= PyROOT type proxy construction/destruction ===============================
   PyObject* meta_alloc( PyTypeObject* metatype, Py_ssize_t nitems )
   {
   // specialized allocator, fitting in a few extra bytes for a TClassRef
      PyObject* pyclass = PyType_Type.tp_alloc( metatype, nitems );

      return pyclass;
   }

////////////////////////////////////////////////////////////////////////////////

   void meta_dealloc( PyRootClass* pytype )
   {
      return PyType_Type.tp_dealloc( (PyObject*)pytype );
   }

////////////////////////////////////////////////////////////////////////////////
/// Called when PyRootType acts as a metaclass; since type_new always resets
/// tp_alloc, and since it does not call tp_init on types, the metaclass is
/// being fixed up here, and the class is initialized here as well.

   PyObject* pt_new( PyTypeObject* subtype, PyObject* args, PyObject* kwds )
   {
   // fixup of metaclass (left permanent, and in principle only called once b/c
   // PyROOT caches python classes)
      subtype->tp_alloc   = (allocfunc)meta_alloc;
      subtype->tp_dealloc = (destructor)meta_dealloc;

   // creation of the python-side class
      PyRootClass* result = (PyRootClass*)PyType_Type.tp_new( subtype, args, kwds );

   // initialization of class (based on name only, initially, which is lazy)

   // there's a snag here: if a python class is derived from the bound class,
   // the name will not be known by TClassRef, hence we'll use the meta class
   // name from the subtype, rather than given class name

      const char* mp = strstr( subtype->tp_name, "_meta" );
      if ( ! mp ) {
      // there has been a user meta class override in a derived class, so do
      // the consistent thing, thus allowing user control over naming
         result->fCppType = Cppyy::GetScope(
            PyROOT_PyUnicode_AsString( PyTuple_GET_ITEM( args, 0 ) ) );
      } else {
      // coming here from PyROOT, use meta class name instead of given name,
      // so that it is safe to inherit python classes from the bound class
         result->fCppType = Cppyy::GetScope(
            std::string( subtype->tp_name ).substr( 0, mp-subtype->tp_name ).c_str() );
      }

      return (PyObject*)result;
   }


//= PyROOT type metaclass behavior ===========================================
   PyObject* pt_getattro( PyObject* pyclass, PyObject* pyname )
   {
   // normal type lookup
      PyObject* attr = PyType_Type.tp_getattro( pyclass, pyname );

   // extra ROOT lookup in case of failure (e.g. for inner classes on demand)
      if ( ! attr && PyROOT_PyUnicode_CheckExact( pyname ) ) {
         PyObject *etype, *value, *trace;
         PyErr_Fetch( &etype, &value, &trace );         // clears current exception

      // filter for python specials and lookup qualified class or function
         std::string name = PyROOT_PyUnicode_AsString( pyname );
         if ( name.size() <= 2 || name.substr( 0, 2 ) != "__" ) {
            attr = CreateScopeProxy( name, pyclass );

         // namespaces may have seen updates in their list of global functions, which
         // are available as "methods" even though they're not really that
            if ( ! attr && ! PyRootType_CheckExact( pyclass ) && PyType_Check( pyclass ) ) {
               PyErr_Clear();
               PyObject* pycppname = PyObject_GetAttr( pyclass, PyStrings::gCppName );
               char* cppname = PyROOT_PyUnicode_AsString(pycppname);
               Py_DECREF(pycppname);
               Cppyy::TCppScope_t scope = Cppyy::GetScope( cppname );
               TClass* klass = TClass::GetClass( cppname );
               if ( Cppyy::IsNamespace( scope ) ) {

               // tickle lazy lookup of functions
                  if ( ! attr ) {
                     if ( klass->GetListOfMethods()->FindObject( name.c_str() ) ) {
                     // function exists, now collect overloads
                        std::vector< PyCallable* > overloads;
                        const size_t nmeth = Cppyy::GetNumMethods( scope );
                        for ( size_t imeth = 0; imeth < nmeth; ++imeth ) {
                           Cppyy::TCppMethod_t method = Cppyy::GetMethod( scope, imeth );
                           if ( Cppyy::GetMethodName( method ) == name )
                              overloads.push_back( new TFunctionHolder( scope, method ) );
                        }

                     // Note: can't re-use Utility::AddClass here, as there's the risk of
                     // a recursive call. Simply add method directly, as we're guaranteed
                     // that it doesn't exist yet.
                        attr = (PyObject*)MethodProxy_New( name.c_str(), overloads );
                     }
                  }

               // tickle lazy lookup of data members
                  if ( ! attr ) {
                      Cppyy::TCppIndex_t dmi = Cppyy::GetDatamemberIndex( scope, name );
                      if ( 0 <= dmi ) attr = (PyObject*)PropertyProxy_New( scope, dmi );
                  }
               }

            // function templates that have not been instantiated
               if ( ! attr && klass ) {
                  TFunctionTemplate* tmpl = klass->GetFunctionTemplate( name.c_str() );
                  if ( tmpl )
                     attr = (PyObject*)TemplateProxy_New( name, pyclass );
               }

            // enums types requested as type (rather than the constants)
               if ( ! attr && klass && klass->GetListOfEnums()->FindObject( name.c_str() ) ) {
               // special case; enum types; for now, pretend int
               // TODO: although fine for C++98, this isn't correct in C++11
                  Py_INCREF( &PyInt_Type );
                  attr = (PyObject*)&PyInt_Type;
               }

               if ( attr ) {
                  PyObject_SetAttr( pyclass, pyname, attr );
                  Py_DECREF( attr );
                  attr = PyType_Type.tp_getattro( pyclass, pyname );
               }
            }

            if ( ! attr && ! PyRootType_Check( pyclass ) /* at global or module-level only */ ) {
               PyErr_Clear();
            // get class name to look up CINT tag info ...
               attr = GetCppGlobal( name /*, tag */ );
               if ( PropertyProxy_Check( attr ) ) {
                  PyObject_SetAttr( (PyObject*)Py_TYPE(pyclass), pyname, attr );
                  Py_DECREF( attr );
                  attr = PyType_Type.tp_getattro( pyclass, pyname );
               } else if ( attr )
                  PyObject_SetAttr( pyclass, pyname, attr );
            }

         }

      // if failed, then the original error is likely to be more instructive
         if ( ! attr && etype )
            PyErr_Restore( etype, value, trace );
         else if ( ! attr ) {
            PyObject* sklass = PyObject_Str( pyclass );
            PyErr_Format( PyExc_AttributeError, "%s has no attribute \'%s\'",
               PyROOT_PyUnicode_AsString( sklass ), PyROOT_PyUnicode_AsString( pyname ) );
            Py_DECREF( sklass );
         }

      // attribute is cached, if found
      }

      return attr;
   }

} // unnamed namespace


//= PyROOT object proxy type type ============================================
PyTypeObject PyRootType_Type = {
   PyVarObject_HEAD_INIT(&PyType_Type, 0)(char *) "ROOT.PyRootType", // tp_name
   sizeof(PyROOT::PyRootClass),                                      // tp_basicsize
   0,                                                                // tp_itemsize
   nullptr,                                                          // tp_dealloc
   nullptr,                                                          // tp_print
   nullptr,                                                          // tp_getattr
   nullptr,                                                          // tp_setattr
   nullptr,                                                          // tp_compare
   nullptr,                                                          // tp_repr
   nullptr,                                                          // tp_as_number
   nullptr,                                                          // tp_as_sequence
   nullptr,                                                          // tp_as_mapping
   nullptr,                                                          // tp_hash
   nullptr,                                                          // tp_call
   nullptr,                                                          // tp_str
   (getattrofunc)pt_getattro,                                        // tp_getattro
   nullptr,                                                          // tp_setattro
   nullptr,                                                          // tp_as_buffer
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,                         // tp_flags
   (char *)"PyROOT metatype (internal)",                             // tp_doc
   nullptr,                                                          // tp_traverse
   nullptr,                                                          // tp_clear
   nullptr,                                                          // tp_richcompare
   0,                                                                // tp_weaklistoffset
   nullptr,                                                          // tp_iter
   nullptr,                                                          // tp_iternext
   nullptr,                                                          // tp_methods
   nullptr,                                                          // tp_members
   nullptr,                                                          // tp_getset
   &PyType_Type,                                                     // tp_base
   nullptr,                                                          // tp_dict
   nullptr,                                                          // tp_descr_get
   nullptr,                                                          // tp_descr_set
   0,                                                                // tp_dictoffset
   nullptr,                                                          // tp_init
   nullptr,                                                          // tp_alloc
   (newfunc)pt_new,                                                  // tp_new
   nullptr,                                                          // tp_free
   nullptr,                                                          // tp_is_gc
   nullptr,                                                          // tp_bases
   nullptr,                                                          // tp_mro
   nullptr,                                                          // tp_cache
   nullptr,                                                          // tp_subclasses
   nullptr                                                           // tp_weaklist
#if PY_VERSION_HEX >= 0x02030000
   ,
   nullptr // tp_del
#endif
#if PY_VERSION_HEX >= 0x02060000
   ,
   0 // tp_version_tag
#endif
#if PY_VERSION_HEX >= 0x03040000
   ,
   0 // tp_finalize
#endif
};

} // namespace PyROOT
