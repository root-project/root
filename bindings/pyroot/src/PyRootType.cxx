// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Jan 2005

// Bindings
#include "PyROOT.h"
#include "PyRootType.h"
#include "RootWrapper.h"
#include "ClassMethodHolder.h"
#include "MethodProxy.h"
#include "PropertyProxy.h"
#include "Adapters.h"

// Standard
#include <string.h>
#include <string>


namespace PyROOT {

namespace {

//= PyROOT type proxy construction/destruction ===============================
   PyObject* meta_alloc( PyTypeObject* metatype, Py_ssize_t nitems )
   {
   // specialized allocator, fitting in a few extra bytes for a TClassRef
      int basicsize = metatype->tp_basicsize;
      metatype->tp_basicsize = sizeof(PyRootClass);
      PyObject* pyclass = PyType_Type.tp_alloc( metatype, nitems );
      metatype->tp_basicsize = basicsize;

      return pyclass;
   }

//____________________________________________________________________________
   void meta_dealloc( PyRootClass* pytype )
   {
      pytype->fClass.~TClassRef();
      return PyType_Type.tp_dealloc( (PyObject*)pytype );
   }

//____________________________________________________________________________
   PyObject* pt_new( PyTypeObject* subtype, PyObject* args, PyObject* kwds )
   {
   // Called when PyRootType acts as a metaclass; since type_new always resets
   // tp_alloc, and since it does not call tp_init on types, the metaclass is
   // being fixed up here, and the class is initialized here as well.

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
         new (&result->fClass) TClassRef( PyROOT_PyUnicode_AsString( PyTuple_GET_ITEM( args, 0 ) ) );
      } else {
      // coming here from PyROOT, use meta class name instead of given name,
      // so that it is safe to inherit python classes from the bound class
         new (&result->fClass) TClassRef( std::string( subtype->tp_name ).substr( 0, mp-subtype->tp_name ).c_str() );
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

            attr = MakeRootClassFromString< TScopeAdapter, TBaseAdapter, TMemberAdapter >( name, pyclass );

         // namespaces may have seen updates in their list of global functions, which
         // are available as "methods" even though they're not really that
            if ( ! attr && PyType_Check( pyclass ) ) {
               PyErr_Clear();
 
               TScopeAdapter klass = TScopeAdapter::ByName( ((PyTypeObject*)pyclass)->tp_name );
               if ( klass.IsNamespace() ) {

                  for ( size_t i = 0; i < klass.FunctionMemberSize(); ++i ) {
                     TMemberAdapter m = klass.FunctionMemberAt( i );
                     if ( m.Name() == name ) {
                     // Note: can't re-use Utility::AddClass here, as there's the risk of
                     // a recursive call. Simply add method directly, as we're guaranteed
                     // that it doesn't exist yet.
                        PyCallable* pyfunc = new TClassMethodHolder< TScopeAdapter, TMemberAdapter >( klass, m );
                        attr = (PyObject*)MethodProxy_New( name.c_str(), pyfunc );
                        PyObject_SetAttr( pyclass, pyname, attr );
                        Py_DECREF( attr );
                        attr = PyType_Type.tp_getattro( pyclass, pyname );
                        break;
                     }
                  }

               }
            }

            if ( ! attr ) {
               PyErr_Clear();
            // get class name to look up CINT tag info ...
               attr = GetRootGlobalFromString( name /*, tag */ );
               if ( PropertyProxy_Check( attr ) ) {
                  PyObject_SetAttr( (PyObject*)Py_TYPE(pyclass), pyname, attr );
                  Py_DECREF( attr );
                  attr = PyType_Type.tp_getattro( pyclass, pyname );
               } else if ( attr )
                  PyObject_SetAttr( pyclass, pyname, attr );
            }

         }

      // if failed, then the original error is likely to be more instructive
         if ( ! attr )
            PyErr_Restore( etype, value, trace );

      // attribute is cached, if found
      }

      return attr;
   }

} // unnamed namespace


//= PyROOT object proxy type type ============================================
PyTypeObject PyRootType_Type = {
   PyVarObject_HEAD_INIT( &PyType_Type, 0 )
   (char*)"ROOT.PyRootType",  // tp_name
   0,                         // tp_basicsize
   0,                         // tp_itemsize
   0,                         // tp_dealloc
   0,                         // tp_print
   0,                         // tp_getattr
   0,                         // tp_setattr
   0,                         // tp_compare
   0,                         // tp_repr
   0,                         // tp_as_number
   0,                         // tp_as_sequence
   0,                         // tp_as_mapping
   0,                         // tp_hash
   0,                         // tp_call
   0,                         // tp_str
   (getattrofunc)pt_getattro, // tp_getattro
   0,                         // tp_setattro
   0,                         // tp_as_buffer
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     // tp_flags
   (char*)"PyROOT metatype (internal)",          // tp_doc
   0,                         // tp_traverse
   0,                         // tp_clear
   0,                         // tp_richcompare
   0,                         // tp_weaklistoffset
   0,                         // tp_iter
   0,                         // tp_iternext
   0,                         // tp_methods
   0,                         // tp_members
   0,                         // tp_getset
   &PyType_Type,              // tp_base
   0,                         // tp_dict
   0,                         // tp_descr_get
   0,                         // tp_descr_set
   0,                         // tp_dictoffset
   0,                         // tp_init
   0,                         // tp_alloc
   (newfunc)pt_new,           // tp_new
   0,                         // tp_free
   0,                         // tp_is_gc
   0,                         // tp_bases
   0,                         // tp_mro
   0,                         // tp_cache
   0,                         // tp_subclasses
   0                          // tp_weaklist
#if PY_VERSION_HEX >= 0x02030000
   , 0                        // tp_del
#endif
#if PY_VERSION_HEX >= 0x02060000
   , 0                        // tp_version_tag
#endif
#if PY_VERSION_HEX >= 0x03040000
   , 0                        // tp_finalize
#endif
};

} // namespace PyROOT
