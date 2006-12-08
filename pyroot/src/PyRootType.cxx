// @(#)root/pyroot:$Name:  $:$Id: PyRootType.cxx,v 1.6 2006/08/14 00:21:56 rdm Exp $
// Author: Wim Lavrijsen, Jan 2005

// Bindings
#include "PyROOT.h"
#include "PyRootType.h"
#include "RootWrapper.h"
#include "Adapters.h"

// Standard
#include <string>


namespace PyROOT {

namespace {

   PyObject* pt_getattro( PyObject* pyclass, PyObject* pyname )
   {
   // normal type lookup
      PyObject* attr = PyType_Type.tp_getattro( pyclass, pyname );

   // extra ROOT lookup in case of failure (e.g. for inner classes on demand)
      if ( ! attr && PyString_CheckExact( pyname ) ) {
         PyObject *etype, *value, *trace;
         PyErr_Fetch( &etype, &value, &trace );         // clears current exception

      // filter for python specials and lookup qualified class or function
         std::string name = PyString_AS_STRING( pyname );
         if ( name.size() <= 2 || name.substr( 0, 2 ) != "__" ) {

            attr = MakeRootClassFromString< TScopeAdapter, TBaseAdapter, TMemberAdapter >( name, pyclass );

            if ( ! attr ) {
               PyErr_Clear();
            // get class name to look up CINT tag info ... 
               attr = GetRootGlobalFromString( name /*, tag */ );
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
   PyObject_HEAD_INIT( &PyType_Type )
   0,                         // ob_size
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
   pt_getattro,               // tp_getattro
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
   0,                         // tp_new
   0,                         // tp_free
   0,                         // tp_is_gc
   0,                         // tp_bases
   0,                         // tp_mro
   0,                         // tp_cache
   0,                         // tp_subclasses
   0,                         // tp_weaklist

#if PY_MAJOR_VERSION >= 2 && PY_MINOR_VERSION >= 3
   0                          // tp_del
#endif
};

} // namespace PyROOT
