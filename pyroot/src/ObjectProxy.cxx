// @(#)root/pyroot:$Name:  $:$Id: ObjectProxy.cxx,v 1.5 2005/06/06 15:08:40 brun Exp $
// Author: Wim Lavrijsen, Jan 2005

// Bindings
#include "PyROOT.h"
#include "ObjectProxy.h"
#include "PyRootType.h"

// ROOT
#include "TObject.h"


namespace PyROOT {

namespace {

//= PyROOT method proxy construction/destruction =============================
   ObjectProxy* op_new( PyTypeObject* subtype, PyObject*, PyObject* )
   {
      ObjectProxy* pyobj = (ObjectProxy*)subtype->tp_alloc( subtype, 0 );
      pyobj->fObject = NULL;
      new (&pyobj->fClass) TClassRef( (TClass*)0 );
      pyobj->fFlags  = 0;

      return pyobj;
   }

//____________________________________________________________________________
   void op_dealloc( ObjectProxy* pyobj )
   {
      if ( pyobj->fObject && ( pyobj->fFlags & ObjectProxy::kIsOwner ) ) {
         pyobj->fClass->Destructor( pyobj->fObject );
      }

      pyobj->fClass.~TClassRef();
      pyobj->ob_type->tp_free( (PyObject*)pyobj );
   }

} // unnamed namespace


//= PyROOT object proxy type =================================================
PyTypeObject ObjectProxy_Type = {
   PyObject_HEAD_INIT( &PyRootType_Type )
   0,                         // ob_size
   (char*)"ROOT.ObjectProxy", // tp_name
   sizeof(ObjectProxy),       // tp_basicsize
   0,                         // tp_itemsize
   (destructor)op_dealloc,    // tp_dealloc
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
   0,                         // tp_getattro
   0,                         // tp_setattro
   0,                         // tp_as_buffer
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
      Py_TPFLAGS_HAVE_GC,                        // tp_flags
   (char*)"PyROOT object proxy (internal)",      // tp_doc
   0,                         // tp_traverse
   0,                         // tp_clear
   0,                         // tp_richcompare
   0,                         // tp_weaklistoffset
   0,                         // tp_iter
   0,                         // tp_iternext
   0,                         // tp_methods
   0,                         // tp_members
   0,                         // tp_getset
   0,                         // tp_base
   0,                         // tp_dict
   0,                         // tp_descr_get
   0,                         // tp_descr_set
   0,                         // tp_dictoffset
   0,                         // tp_init
   0,                         // tp_alloc
   (newfunc)op_new,           // tp_new
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
