// @(#)root/pyroot:$Name:  $:$Id: ObjectProxy.cxx,v 1.1 2005/03/04 07:44:11 brun Exp $
// Author: Wim Lavrijsen, Jan 2005

// Bindings
#include "PyROOT.h"
#include "ObjectProxy.h"

// ROOT
#include "TObject.h"
#include "TClass.h"


namespace PyROOT {

//= PyROOT object proxy pointer arithmetic support ===========================
namespace {

   int IsNotZero( ObjectProxy* self )
   {
      return (bool) self->GetObject();
   }


   PyNumberMethods ObjectProxyAsNumber = {
      0,                     // nb_add
      0,                     // nb_subtract
      0,                     // nb_multiply
      0,                     // nb_divide
      0,                     // nb_remainder
      0,                     // nb_divmod
      0,                     // nb_power
      0,                     // nb_negative
      0,                     // nb_positive
      0,                     // nb_absolute
      (inquiry)IsNotZero,    // nb_nonzero
      0,                     // nb_invert
      0,                     // nb_lshift
      0,                     // nb_rshift
      0,                     // nb_and
      0,                     // nb_xor
      0,                     // nb_or
      0,                     // nb_coerce
      0,                     // nb_int
      0,                     // nb_long
      0,                     // nb_float
      0,                     // nb_oct
      0,                     // nb_hex
      0,                     // nb_inplace_add
      0,                     // nb_inplace_subtract
      0,                     // nb_inplace_multiply
      0,                     // nb_inplace_divide
      0,                     // nb_inplace_remainder
      0,                     // nb_inplace_power
      0,                     // nb_inplace_lshift
      0,                     // nb_inplace_rshift
      0,                     // nb_inplace_and
      0,                     // nb_inplace_xor
      0,                     // nb_inplace_or
      0,                     // nb_floor_divide
      0,                     // nb_true_divide
      0,                     // nb_inplace_floor_divide
      0                      // nb_inplace_true_divide
   };


//= PyROOT method proxy construction/destruction =================================
   void op_dealloc( ObjectProxy* pyobj )
   {
      if ( pyobj->fObject && ( pyobj->fFlags & ObjectProxy::kIsOwner ) ) {
         pyobj->fClass->Destructor( pyobj->fObject );
      }

      pyobj->ob_type->tp_free( (PyObject*)pyobj );
   }

} // unnamed namespace


//= PyROOT object proxy type =================================================
PyTypeObject ObjectProxy_Type = {
   PyObject_HEAD_INIT( &PyType_Type )
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
   &ObjectProxyAsNumber,      // tp_as_number
   0,                         // tp_as_sequence
   0,                         // tp_as_mapping
   0,                         // tp_hash
   0,                         // tp_call
   0,                         // tp_str
   0,                         // tp_getattro
   0,                         // tp_setattro
   0,                         // tp_as_buffer
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     // tp_flags
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
   PyType_GenericNew,         // tp_new
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
