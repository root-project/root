// Author: Wim Lavrijsen, Dec 2006

// Bindings
#include "PyROOT.h"
#include "TTupleOfInstances.h"
#include "RootWrapper.h"


namespace PyROOT {

//= support for C-style arrays of objects ====================================
PyObject* TTupleOfInstances_New(
      Cppyy::TCppObject_t address, Cppyy::TCppType_t klass, Py_ssize_t nelems )
{
// TODO: the extra copy is inefficient, but it appears that the only way to
// initialize a subclass of a tuple is through a sequence
   PyObject* tup = PyTuple_New( nelems );
   for ( int i = 0; i < nelems; ++i ) {
   // TODO: there's an assumption here that there is no padding, which is bound
   // to be incorrect in certain cases
      PyTuple_SetItem( tup, i,
         BindCppObject( (char*)address + i*Cppyy::SizeOf( klass ), klass, kFALSE /* isRef */ ) );
   // Note: objects are bound as pointers, yet since the pointer value stays in
   // place, updates propagate just as if they were bound by-reference
   }

   PyObject* args = PyTuple_New( 1 );
   Py_INCREF( tup ); PyTuple_SET_ITEM( args, 0, tup );
   PyObject *arr = PyTuple_Type.tp_new(&TTupleOfInstances_Type, args, nullptr);
   if ( PyErr_Occurred() ) PyErr_Print();

   Py_DECREF( args );
   // tup ref eaten by SET_ITEM on args

   return arr;
}

//= PyROOT custom tuple-like array type ======================================
PyTypeObject TTupleOfInstances_Type = {
   PyVarObject_HEAD_INIT(&PyType_Type, 0)(char *) "ROOT.InstancesArray", // tp_name
   0,                                                                    // tp_basicsize
   0,                                                                    // tp_itemsize
   nullptr,                                                              // tp_dealloc
   nullptr,                                                              // tp_print
   nullptr,                                                              // tp_getattr
   nullptr,                                                              // tp_setattr
   nullptr,                                                              // tp_compare
   nullptr,                                                              // tp_repr
   nullptr,                                                              // tp_as_number
   nullptr,                                                              // tp_as_sequence
   nullptr,                                                              // tp_as_mapping
   nullptr,                                                              // tp_hash
   nullptr,                                                              // tp_call
   nullptr,                                                              // tp_str
   nullptr,                                                              // tp_getattro
   nullptr,                                                              // tp_setattro
   nullptr,                                                              // tp_as_buffer
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_CHECKTYPES | Py_TPFLAGS_BASETYPE,     // tp_flags
   (char *)"PyROOT long object for pass by reference",                   // tp_doc
   nullptr,                                                              // tp_traverse
   nullptr,                                                              // tp_clear
   nullptr,                                                              // tp_richcompare
   0,                                                                    // tp_weaklistoffset
   nullptr,                                                              // tp_iter
   nullptr,                                                              // tp_iternext
   nullptr,                                                              // tp_methods
   nullptr,                                                              // tp_members
   nullptr,                                                              // tp_getset
   &PyTuple_Type,                                                        // tp_base
   nullptr,                                                              // tp_dict
   nullptr,                                                              // tp_descr_get
   nullptr,                                                              // tp_descr_set
   0,                                                                    // tp_dictoffset
   nullptr,                                                              // tp_init
   nullptr,                                                              // tp_alloc
   nullptr,                                                              // tp_new
   nullptr,                                                              // tp_free
   nullptr,                                                              // tp_is_gc
   nullptr,                                                              // tp_bases
   nullptr,                                                              // tp_mro
   nullptr,                                                              // tp_cache
   nullptr,                                                              // tp_subclasses
   nullptr                                                               // tp_weaklist
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
