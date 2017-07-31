// @(#)root/pyroot:$Id$
// Author: Jim Pivarski, Jul 2017

#ifndef PYROOT_NUMPYITERATOR_H
#define PYROOT_NUMPYITERATOR_H

#include <Python.h>

class NumpyIterator;

typedef struct {
  PyObject_HEAD
  NumpyIterator* iter;
} PyNumpyIterator;

static PyObject* PyNumpyIterator_iter(PyObject* self);
static PyObject* PyNumpyIterator_next(PyObject* self);
static void PyNumpyIterator_del(PyNumpyIterator* self);

#if PY_MAJOR_VERSION >= 3
#define Py_TPFLAGS_HAVE_ITER 0
#endif

static PyTypeObject PyNumpyIteratorType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "numpyinterface.NumpyIterator", /*tp_name*/
  sizeof(PyNumpyIterator),  /*tp_basicsize*/
  0,                         /*tp_itemsize*/
  (destructor)PyNumpyIterator_del, /*tp_dealloc*/
  0,                         /*tp_print*/
  0,                         /*tp_getattr*/
  0,                         /*tp_setattr*/
  0,                         /*tp_compare*/
  0,                         /*tp_repr*/
  0,                         /*tp_as_number*/
  0,                         /*tp_as_sequence*/
  0,                         /*tp_as_mapping*/
  0,                         /*tp_hash */
  0,                         /*tp_call*/
  0,                         /*tp_str*/
  0,                         /*tp_getattro*/
  0,                         /*tp_setattro*/
  0,                         /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_ITER, /* tp_flags */
  "Iterator over selected TTree branches, yielding a tuple of (entry_start, entry_end, *arrays) for each cluster.", /* tp_doc */
  0,                         /* tp_traverse */
  0,                         /* tp_clear */
  0,                         /* tp_richcompare */
  0,                         /* tp_weaklistoffset */
  PyNumpyIterator_iter, /* tp_iter: __iter__() method */
  PyNumpyIterator_next, /* tp_iternext: __next__() method */

      0,                         // tp_methods
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
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

static PyObject* GetNumpyIterator(PyObject* self, PyObject* args, PyObject* kwds);
static PyObject* GetNumpyTypeAndSize(PyObject* self, PyObject* args, PyObject* kwds);

#endif // PYROOT_NUMPYITERATOR_H
