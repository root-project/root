// @(#)root/pyroot:$Id$
// Author: Jim Pivarski, Jul 2017

#ifndef PYROOT_NUMPYITERATOR_H
#define PYROOT_NUMPYITERATOR_H

#include <Python.h>

// Numpy include must be first
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// ROOT
#include <TBranch.h>
#include <TBufferFile.h>

// Standard
#include <vector>

namespace PyROOT {

#if PY_VERSION_HEX >= 0x03000000
void* InitializeNumpy();
#else
void InitializeNumpy();
#endif

PyObject* GetNumpyIterator(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* GetNumpyIteratorInfo(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* FillNumpy(PyObject* self, PyObject* args);

class ArrayInfo {
public:
  PyArray_Descr* dtype;
  int nd;
  std::vector<int> dims;
  std::string counter;
};

class Request {
public:
  TBranch* branch;
  std::string leaf;
};

class ClusterBuffer {
private:
  const Request fRequest;
  const Long64_t fItemSize;
  const bool fSwapBytes;
  TBufferFile fBufferFile;
  std::vector<char> fExtra;
  bool fUsingExtra;

  // always numbers of entries (not bytes) and always inclusive on start, exclusive on end (like Python)
  // also, the TBufferFile is always ahead of the extra buffer and there's no gap between them
  Long64_t bfEntryStart;
  Long64_t bfEntryEnd;
  Long64_t exEntryStart;
  Long64_t exEntryEnd;

  void CopyToExtra(Long64_t keep_start);
  void CheckExtraAllocations();

public:
  ClusterBuffer(const Request request, const Long64_t itemsize, const bool swap_bytes) :
    fRequest(request), fItemSize(itemsize), fSwapBytes(swap_bytes), fBufferFile(TBuffer::kWrite, 32*1024), fUsingExtra(false),
    bfEntryStart(0), bfEntryEnd(0), exEntryStart(0), exEntryEnd(0)
  {
    // required for re-readability
    fRequest.branch->DropBaskets();
  }

  void ReadOne(Long64_t keep_start, const char* &error_string);
  void* GetBuffer(Long64_t &numbytes, Long64_t entry_start, Long64_t entry_end);
  Long64_t EntryEnd();
  void Reset();
};

class NumpyIterator {
private:
  std::vector<std::unique_ptr<ClusterBuffer>> fClusterBuffers;
  const std::vector<ArrayInfo> fArrayInfo;   // has the same length as fClusterBuffers
  const Long64_t fNumEntries;
  const bool fReturnNewBuffers;
  Long64_t fCurrentStart;
  Long64_t fCurrentEnd;

  bool StepForward(const char* &error_string);

public:
  NumpyIterator(const std::vector<Request> &requests, const std::vector<ArrayInfo> arrayinfo, Long64_t num_entries, bool return_new_buffers, bool swap_bytes) :
    fArrayInfo(arrayinfo), fNumEntries(num_entries), fReturnNewBuffers(return_new_buffers), fCurrentStart(0), fCurrentEnd(0)
  {
    for (unsigned int i = 0;  i < fArrayInfo.size();  i++) {
      fClusterBuffers.push_back(std::unique_ptr<ClusterBuffer>(new ClusterBuffer(requests[i], fArrayInfo[i].dtype->elsize, swap_bytes)));
    }
  }

  PyObject* arrays();
  void Reset();
};

typedef struct {
  PyObject_HEAD
  NumpyIterator* iter;
} PyNumpyIterator;

static PyObject* PyNumpyIterator_iter(PyObject* self) {
  PyNumpyIterator* thyself = reinterpret_cast<PyNumpyIterator*>(self);
  thyself->iter->Reset();
  Py_INCREF(self);
  return self;
}

static PyObject* PyNumpyIterator_next(PyObject* self) {
  PyNumpyIterator* thyself = reinterpret_cast<PyNumpyIterator*>(self);
  return thyself->iter->arrays();
}

static void PyNumpyIterator_del(PyNumpyIterator* thyself) {
  if (thyself->iter != 0)
    delete thyself->iter;
  PyObject_Del(reinterpret_cast<PyObject*>(thyself));   // because NumpyIterator is not subclassable
}

#if PY_MAJOR_VERSION >= 3
#define Py_TPFLAGS_HAVE_ITER 0
#endif

static PyTypeObject PyNumpyIteratorType = {
  PyVarObject_HEAD_INIT( &PyType_Type, 0 )
  "ROOT.NumpyIterator",      /*tp_name*/
  sizeof(PyNumpyIterator),   /*tp_basicsize*/
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
  PyNumpyIterator_iter,      /* tp_iter: __iter__() method */
  PyNumpyIterator_next,      /* tp_iternext: __next__() method */
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
  0,                         // tp_new
  0,                         // tp_free
  0,                         // tp_is_gc
  0,                         // tp_bases
  0,                         // tp_mro
  0,                         // tp_cache
  0,                         // tp_subclasses
  0                          // tp_weaklist
#if PY_VERSION_HEX >= 0x02030000
, 0                           // tp_del
#endif
#if PY_VERSION_HEX >= 0x02060000
, 0                           // tp_version_tag
#endif
#if PY_VERSION_HEX >= 0x03040000
, 0                           // tp_finalize
#endif
};

} // namespace PyROOT

#endif // PYROOT_NUMPYITERATOR_H
