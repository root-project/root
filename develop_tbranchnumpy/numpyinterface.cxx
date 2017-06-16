#include <deque>
#include <string>
#include <vector>

#include <Python.h>
#include <numpy/arrayobject.h>

#include <ROOT/TBulkBranchRead.hxx>
#include <TBranch.h>
#include <TBufferFile.h>
#include <TClass.h>
#include <TDataType.h>
#include <TFile.h>
#include <TLeafB.h>
#include <TLeafB.h>
#include <TLeafD.h>
#include <TLeafF.h>
#include <TLeaf.h>
#include <TLeafI.h>
#include <TLeafI.h>
#include <TLeafL.h>
#include <TLeafL.h>
#include <TLeafO.h>
#include <TLeafS.h>
#include <TLeafS.h>
#include <TObjArray.h>
#include <TTree.h>

/////////////////////////////////////////////////////// module

// performance counters for diagnostics
static Long64_t baskets_loaded = 0;
static Long64_t bytes_loaded = 0;
static Long64_t baskets_copied = 0;
static Long64_t bytes_copied = 0;

class BasketBuffer {
public:
  Long64_t entry_start;
  Long64_t entry_end;
  TBufferFile buffer;

  BasketBuffer() : entry_start(0), entry_end(0), buffer(TBuffer::kWrite, 32*1024) {}

  void read_basket(Long64_t entry, TBranch* branch) {
    entry_start = entry;
    entry_end = entry_start + branch->GetBulkRead().GetEntriesSerialized(entry, buffer);
    if (entry_end < entry_ start)
      entry_end = -1;
  }
};

class BranchData {
public:
  TBranch* branch;
  std::deque<BasketBuffer*> buffers;
  TBufferFile extra_buffer(TBuffer::kWrite, 32*1024);
  
  BranchData* counter;

  BranchData(TBranch* branch) : branch(branch) {
    buffers.push_back(new BasketBuffer);
  }

  ~BranchData() {
    while (!buffers.empty()) {
      delete buffers.front();
      buffers.pop_front();
    }
  }

  void* getdata(Long64_t alignment, Long64_t &numbytes) {
    // HERE
  }
};

typedef struct {
  PyObject_HEAD
  Long64_t alignment;
  Long64_t num_entries;
  Long64_t entry_start;
  Long64_t entry_end;
  std::vector<BranchData> requested;
  std::vector<PyArray_Descr*> dtypes;
  std::vector<BranchData> extra_counters;
} BranchesIterator;

static PyObject* BranchesIterator_iter(PyObject* self);
static PyObject* BranchesIterator_next(PyObject* self);

static PyObject* iterate(PyObject* self, PyObject* args);

#if PY_MAJOR_VERSION >=3
#define Py_TPFLAGS_HAVE_ITER 0
#endif

static PyTypeObject BranchesIteratorType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "numpyinterface.BranchesIterator", /*tp_name*/
  sizeof(BranchesIterator), /*tp_basicsize*/
  0,                         /*tp_itemsize*/
  0,                         /*tp_dealloc*/
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
  BranchesIterator_iter, /* tp_iter: __iter__() method */
  BranchesIterator_next  /* tp_iternext: __next__() method */
};

static PyMethodDef module_methods[] = {
  {"iterate", (PyCFunction)iterate, METH_VARARGS, "Get an iterator over a selected set of TTree branches, yielding a tuple of (entry_start, entry_end, *arrays) for each cluster.\n\n    filePath (str): name of the TFile\n    treePath (str): name of the TTree\n    *branchNames (strs): name of requested branches\n\nAlternatively, TBranch objects from PyROOT may be supplied (FIXME).\n\n    alignment=0: if supplied and positive, guarantee that the data are aligned to this number of bytes, even if that means copying data."},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "numpyinterface",
  NULL,
  0,
  module_methods,
  NULL,
  NULL,
  NULL,
  NULL
};

PyMODINIT_FUNC PyInit_numpyinterface(void) {
  BranchesIteratorType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&BranchesIteratorType) < 0)
    return NULL;

  PyObject* module = PyModule_Create(&moduledef);
  if (module == NULL)
    return NULL;

  import_array();

  Py_INCREF(&BranchesIteratorType);
  PyModule_AddObject(module, "BranchesIterator", (PyObject*)&BranchesIteratorType);

  return module;
}

#else // PY_MAJOR_VERSION <= 2

PyMODINIT_FUNC initnumpyinterface(void) {
  BranchesIteratorType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&BranchesIteratorType) < 0)
    return;

  PyObject* module = Py_InitModule3("numpyinterface", module_methods, "");
  if (module == NULL)
    return;

  Py_INCREF(&BranchesIteratorType);
  PyModule_AddObject(module, "BranchesIterator", (PyObject*)&BranchesIteratorType);

  if (module != NULL)
    import_array();
}

#endif

static PyObject* BranchesIterator_iter(PyObject* self) {
  Py_INCREF(self);
  return self;
}

bool update_BranchesIterator(BranchesIterator* thyself, const char* &error_string) {





}


static PyObject* BranchesIterator_next(PyObject* self) {
  BranchesIterator* thyself = reinterpret_cast<BranchesIterator*>(self);

  const char* error_string = NULL;
  bool done = update_BranchesIterator(thyself, error_string);

  if (error_string != NULL) {
    PyErr_SetString(PyExc_IOError, error_string);
    return NULL;
  }
  else if (done) {
    PyErr_SetNone(PyExc_StopIteration);
    return NULL;
  }
  else {
    PyObject* out = PyTuple_New(2 + thyself->requested.size());
    PyTuple_SET_ITEM(out, 0, PyLong_FromLong(thyself->entry_start));
    PyTuple_SET_ITEM(out, 1, PyLong_FromLong(thyself->entry_end));

    for (unsigned int i = 0;  i < thyself->requested.size();  i++) {
      Long64_t numbytes;
      void* data = thyself->requested[i].getdata(thyself->alignment, numbytes);

      npy_intp dims[1];
      dims[0] = numbytes / thyself->dtypes[i].elsize;

      int flags = NPY_ARRAY_C_CONTIGUOUS;
      if (thyself->alignment > 0)
        flags |= NPY_ARRAY_ALIGNED;

      PyObject* array = PyArray_NewFromDescr(&PyArray_Type, thyself->dtypes[i], 1, dims, NULL, data, flags, NULL);

      PyTuple_SET_ITEM(out, i + 2, array);
    }

    return out;
  }
}

static PyObject* iterate(PyObject* self, PyObject* args) {

  // Long64_t alignment;
  // Long64_t num_entries;
  // Long64_t entry_start;
  // Long64_t entry_end;
  // std::vector<BranchData> requested;
  // std::vector<PyArray_Descr*> dtypes;
  // std::vector<BranchData> extra_counters;



}
