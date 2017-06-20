#include <iostream>

#include <string>
#include <vector>

#include <Python.h>
#include <numpy/arrayobject.h>

#include <ROOT/TBulkBranchRead.hxx>
#include <TBranch.h>
#include <TBufferFile.h>
#include <TBuffer.h>
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

#define ALIGNMENT 8    // if a pointer % ALIGNMENT == 0, declare it "aligned"

/////////////////////////////////////////////////////// helper classes

class ArrayInfo {
public:
  PyArray_Descr* dtype;
  int nd;
  std::vector<int> dims;
  bool varlen;
};

class ClusterBuffer {
private:
  const TBranch* branch;
  const Long64_t itemsize;
  TBufferFile bf;
  std::vector<char> extra;

  // always numbers of entries (not bytes) and always inclusive on start, exclusive on end (like Python)
  // also, the TBufferFile is always ahead of the extra buffer and there's no gap between them
  Long64_t bf_entry_start;
  Long64_t bf_entry_end;
  Long64_t ex_entry_start;
  Long64_t ex_entry_end;

  void copy_to_extra(Long64_t keep_start);

public:
  ClusterBuffer(const TBranch* branch, const Long64_t itemsize) :
    branch(branch), itemsize(itemsize), bf(TBuffer::kWrite, 32*1024),
    bf_entry_start(0), bf_entry_end(0), ex_entry_start(0), ex_entry_end(0) { }

  void readone(Long64_t keep_start, const char* &error_string);
  void* getbuffer(Long64_t &numbytes, bool require_alignment, Long64_t entry_start, Long64_t entry_end);

  Long64_t entry_end() { return bf_entry_end; }  // hide the distinction between bf and extra
};

class ClusterIterator {
private:
  std::vector<std::unique_ptr<ClusterBuffer>> requested;
  std::vector<std::unique_ptr<ClusterBuffer>> extra_counters;
  const std::vector<ArrayInfo> arrayinfo;   // has the same length as requested
  const Long64_t num_entries;
  const bool return_new_buffers;
  const bool require_alignment;
  Long64_t current_start;
  Long64_t current_end;

  bool stepforward(const char* &error_string);

public:
  ClusterIterator(const std::vector<TBranch*> &requested_branches, const std::vector<TBranch*> &unrequested_counters, const std::vector<ArrayInfo> arrayinfo, Long64_t num_entries, bool return_new_buffers, bool require_alignment) :
    arrayinfo(arrayinfo), num_entries(num_entries), return_new_buffers(return_new_buffers), require_alignment(require_alignment), current_start(0), current_end(0) {
    for (unsigned int i = 0;  i < arrayinfo.size();  i++)
      requested.push_back(std::unique_ptr<ClusterBuffer>(new ClusterBuffer(requested_branches[i], arrayinfo[i].dtype->elsize)));
  }

  PyObject* arrays();

  void reset() {
    current_start = 0;
    current_end = 0;
  }
};    

void ClusterBuffer::copy_to_extra(Long64_t keep_start) {
  const Long64_t numbytes = (bf_entry_end - bf_entry_start) * itemsize;

  // if the extra buffer has anything worth saving in it, append
  if (ex_entry_end > keep_start) {
    const Long64_t oldsize = extra.size();
    extra.resize(oldsize + numbytes);
    memcpy(&extra.data()[oldsize], bf.GetCurrent(), numbytes);
    ex_entry_end = bf_entry_end;
  }
  // otherwise, replace
  else {
    extra.resize(numbytes);
    memcpy(extra.data(), bf.GetCurrent(), numbytes);
    ex_entry_start = bf_entry_start;
    ex_entry_end = bf_entry_end;
  }
}

// readone asks ROOT to read one basket from the file
// and ClusterBuffer ensures that entries as old as keep_start are preserved
void ClusterBuffer::readone(Long64_t keep_start, const char* &error_string) {
  // if the TBufferFile has anything worth saving in it, save it using the extra buffer
  if (bf_entry_end > keep_start)
    copy_to_extra(keep_start);

  // read in one more basket, starting at the old bf_entry_end
  Long64_t numentries = branch->GetBulkRead().GetEntriesSerialized(bf_entry_end, bf);

  // update the range
  bf_entry_start = bf_entry_end;
  bf_entry_end = bf_entry_start + numentries;

  // check for errors
  if (numentries <= 0) {
    bf_entry_end = bf_entry_start;
    error_string = "failed to read TBasket into TBufferFile (using GetBulkRead().GetEntriesSerialized)";
  }
}

// getbuffer returns a pointer to contiguous data with its size
// if you're lucky (and ask for it), this is performed without any copies
void* ClusterBuffer::getbuffer(Long64_t &numbytes, bool require_alignment, Long64_t entry_start, Long64_t entry_end) {
  // if the TBufferFile is a perfect match to the request and we either don't care about alignment or it is aligned, return it directly
  if (bf_entry_start == entry_start  &&  bf_entry_end == entry_end  &&  (!require_alignment  ||  (size_t)bf.GetCurrent() % ALIGNMENT == 0)) {
    numbytes = (entry_end - entry_start) * itemsize;
    return bf.GetCurrent();
  }
  // otherwise, move everything into the extra buffer and return it
  else {
    copy_to_extra(entry_start);
    numbytes = (entry_end - entry_start) * itemsize;
    return &extra.data()[(entry_start - ex_entry_start) * itemsize];
  }
}

// step all ClusterBuffers forward, for all branches, returning true when done and setting error_string on any errors
bool ClusterIterator::stepforward(const char* &error_string) {
  // put your feet together for the next step
  current_start = current_end;

  // check for done
  if (current_end >= num_entries)
    return true;

  // increment the branches that are at the forefront
  for (unsigned int i = 0;  i < requested.size();  i++) {
    ClusterBuffer &buf = *requested[i];
    if (buf.entry_end() == current_start) {
      buf.readone(current_start, error_string);
      if (error_string != nullptr)
        return true;
    }
  }

  // find the maximum entry_end
  current_end = -1;
  for (unsigned int i = 0;  i < requested.size();  i++) {
    ClusterBuffer &buf = *requested[i];
    if (buf.entry_end() > current_end)
      current_end = buf.entry_end();
  }

  // bring all others up to at least current_end
  for (unsigned int i = 0;  i < requested.size();  i++) {
    ClusterBuffer &buf = *requested[i];
    while (buf.entry_end() < current_end) {
      buf.readone(current_start, error_string);
      if (error_string != nullptr)
        return true;
    }
  }

  return false;
}

// get a Python tuple of arrays for all buffers
PyObject* ClusterIterator::arrays() {
  // step forward, handling errors
  const char* error_string = nullptr;
  if (stepforward(error_string)) {
    if (error_string != nullptr) {
      PyErr_SetString(PyExc_IOError, error_string);
      return NULL;
    }
    else {
      PyErr_SetNone(PyExc_StopIteration);
      return NULL;
    }
  }

  // create a tuple of results
  PyObject* out = PyTuple_New(2 + requested.size());
  PyTuple_SET_ITEM(out, 0, PyLong_FromLong(current_start));
  PyTuple_SET_ITEM(out, 1, PyLong_FromLong(current_end));

  for (unsigned int i = 0;  i < requested.size();  i++) {
    ClusterBuffer &buf = *requested[i];
    const ArrayInfo &ai = arrayinfo[i];

    Long64_t numbytes;
    void* ptr = buf.getbuffer(numbytes, require_alignment, current_start, current_end);

    npy_intp dims[ai.nd];
    dims[0] = numbytes / ai.dtype->elsize;
    for (unsigned int j = 0;  j < ai.dims.size();  j++) {
      dims[j + 1] = ai.dims[j];
    }

    Py_INCREF(ai.dtype);

    PyObject* array;
    if (return_new_buffers) {
      array = PyArray_Empty(ai.nd, dims, ai.dtype, false);
      memcpy(PyArray_DATA(array), ptr, numbytes);
    }
    else {
      int flags = NPY_ARRAY_C_CONTIGUOUS;
      if ((size_t)ptr % ALIGNMENT == 0)
        flags |= NPY_ARRAY_ALIGNED;

      array = PyArray_NewFromDescr(&PyArray_Type, ai.dtype, ai.nd, dims, NULL, ptr, flags, NULL);
    }

    PyTuple_SET_ITEM(out, i + 2, array);
  }

  return out;
}

/////////////////////////////////////////////////////// utility functions

bool getfile(TFile* &file, const char* filePath) {
  file = TFile::Open(filePath);
  if (file == NULL  ||  !file->IsOpen()) {
    PyErr_Format(PyExc_IOError, "could not open file \"%s\"", filePath);
    return false;
  }
  else
    return true;
}

bool gettree(TTree* &tree, TFile* file, const char* filePath, const char* treePath) {
  file->GetObject(treePath, tree);
  if (tree == NULL) {
    PyErr_Format(PyExc_IOError, "could not read tree \"%s\" from file \"%s\"", treePath, filePath);
    return false;
  }
  else
    return true;
}

bool getbranch(TBranch* &branch, TTree* tree, const char* filePath, const char* treePath, const char* branchName) {
  branch = tree->GetBranch(branchName);
  if (branch == NULL) {
    PyErr_Format(PyExc_IOError, "could not read branch \"%s\" from tree \"%s\" from file \"%s\"", branchName, treePath, filePath);
    return false;
  }
  else
    return true;
}

const char* leaftype(TLeaf* leaf) {
  if (leaf->IsA() == TLeafO::Class()) {
    return "bool";
  }
  else if (leaf->IsA() == TLeafB::Class()  &&  leaf->IsUnsigned()) {
    return "u1";
  }
  else if (leaf->IsA() == TLeafB::Class()) {
    return "i1";
  }
  else if (leaf->IsA() == TLeafS::Class()  &&  leaf->IsUnsigned()) {
    return ">u2";
  }
  else if (leaf->IsA() == TLeafS::Class()) {
    return ">i2";
  }
  else if (leaf->IsA() == TLeafI::Class()  &&  leaf->IsUnsigned()) {
    return ">u4";
  }
  else if (leaf->IsA() == TLeafI::Class()) {
    return ">i4";
  }
  else if (leaf->IsA() == TLeafL::Class()  &&  leaf->IsUnsigned()) {
    return ">u8";
  }
  else if (leaf->IsA() == TLeafL::Class()) {
    return ">i8";
  }
  else if (leaf->IsA() == TLeafF::Class()) {
    return ">f4";
  }
  else if (leaf->IsA() == TLeafD::Class()) {
    return ">f8";
  }
  else {
    TClass* expectedClass;
    EDataType expectedType;
    leaf->GetBranch()->GetExpectedType(expectedClass, expectedType);
    switch (expectedType) {
      case kBool_t:     return "bool";
      case kUChar_t:    return "u1";
      case kchar:       return "i1";
      case kChar_t:     return "i1";
      case kUShort_t:   return ">u2";
      case kShort_t:    return ">i2";
      case kUInt_t:     return ">u4";
      case kInt_t:      return ">i4";
      case kULong_t:    return ">u8";
      case kLong_t:     return ">i8";
      case kULong64_t:  return ">u8";
      case kLong64_t:   return ">i8";
      case kFloat_t:    return ">f4";
      case kDouble32_t: return ">f4";
      case kDouble_t:   return ">f8";
      default: return NULL;
    }
  }
  return NULL;
}

void getdim(TLeaf* leaf, std::vector<int>& dims, std::vector<std::string>& counters) {
  const char* title = leaf->GetTitle();
  bool iscounter = false;

  for (const char* c = title;  *c != 0;  c++) {
    if (*c == '[') {
      dims.push_back(0);
      counters.push_back(std::string());
      iscounter = false;
    }

    else if (*c == ']') {
      if (iscounter)           // a dimension either fills int-valued dims or string-valued counters
        dims.pop_back();       // because we don't handle both
      else
        counters.pop_back();
    }

    else if (!dims.empty()) {
      if ('0' <= *c  &&  *c <= '9')
        dims.back() = dims.back() * 10 + (*c - '0');
      else
        iscounter = true;
      counters.back() = counters.back() + *c;   // accumulate any char that isn't '[' or ']'
    }

    // else this is part of the TLeaf name (before the first '[')
  }
}

bool dtypedim(PyArray_Descr* &dtype, TLeaf* leaf) {
  dtype = PyArray_DescrFromType(0);

  const char* asstring = leaftype(leaf);
  if (asstring == NULL) {
    PyErr_Format(PyExc_ValueError, "cannot convert type of TLeaf \"%s\" to Numpy", leaf->GetName());
    return false;
  }

  if (!PyArray_DescrConverter(PyUnicode_FromString(asstring), &dtype)) {
    PyErr_SetString(PyExc_ValueError, "cannot create a dtype");
    return NULL;
  }

  return true;
}

bool dtypedim_unileaf(ArrayInfo &arrayinfo, TLeaf* leaf) {
  std::vector<std::string> counters;
  getdim(leaf, arrayinfo.dims, counters);
  arrayinfo.nd = 1 + arrayinfo.dims.size();    // first dimension is for the set of entries itself
  arrayinfo.varlen = !counters.empty();

  if (arrayinfo.nd > 1  &&  arrayinfo.varlen) {
    PyErr_Format(PyExc_ValueError, "TLeaf \"%s\" has both fixed-length dimensions and variable-length dimensions", leaf->GetTitle());
    return false;
  }

  return dtypedim(arrayinfo.dtype, leaf);
}

bool dtypedim_multileaf(ArrayInfo &arrayinfo, TObjArray* leaves) {
  // TODO: Numpy recarray dtype
  PyErr_SetString(PyExc_NotImplementedError, "multileaf");
  return false;
}

bool dtypedim_branch(ArrayInfo &arrayinfo, TBranch* branch) {
  TObjArray* subbranches = branch->GetListOfBranches();
  if (subbranches->GetEntries() != 0) {
    PyErr_Format(PyExc_ValueError, "TBranch \"%s\" has subbranches; only branches of TLeaves are allowed", branch->GetName());
    return false;
  }

  TObjArray* leaves = branch->GetListOfLeaves();
  if (leaves->GetEntries() == 1)
    return dtypedim_unileaf(arrayinfo, dynamic_cast<TLeaf*>(leaves->First()));
  else
    return dtypedim_multileaf(arrayinfo, leaves);
}

const char* gettuplestring(PyObject* p, Py_ssize_t pos) {
  PyObject* obj = PyTuple_GET_ITEM(p, pos);
  if (PyString_Check(obj))
    return PyString_AsString(obj);
  else {
    PyErr_Format(PyExc_TypeError, "expected a string in argument %ld", pos);
    return NULL;
  }
}

/////////////////////////////////////////////////////// Python module

typedef struct {
  PyObject_HEAD
  ClusterIterator* iter;
} PyClusterIterator;

static PyObject* PyClusterIterator_iter(PyObject* self);
static PyObject* PyClusterIterator_next(PyObject* self);

static PyObject* iterate(PyObject* self, PyObject* args, PyObject* kwds);

#if PY_MAJOR_VERSION >= 3
#define Py_TPFLAGS_HAVE_ITER 0
#endif

static PyTypeObject PyClusterIteratorType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "numpyinterface.ClusterIterator", /*tp_name*/
  sizeof(PyClusterIterator),  /*tp_basicsize*/
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
  PyClusterIterator_iter, /* tp_iter: __iter__() method */
  PyClusterIterator_next  /* tp_iternext: __next__() method */
};

static PyMethodDef module_methods[] = {
  {"iterate", (PyCFunction)iterate, METH_VARARGS | METH_KEYWORDS, "Get an iterator over a selected set of TTree branches, yielding a tuple of (entry_start, entry_end, *arrays) for each cluster.\n\nPositional arguments:\n\n    filePath (str): name of the TFile\n    treePath (str): name of the TTree\n    *branchNames (strs): name of requested branches\n\nAlternative positional arguments:\n\n    *branches (PyROOT TBranch objects): to avoid re-opening the file (FIXME: not implemented yet!).\n\nKeyword arguments:\n\n    return_new_buffers=False:\n        if True, new memory is allocated during iteration for the arrays, and it is safe to use the arrays after the iterator steps;\n        if False, arrays merely wrap internal memory buffers that may be reused or deleted after the iterator steps: provides higher performance during iteration, but may result in stale data or segmentation faults if array data are accessed after the iterator steps\n\n    require_alignment=False:\n        if True, guarantee that the data are aligned in memory, even if that means copying data internally;\n        if False, smaller chance of internal memory copy, but array data may start at any memory address, possibly thwarting vectorized processing of the array\n        (ignored if return_new_buffers is True because Numpy arrays do their own alignment)"},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

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
  PyClusterIteratorType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyClusterIteratorType) < 0)
    return NULL;

  PyObject* module = PyModule_Create(&moduledef);
  if (module == NULL)
    return NULL;

  import_array();

  Py_INCREF(&PyClusterIteratorType);
  PyModule_AddObject(module, "ClusterIterator", reinterpret_cast<PyObject*>(&PyClusterIteratorType));

  return module;
}

#else // PY_MAJOR_VERSION <= 2

PyMODINIT_FUNC initnumpyinterface(void) {
  PyClusterIteratorType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyClusterIteratorType) < 0)
    return;

  PyObject* module = Py_InitModule3("numpyinterface", module_methods, "");
  if (module == NULL)
    return;

  Py_INCREF(&PyClusterIteratorType);
  PyModule_AddObject(module, "ClusterIterator", reinterpret_cast<PyObject*>(&PyClusterIteratorType));

  if (module != NULL)
    import_array();
}

#endif

/////////////////////////////////////////////////////// Python functions

static PyObject* PyClusterIterator_iter(PyObject* self) {
  PyClusterIterator* thyself = reinterpret_cast<PyClusterIterator*>(self);
  thyself->iter->reset();
  Py_INCREF(self);
  return self;
}

static PyObject* PyClusterIterator_next(PyObject* self) {
  PyClusterIterator* thyself = reinterpret_cast<PyClusterIterator*>(self);
  return thyself->iter->arrays();
}

static PyObject* iterate(PyObject* self, PyObject* args, PyObject* kwds) {
  std::vector<TBranch*> requested_branches;
  bool return_new_buffers = false;
  bool require_alignment = false;

  if (PyTuple_GET_SIZE(args) < 1) {
    PyErr_SetString(PyExc_TypeError, "at least one argument is required");
    return NULL;
  }

  if (PyString_Check(PyTuple_GET_ITEM(args, 0))) {
    // first argument is a string: filePath, treePath, branchNames... signature

    // first two arguments are filePath and treePath, and then there must be at least one branchName
    if (PyTuple_GET_SIZE(args) < 3) {
      PyErr_SetString(PyExc_TypeError, "in the string-based signture, at least three arguments are required");
      return NULL;
    }

    const char* filePath = gettuplestring(args, 0);
    const char* treePath = gettuplestring(args, 1);
    if (filePath == NULL  ||  treePath == NULL)
      return NULL;

    TFile* file;
    if (!getfile(file, filePath)) return NULL;

    TTree* tree;
    if (!gettree(tree, file, filePath, treePath)) return NULL;

    for (int i = 2;  i < PyTuple_GET_SIZE(args);  i++) {
      const char* branchName = gettuplestring(args, i);
      TBranch* branch;
      if (!getbranch(branch, tree, filePath, treePath, branchName)) return NULL;
      requested_branches.push_back(branch);
    }

    if (kwds != NULL) {
      PyObject* key;
      PyObject* value;
      Py_ssize_t pos = 0;
      while (PyDict_Next(kwds, &pos, &key, &value)) {
        if (std::string(PyString_AsString(key)) == std::string("return_new_buffers")) {
          if (PyObject_IsTrue(value))
            return_new_buffers = true;
          else
            return_new_buffers = false;
        }

        else if (std::string(PyString_AsString(key)) == std::string("require_alignment")) {
          if (PyObject_IsTrue(value))
            require_alignment = true;
          else
            require_alignment = false;
        }

        else {
          PyErr_Format(PyExc_TypeError, "unrecognized option: %s", PyString_AsString(key));
          return NULL;
        }
      }
    }
  }

  else {
    // first argument is an object: TBranch, TBranch, TBranch... signature
    // TODO: insist that all branches come from the same TTree
    PyErr_SetString(PyExc_NotImplementedError, "FIXME: accept PyROOT TBranches");
    return NULL;
  }

  std::vector<TBranch*> unrequested_counters;
  std::vector<ArrayInfo> arrayinfo;
  Long64_t num_entries = requested_branches.back()->GetTree()->GetEntries();

  for (unsigned int i = 0;  i < requested_branches.size();  i++) {
    arrayinfo.push_back(ArrayInfo());
    if (!dtypedim_branch(arrayinfo.back(), requested_branches[i]))
      return NULL;
  }

  PyClusterIterator* out = PyObject_New(PyClusterIterator, &PyClusterIteratorType);

  if (!PyObject_Init(reinterpret_cast<PyObject*>(out), &PyClusterIteratorType)) {
    Py_DECREF(out);
    return NULL;
  }

  out->iter = new ClusterIterator(requested_branches, unrequested_counters, arrayinfo, num_entries, return_new_buffers, require_alignment);

  return reinterpret_cast<PyObject*>(out);
}
