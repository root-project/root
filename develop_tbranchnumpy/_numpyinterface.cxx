#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <iostream>

#include <string>
#include <vector>

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

#define IS_ALIGNED(ptr) ((size_t)ptr % 8 == 0)     // FIXME: is there a better way to check for alignment?

// performance counters for diagnostics
static Long64_t perf_baskets_loaded = 0;
static Long64_t perf_bytes_loaded = 0;
static Long64_t perf_baskets_copied_to_extra = 0;
static Long64_t perf_bytes_copied_to_extra = 0;
static Long64_t perf_bytes_moved_in_extra = 0;
static Long64_t perf_extra_allocations = 0;
static Long64_t perf_clusters_copied_to_arrays = 0;
static Long64_t perf_bytes_copied_to_arrays = 0;

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
  const bool swap_bytes;
  TBufferFile bf;
  std::vector<char> extra;
  void* oldextra;
  bool usingextra;

  // always numbers of entries (not bytes) and always inclusive on start, exclusive on end (like Python)
  // also, the TBufferFile is always ahead of the extra buffer and there's no gap between them
  Long64_t bf_entry_start;
  Long64_t bf_entry_end;
  Long64_t ex_entry_start;
  Long64_t ex_entry_end;

  void copy_to_extra(Long64_t keep_start);

  inline void check_extra_allocations() {
    if (oldextra != reinterpret_cast<void*>(extra.data())) {
      oldextra = reinterpret_cast<void*>(extra.data());
      perf_extra_allocations++;
    }
  }

public:
  ClusterBuffer(const TBranch* branch, const Long64_t itemsize, const bool swap_bytes) :
    branch(branch), itemsize(itemsize), swap_bytes(swap_bytes), bf(TBuffer::kWrite, 32*1024), oldextra(nullptr), usingextra(false),
    bf_entry_start(0), bf_entry_end(0), ex_entry_start(0), ex_entry_end(0)
  {
    check_extra_allocations();
  }

  void readone(Long64_t keep_start, const char* &error_string);
  void* getbuffer(Long64_t &numbytes, Long64_t entry_start, Long64_t entry_end);

  Long64_t entry_end() {
    return bf_entry_end;  // hide the distinction between bf and extra
  }

  void reset() {
    extra.clear();
    bf_entry_start = 0;
    bf_entry_end = 0;
    ex_entry_start = 0;
    ex_entry_end = 0;
  }
};

class ClusterIterator {
private:
  std::vector<std::unique_ptr<ClusterBuffer>> requested;
  std::vector<std::unique_ptr<ClusterBuffer>> extra_counters;
  const std::vector<ArrayInfo> arrayinfo;   // has the same length as requested
  const Long64_t num_entries;
  const bool return_new_buffers;
  Long64_t current_start;
  Long64_t current_end;

  bool stepforward(const char* &error_string);

public:
  ClusterIterator(const std::vector<TBranch*> &requested_branches, const std::vector<TBranch*> &unrequested_counters, const std::vector<ArrayInfo> arrayinfo, Long64_t num_entries, bool return_new_buffers, bool swap_bytes) :
    arrayinfo(arrayinfo), num_entries(num_entries), return_new_buffers(return_new_buffers), current_start(0), current_end(0) {
    for (unsigned int i = 0;  i < arrayinfo.size();  i++)
      requested.push_back(std::unique_ptr<ClusterBuffer>(new ClusterBuffer(requested_branches[i], arrayinfo[i].dtype->elsize, swap_bytes)));
  }

  PyObject* arrays();

  void reset() {
    current_start = 0;
    current_end = 0;
    for (unsigned int i = 0;  i < requested.size();  i++) {
      requested[i]->reset();
    }
    for (unsigned int i = 0;  i < extra_counters.size();  i++) {
      extra_counters[i]->reset();
    }
  }
};    

void ClusterBuffer::copy_to_extra(Long64_t keep_start) {
  // this is a safer algorithm than is necessary, and it could impact performance, but significantly less so than the other speed-ups

  // remove data from the start of the extra buffer to keep it from growing too much
  if (ex_entry_start < keep_start) {
    const Long64_t offset = (keep_start - ex_entry_start) * itemsize;
    const Long64_t newsize = (ex_entry_end - keep_start) * itemsize;

    memmove(extra.data(), &extra.data()[offset], newsize);
    perf_bytes_moved_in_extra += newsize;

    extra.resize(newsize);
    check_extra_allocations();

    ex_entry_start = keep_start;
  }

  // append the BufferFile at the end of the extra buffer
  const Long64_t oldsize = extra.size();
  const Long64_t additional = (bf_entry_end - bf_entry_start) * itemsize;

  if (additional > 0) {
    extra.resize(oldsize + additional);
    check_extra_allocations();

    memcpy(&extra.data()[oldsize], bf.GetCurrent(), additional);
    perf_baskets_copied_to_extra++;
    perf_bytes_copied_to_extra += additional;

    ex_entry_end = bf_entry_end;
  }
}

// readone asks ROOT to read one basket from the file
// and ClusterBuffer ensures that entries as old as keep_start are preserved
void ClusterBuffer::readone(Long64_t keep_start, const char* &error_string) {
  if (!usingextra  &&  bf_entry_end > keep_start) {
    // need to overwrite the TBufferFile before we're done with it, so we need to start using extra now
    copy_to_extra(0);
    usingextra = true;
  }

  // read in one more basket, starting at the old bf_entry_end
  Long64_t numentries = branch->GetBulkRead().GetEntriesSerialized(bf_entry_end, bf);
  perf_baskets_loaded++;
  perf_bytes_loaded += numentries * itemsize;

  if (swap_bytes) {
    switch (itemsize) {
      case 8:
        {
          Long64_t* buffer64 = reinterpret_cast<Long64_t*>(bf.GetCurrent());
          for (Long64_t i = 0;  i < numentries;  i++)
            buffer64[i] = __builtin_bswap64(buffer64[i]);
          break;
        }

      case 4:
        {
          Int_t* buffer32 = reinterpret_cast<Int_t*>(bf.GetCurrent());
          for (Long64_t i = 0;  i < numentries;  i++)
            buffer32[i] = __builtin_bswap32(buffer32[i]);
          break;
        }

      case 2:
        {
          Short_t* buffer16 = reinterpret_cast<Short_t*>(bf.GetCurrent());
          for (Long64_t i = 0;  i < numentries;  i++)
            buffer16[i] = __builtin_bswap16(buffer16[i]);
          break;
        }

      default:
        error_string = "illegal itemsize";
        return;
    }
  }

  // update the range
  bf_entry_start = bf_entry_end;
  bf_entry_end = bf_entry_start + numentries;

  // check for errors
  if (numentries <= 0) {
    bf_entry_end = bf_entry_start;
    error_string = "failed to read TBasket into TBufferFile (using GetBulkRead().GetEntriesSerialized)";
  }

  // for now, always mirror to the extra buffer
  if (usingextra)
    copy_to_extra(keep_start);
}

// getbuffer returns a pointer to contiguous data with its size
// if you're lucky (and ask for it), this is performed without any copies
void* ClusterBuffer::getbuffer(Long64_t &numbytes, Long64_t entry_start, Long64_t entry_end) {
  numbytes = (entry_end - entry_start) * itemsize;

  if (usingextra) {
    const Long64_t offset = (entry_start - ex_entry_start) * itemsize;
    return &extra.data()[offset];
  }
  else {
    const Long64_t offset = (entry_start - bf_entry_start) * itemsize;
    return &bf.GetCurrent()[offset];
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
    void* ptr = buf.getbuffer(numbytes, current_start, current_end);

    npy_intp dims[ai.nd];
    dims[0] = numbytes / ai.dtype->elsize;
    for (unsigned int j = 0;  j < ai.dims.size();  j++) {
      dims[j + 1] = ai.dims[j];
    }

    Py_INCREF(ai.dtype);

    PyObject* array;
    if (return_new_buffers) {
      array = PyArray_Empty(ai.nd, dims, ai.dtype, false);
      memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)), ptr, numbytes);
      perf_clusters_copied_to_arrays++;
      perf_bytes_copied_to_arrays += numbytes;
    }
    else {
      int flags = NPY_ARRAY_C_CONTIGUOUS;
      if (!IS_ALIGNED(ptr))
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

const char* leaftype(TLeaf* leaf, bool swap_bytes) {
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
    return swap_bytes ? "<u2" : ">u2";
  }
  else if (leaf->IsA() == TLeafS::Class()) {
    return swap_bytes ? "<i2" : ">i2";
  }
  else if (leaf->IsA() == TLeafI::Class()  &&  leaf->IsUnsigned()) {
    return swap_bytes ? "<u4" : ">u4";
  }
  else if (leaf->IsA() == TLeafI::Class()) {
    return swap_bytes ? "<i4" : ">i4";
  }
  else if (leaf->IsA() == TLeafL::Class()  &&  leaf->IsUnsigned()) {
    return swap_bytes ? "<u8" : ">u8";
  }
  else if (leaf->IsA() == TLeafL::Class()) {
    return swap_bytes ? "<i8" : ">i8";
  }
  else if (leaf->IsA() == TLeafF::Class()) {
    return swap_bytes ? "<f4" : ">f4";
  }
  else if (leaf->IsA() == TLeafD::Class()) {
    return swap_bytes ? "<f8" : ">f8";
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
      case kUShort_t:   return swap_bytes ? "<u2" : ">u2";
      case kShort_t:    return swap_bytes ? "<i2" : ">i2";
      case kUInt_t:     return swap_bytes ? "<u4" : ">u4";
      case kInt_t:      return swap_bytes ? "<i4" : ">i4";
      case kULong_t:    return swap_bytes ? "<u8" : ">u8";
      case kLong_t:     return swap_bytes ? "<i8" : ">i8";
      case kULong64_t:  return swap_bytes ? "<u8" : ">u8";
      case kLong64_t:   return swap_bytes ? "<i8" : ">i8";
      case kFloat_t:    return swap_bytes ? "<f4" : ">f4";
      case kDouble32_t: return swap_bytes ? "<f4" : ">f4";
      case kDouble_t:   return swap_bytes ? "<f8" : ">f8";
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

bool dtypedim(PyArray_Descr* &dtype, TLeaf* leaf, bool swap_bytes) {
  dtype = PyArray_DescrFromType(0);

  const char* asstring = leaftype(leaf, swap_bytes);
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

bool dtypedim_unileaf(ArrayInfo &arrayinfo, TLeaf* leaf, bool swap_bytes) {
  std::vector<std::string> counters;
  getdim(leaf, arrayinfo.dims, counters);
  arrayinfo.nd = 1 + arrayinfo.dims.size();    // first dimension is for the set of entries itself
  arrayinfo.varlen = !counters.empty();

  if (arrayinfo.nd > 1  &&  arrayinfo.varlen) {
    PyErr_Format(PyExc_ValueError, "TLeaf \"%s\" has both fixed-length dimensions and variable-length dimensions", leaf->GetTitle());
    return false;
  }

  return dtypedim(arrayinfo.dtype, leaf, swap_bytes);
}

bool dtypedim_multileaf(ArrayInfo &arrayinfo, TObjArray* leaves, bool swap_bytes) {
  // TODO: Numpy recarray dtype
  PyErr_SetString(PyExc_NotImplementedError, "multileaf");
  return false;
}

bool dtypedim_branch(ArrayInfo &arrayinfo, TBranch* branch, bool swap_bytes) {
  TObjArray* subbranches = branch->GetListOfBranches();
  if (subbranches->GetEntries() != 0) {
    PyErr_Format(PyExc_ValueError, "TBranch \"%s\" has subbranches; only branches of TLeaves are allowed", branch->GetName());
    return false;
  }

  TObjArray* leaves = branch->GetListOfLeaves();
  if (leaves->GetEntries() == 1)
    return dtypedim_unileaf(arrayinfo, dynamic_cast<TLeaf*>(leaves->First()), swap_bytes);
  else
    return dtypedim_multileaf(arrayinfo, leaves, swap_bytes);
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
static void PyClusterIterator_del(PyClusterIterator* self);

static PyObject* iterate(PyObject* self, PyObject* args, PyObject* kwds);
static PyObject* dtypeshape(PyObject* self, PyObject* args, PyObject* kwds);
static PyObject* performance(PyObject* self);

#if PY_MAJOR_VERSION >= 3
#define Py_TPFLAGS_HAVE_ITER 0
#endif

static PyTypeObject PyClusterIteratorType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "numpyinterface.ClusterIterator", /*tp_name*/
  sizeof(PyClusterIterator),  /*tp_basicsize*/
  0,                         /*tp_itemsize*/
  (destructor)PyClusterIterator_del, /*tp_dealloc*/
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
  {"iterate", (PyCFunction)iterate, METH_VARARGS | METH_KEYWORDS, "Get an iterator over a selected set of TTree branches, yielding a tuple of (entry_start, entry_end, *arrays) for each cluster.\n\nPositional arguments:\n\n    filePath (str): name of the TFile\n    treePath (str): name of the TTree\n    *branchNames (strs): name of requested branches\n\nAlternative positional arguments:\n\n    *branches (PyROOT TBranch objects): to avoid re-opening the file (FIXME: not implemented yet!).\n\nKeyword arguments:\n\n    return_new_buffers=True:\n        if True, new memory is allocated during iteration for the arrays, and it is safe to use the arrays after the iterator steps;\n        if False, arrays merely wrap internal memory buffers that may be reused or deleted after the iterator steps: provides higher performance during iteration, but may result in stale data or segmentation faults if array data are accessed after the iterator steps\n\n    swap_bytes=True:\n        if True, swap bytes while reading and produce a little-endian Numpy array;\n        if False, return data as-is and produce a big-endian Numpy array"},
  {"dtypeshape", (PyCFunction)dtypeshape, METH_VARARGS | METH_KEYWORDS, "Returns a tuple of (name, dtype, shape) for all provided TTree branches, where the first element of 'shape' is a slight overestimate of the array size (since it includes headers) and TLeaf dimensions are its subsequent elements.\n\nPositional arguments:\n\n    filePath (str): name of the TFile\n    treePath (str): name of the TTree\n    *branchNames (strs): name of requested branches\n\nAlternative positional arguments:\n\n    *branches (PyROOT TBranch objects): to avoid re-opening the file (FIXME: not implemented yet!).\n\nKeyword arguments:\n\n    swap_bytes=True:\n        if True, swap bytes while reading and produce a little-endian Numpy array;\n        if False, return data as-is and produce a big-endian Numpy array"},
  {"performance", (PyCFunction)performance, METH_NOARGS, "Get a dictionary of performance counters:\n\n    \"baskets-loaded\": number of baskets loaded from the ROOT file; merely indicates how much was read\n    \"bytes-loaded\": same, but counting bytes\n    \"baskets-copied-to-extra\": number of baskets that had to be copied to an internal buffer; ideally zero, hard to achieve in practice\n    \"bytes-copied-to-extra\": same, but counting bytes\n    \"extra-allocations\": number of times the internal buffer had to be reallocated to allow for a new high water mark in internal memory use; this should not scale with the total data read, but should quickly reach some plateau\n    \"clusters-copied-to-arrays\": number of internal buffers copied to new arrays to satisfy the user's choice (return_new_buffers is True)\n    same, but counting bytes"},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "_numpyinterface",
  NULL,
  0,
  module_methods,
  NULL,
  NULL,
  NULL,
  NULL
};

PyMODINIT_FUNC PyInit__numpyinterface(void) {
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

PyMODINIT_FUNC init_numpyinterface(void) {
  PyClusterIteratorType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyClusterIteratorType) < 0)
    return;

  PyObject* module = Py_InitModule3("_numpyinterface", module_methods, "");
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

static void PyClusterIterator_del(PyClusterIterator* thyself) {
  delete thyself->iter;
  Py_TYPE(thyself)->tp_free(reinterpret_cast<PyObject*>(thyself));
}

bool getbranches(PyObject* args, std::vector<TBranch*> &requested_branches) {
  if (PyTuple_GET_SIZE(args) < 1) {
    PyErr_SetString(PyExc_TypeError, "at least one argument is required");
    return false;
  }

  if (PyString_Check(PyTuple_GET_ITEM(args, 0))) {
    // first argument is a string: filePath, treePath, branchNames... signature

    // first two arguments are filePath and treePath, and then there must be at least one branchName
    if (PyTuple_GET_SIZE(args) < 3) {
      PyErr_SetString(PyExc_TypeError, "in the string-based signture, at least three arguments are required");
      return false;
    }

    const char* filePath = gettuplestring(args, 0);
    const char* treePath = gettuplestring(args, 1);
    if (filePath == NULL  ||  treePath == NULL)
      return false;

    TFile* file;
    if (!getfile(file, filePath)) return false;

    TTree* tree;
    if (!gettree(tree, file, filePath, treePath)) return false;

    for (int i = 2;  i < PyTuple_GET_SIZE(args);  i++) {
      const char* branchName = gettuplestring(args, i);
      TBranch* branch;
      if (!getbranch(branch, tree, filePath, treePath, branchName)) return false;
      requested_branches.push_back(branch);
    }
  }

  else {
    // first argument is an object: TBranch, TBranch, TBranch... signature
    // TODO: insist that all branches come from the same TTree
    PyErr_SetString(PyExc_NotImplementedError, "FIXME: accept PyROOT TBranches");
    return false;
  }

  return true;
}

static PyObject* iterate(PyObject* self, PyObject* args, PyObject* kwds) {
  std::vector<TBranch*> requested_branches;
  if (!getbranches(args, requested_branches))
    return NULL;

  bool return_new_buffers = true;
  bool swap_bytes = true;

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

      else if (std::string(PyString_AsString(key)) == std::string("swap_bytes")) {
        if (PyObject_IsTrue(value))
          swap_bytes = true;
        else
          swap_bytes = false;
      }

      else {
        PyErr_Format(PyExc_TypeError, "unrecognized option: %s", PyString_AsString(key));
        return NULL;
      }
    }
  }

  std::vector<TBranch*> unrequested_counters;
  std::vector<ArrayInfo> arrayinfo;

  for (unsigned int i = 0;  i < requested_branches.size();  i++) {
    arrayinfo.push_back(ArrayInfo());
    if (!dtypedim_branch(arrayinfo.back(), requested_branches[i], swap_bytes))
      return NULL;
  }

  PyClusterIterator* out = PyObject_New(PyClusterIterator, &PyClusterIteratorType);

  if (!PyObject_Init(reinterpret_cast<PyObject*>(out), &PyClusterIteratorType)) {
    Py_DECREF(out);
    return NULL;
  }

  Long64_t num_entries = requested_branches.back()->GetTree()->GetEntries();
  out->iter = new ClusterIterator(requested_branches, unrequested_counters, arrayinfo, num_entries, return_new_buffers, swap_bytes);

  return reinterpret_cast<PyObject*>(out);
}

static PyObject* dtypeshape(PyObject* self, PyObject* args, PyObject* kwds) {
  std::vector<TBranch*> requested_branches;
  if (!getbranches(args, requested_branches))
    return NULL;

  bool swap_bytes = true;

  if (kwds != NULL) {
    PyObject* key;
    PyObject* value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(kwds, &pos, &key, &value)) {
      if (std::string(PyString_AsString(key)) == std::string("swap_bytes")) {
        if (PyObject_IsTrue(value))
          swap_bytes = true;
        else
          swap_bytes = false;
      }

      else {
        PyErr_Format(PyExc_TypeError, "unrecognized option: %s", PyString_AsString(key));
        return NULL;
      }
    }
  }

  PyObject* out = PyTuple_New(requested_branches.size());

  for (unsigned int i = 0;  i < requested_branches.size();  i++) {
    ArrayInfo arrayinfo;
    if (!dtypedim_branch(arrayinfo, requested_branches[i], swap_bytes)) {
      Py_DECREF(out);
      return NULL;
    }

    PyObject* shape = PyTuple_New(arrayinfo.nd);
    PyTuple_SET_ITEM(shape, 0, PyLong_FromLong((int)ceil(1.0 * requested_branches[i]->GetTotalSize() / arrayinfo.dtype->elsize)));

    for (unsigned int j = 0;  j < arrayinfo.dims.size();  j++)
      PyTuple_SET_ITEM(shape, j + 1, PyLong_FromLong(arrayinfo.dims[j]));

    PyObject* triple = PyTuple_New(3);
    Py_INCREF(arrayinfo.dtype);
    PyTuple_SET_ITEM(triple, 0, PyUnicode_FromString(requested_branches[i]->GetName()));
    PyTuple_SET_ITEM(triple, 1, reinterpret_cast<PyObject*>(arrayinfo.dtype));
    PyTuple_SET_ITEM(triple, 2, shape);

    PyTuple_SET_ITEM(out, i, triple);
  }

  return out;
}

static PyObject* performance(PyObject* self) {
  PyObject* out = PyDict_New();
  PyDict_SetItemString(out, "baskets-loaded",            PyLong_FromLong(perf_baskets_loaded           ));
  PyDict_SetItemString(out, "bytes-loaded",              PyLong_FromLong(perf_bytes_loaded             ));
  PyDict_SetItemString(out, "baskets-copied-to-extra",   PyLong_FromLong(perf_baskets_copied_to_extra  ));
  PyDict_SetItemString(out, "bytes-copied-to-extra",     PyLong_FromLong(perf_bytes_copied_to_extra    ));
  PyDict_SetItemString(out, "bytes-moved-in-extra",      PyLong_FromLong(perf_bytes_moved_in_extra     ));
  PyDict_SetItemString(out, "extra-allocations",         PyLong_FromLong(perf_extra_allocations        ));
  PyDict_SetItemString(out, "clusters-copied-to-arrays", PyLong_FromLong(perf_clusters_copied_to_arrays));
  PyDict_SetItemString(out, "bytes-copied-to-arrays",    PyLong_FromLong(perf_bytes_copied_to_arrays   ));
  return out;
}
