#include <iostream>
#include <vector>
#include <string>

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

static char module_docstring[] = "Tools for quickly loading TBranch contents into Numpy arrays.";
static char clusters_docstring[] = "Returns a list of the first entry in each cluster followed by the total number of entries (the length of this list is the number of clusters + 1).";
static char declare_docstring[] = "Declares an array (or set of arrays) that can be filled with a given TBranch.";
static char fill_docstring[] = "Fills a given array (or set of arrays) with TBranch data.";

static PyObject* clusters(PyObject* self, PyObject* args);
static PyObject* declare(PyObject* self, PyObject* args);
static PyObject* fill(PyObject* self, PyObject* args);

static PyMethodDef module_methods[] = {
  {"clusters", (PyCFunction)clusters, METH_VARARGS, clusters_docstring},
  {"declare", (PyCFunction)declare, METH_VARARGS, declare_docstring},
  {"fill", (PyCFunction)fill, METH_VARARGS, fill_docstring},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "tbranchnumpy",
  NULL,
  0,
  module_methods,
  NULL,
  NULL,
  NULL,
  NULL
};

PyMODINIT_FUNC PyInit_tbranchnumpy(void) {
  PyObject* module = PyModule_Create(&moduledef);
  if (module != NULL)
    import_array();
  return module;
}
#else
PyMODINIT_FUNC inittbranchnumpy(void) {
  PyObject* module = Py_InitModule3("tbranchnumpy", module_methods, module_docstring);
  if (module != NULL)
    import_array();
}
#endif

/////////////////////////////////////////////////////// utils

bool getfile(TFile* &file, char* filePath) {
  file = TFile::Open(filePath);
  if (file == NULL  ||  !file->IsOpen()) {
    PyErr_Format(PyExc_IOError, "could not open file \"%s\"", filePath);
    return false;
  }
  else
    return true;
}

bool gettree(TTree* &tree, TFile* file, char* filePath, char* treePath) {
  file->GetObject(treePath, tree);
  if (tree == NULL) {
    PyErr_Format(PyExc_IOError, "could not read tree \"%s\" from file \"%s\"", treePath, filePath);
    return false;
  }
  else
    return true;
}

bool getbranch(TBranch* &branch, TTree* tree, char* filePath, char* treePath, char* branchName) {
  branch = tree->GetBranch(branchName);
  if (branch == NULL) {
    PyErr_Format(PyExc_IOError, "could not read branch \"%s\" from tree \"%s\" from file \"%s\"", branchName, treePath, filePath);
    return false;
  }
  else
    return true;
}

bool leaftype(TLeaf* leaf, const char* &dtype, Long64_t &size) {
  if (leaf->IsA() == TLeafO::Class()) {
    dtype = "bool"; size = 1; return true;
  }
  else if (leaf->IsA() == TLeafB::Class()  &&  leaf->IsUnsigned()) {
    dtype = "u1";   size = 1; return true;
  }
  else if (leaf->IsA() == TLeafB::Class()) {
    dtype = "i1";   size = 1; return true;
  }
  else if (leaf->IsA() == TLeafS::Class()  &&  leaf->IsUnsigned()) {
    dtype = ">u2";  size = 2; return true;
  }
  else if (leaf->IsA() == TLeafS::Class()) {
    dtype = ">i2";  size = 2; return true;
  }
  else if (leaf->IsA() == TLeafI::Class()  &&  leaf->IsUnsigned()) {
    dtype = ">u4";  size = 4; return true;
  }
  else if (leaf->IsA() == TLeafI::Class()) {
    dtype = ">i4";  size = 4; return true;
  }
  else if (leaf->IsA() == TLeafL::Class()  &&  leaf->IsUnsigned()) {
    dtype = ">u8";  size = 8; return true;
  }
  else if (leaf->IsA() == TLeafL::Class()) {
    dtype = ">i8";  size = 8; return true;
  }
  else if (leaf->IsA() == TLeafF::Class()) {
    dtype = ">f4";  size = 4; return true;
  }
  else if (leaf->IsA() == TLeafD::Class()) {
    dtype = ">f8";  size = 8; return true;
  }
  else {
    TClass* expectedClass;
    EDataType expectedType;
    leaf->GetBranch()->GetExpectedType(expectedClass, expectedType);
    switch (expectedType) {
      case kBool_t:     dtype = "bool"; size = 1; return true;
      case kUChar_t:    dtype = "u1";   size = 1; return true;
      case kchar:       dtype = "i1";   size = 1; return true;
      case kChar_t:     dtype = "i1";   size = 1; return true;
      case kUShort_t:   dtype = ">u2";  size = 2; return true;
      case kShort_t:    dtype = ">i2";  size = 2; return true;
      case kUInt_t:     dtype = ">u4";  size = 4; return true;
      case kInt_t:      dtype = ">i4";  size = 4; return true;
      case kULong_t:    dtype = ">u8";  size = 8; return true;
      case kLong_t:     dtype = ">i8";  size = 8; return true;
      case kULong64_t:  dtype = ">u8";  size = 8; return true;
      case kLong64_t:   dtype = ">i8";  size = 8; return true;
      case kFloat_t:    dtype = ">f4";  size = 4; return true;
      case kDouble32_t: dtype = ">f4";  size = 4; return true;
      case kDouble_t:   dtype = ">f8";  size = 8; return true;
    }
  }
  return false;
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
      if (!iscounter)
        counters.back() = std::string();        // clear whatever might have accumulated
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

bool has_varlen(std::vector<std::string>& counters) {
  for (auto counter = counters.begin();  counter != counters.end();  ++counter) {
    if (*counter != std::string())
      return true;
  }
  return false;
}

PyObject* makearray(int nd, npy_intp* dims, PyObject* spec) {
  PyArray_Descr* dtype = PyArray_DescrFromType(0);

  if (!PyArray_DescrConverter(spec, &dtype)) {
    PyErr_SetString(PyExc_ValueError, "cannot create a dtype");
    return NULL;
  }

  return PyArray_Empty(nd, dims, dtype, 0);
}

PyObject* makearray(int nd, npy_intp* dims, const char* spec) {
  return makearray(nd, dims, PyUnicode_FromString(spec));
}

PyObject* makearray(npy_intp length, const char* spec) {
  npy_intp dims[1];
  dims[0] = length;
  return makearray(1, dims, PyUnicode_FromString(spec));
}

/////////////////////////////////////////////////////// info functions

PyObject* clusters_impl(TTree* tree) {
  TTree::TClusterIterator iter = tree->GetClusterIterator(0);

  PyObject* list = PyList_New(0);

  Long64_t i = iter();
  PyList_Append(list, PyInt_FromLong(i));
  do {
    i = iter.Next();
    PyList_Append(list, PyInt_FromLong(i));
  } while (i < tree->GetEntries());

  return list;
}

static PyObject* clusters(PyObject* self, PyObject* args) {
  char* filePath;
  char* treePath;
  // PyObject* pyroot_ttree;

  if (PyArg_ParseTuple(args, "ss", &filePath, &treePath)) {
    TFile* file;
    if (!getfile(file, filePath)) return NULL;

    TTree* tree;
    if (!gettree(tree, file, filePath, treePath)) return NULL;

    return clusters_impl(tree);
  }

  // TODO: check PyArg_ParseTuple(args, "O", &pyroot_ttree) for a PyROOT ttree
  // else if () { }

  PyErr_SetString(PyExc_TypeError, "either supply\n    filePath (str), treePath (str)\nor\n    tbranch (PyROOT)");
  return NULL;
}

/////////////////////////////////////////////////////// declare variants

PyObject* declare_flat(TBranch* branch, TLeaf* leaf, bool wantCounter) {
  const char* dtype;
  Long64_t size;
  if (!leaftype(leaf, dtype, size)) {
    PyErr_Format(PyExc_TypeError, "the TLeaf \"%s\" cannot be converted to Numpy", leaf->GetName());
    return NULL;
  }
  return makearray(branch->GetTotBytes() / size, dtype);
}

PyObject* declare_fixedlen(TBranch* branch, TLeaf* leaf, std::vector<int> &dims, bool wantCounter) {
  // TODO: declare a multidimensional Numpy array
  PyErr_SetString(PyExc_NotImplementedError, "fixedlen");
  return NULL;
}

PyObject* declare_varlen(TBranch* branch, TLeaf* leaf, std::vector<int> &dims, std::vector<std::string> &counters, bool wantCounter) {
  // TODO: declare a single Numpy array for the data (if !wantCounter) or a 2-tuple of arrays for data and counter
  PyErr_SetString(PyExc_NotImplementedError, "varlen");
  return NULL;
}

PyObject* declare_unileaf(TBranch* branch, TLeaf* leaf, bool wantCounter) {
  std::vector<int> dims;
  std::vector<std::string> counters;
  getdim(leaf, dims, counters);

  if (has_varlen(counters))
    return declare_varlen(branch, leaf, dims, counters, wantCounter);
  else if (!dims.empty())
    return declare_fixedlen(branch, leaf, dims, wantCounter);
  else
    return declare_flat(branch, leaf, wantCounter);
}

PyObject* declare_multileaf(TBranch* branch, TObjArray* leaves, bool wantCounter) {
  // TODO: declare a Numpy recarray
  PyErr_SetString(PyExc_NotImplementedError, "multileaf");
  return NULL;
}

PyObject* declare_unibranch(TBranch* branch, bool wantCounter) {
  TObjArray* leaves = branch->GetListOfLeaves();
  if (leaves->GetEntries() == 1)
    return declare_unileaf(branch, dynamic_cast<TLeaf*>(leaves->First()), wantCounter);
  else
    return declare_multileaf(branch, leaves, wantCounter);
}

PyObject* declare_multibranch(TObjArray* branches, bool wantCounter) {
  // TODO: declare a Python dict of arrays
  PyErr_SetString(PyExc_NotImplementedError, "multibranch");
  return NULL;
}

PyObject* declare_branch(TBranch* branch, bool wantCounter) {
  TObjArray* subbranches = branch->GetListOfBranches();
  if (subbranches->GetEntries() == 0)
    return declare_unibranch(branch, wantCounter);
  else
    return declare_multibranch(subbranches, wantCounter);
}

static PyObject* declare(PyObject* self, PyObject* args) {
  char* filePath;
  char* treePath;
  char* branchName;
  // PyObject* pyroot_tbranch;
  PyObject* wantCounter = NULL;

  if (PyArg_ParseTuple(args, "sss|O", &filePath, &treePath, &branchName, &wantCounter)) {
    TFile* file;
    if (!getfile(file, filePath)) return NULL;

    TTree* tree;
    if (!gettree(tree, file, filePath, treePath)) return NULL;

    TBranch* branch;
    if (!getbranch(branch, tree, filePath, treePath, branchName)) return NULL;

    if (wantCounter != NULL  &&  PyObject_IsTrue(wantCounter))
      return declare_branch(branch, true);
    else
      return declare_branch(branch, false);
  }

  // TODO: check PyArg_ParseTuple(args, "O", &pyroot_tbranch) for a PyROOT tbranch
  // else if () { }

  PyErr_SetString(PyExc_TypeError, "either supply\n    filePath (str), treePath (str), and branchName (str)\nor\n    tbranch (PyROOT)");
  return NULL;
}

/////////////////////////////////////////////////////// fill variants

bool fill_impl(TBranch* branch, PyObject* tofill, Long64_t startingEntry) {
  Long64_t numEntries = branch->GetTree()->GetEntries();

  char* arraydata = PyArray_BYTES(tofill);
  size_t arrayindex = 0;
  size_t arrayend = PyArray_NBYTES(tofill);

  TBufferFile buffer(TBuffer::kWrite, 32*1024);

  while (arrayindex < arrayend  &&  startingEntry < numEntries) {
    Long64_t entries = branch->GetBulkRead().GetEntriesSerialized(startingEntry, buffer);
    startingEntry += entries;

    size_t tocopy = buffer.BufferSize();
    if (arrayindex + tocopy > arrayend)
      tocopy = arrayend - arrayindex;

    memcpy(&arraydata[arrayindex], buffer.GetCurrent(), tocopy);
    arrayindex += tocopy;
  }

  return true;
}

bool fill_flat(TBranch* branch, TLeaf* leaf, PyObject* tofill, Long64_t startingEntry) {
  if (!PyArray_Check(tofill)) {
    PyErr_SetString(PyExc_TypeError, "the TBranch has one flat TLeaf; 'tofill' must be an array");
    return false;
  }

  if (PyArray_NDIM(tofill) != 1) {
    PyErr_SetString(PyExc_ValueError, "the 'tofill' array must be one-dimensional");
    return false;
  }

  if (PyArray_DESCR(tofill)->byteorder != '>') {
    PyErr_SetString(PyExc_ValueError, "the 'tofill' array must be big-endian (e.g. dtype='>f4', '>f8', '>i4', '>u4', etc.)");
    return false;
  }

  return fill_impl(branch, tofill, startingEntry);
}

bool fill_fixedlen(TBranch* branch, TLeaf* leaf, std::vector<int> &dims, PyObject* tofill, Long64_t startingEntry) {
  // TODO: fill a multidimensional Numpy array
  PyErr_SetString(PyExc_NotImplementedError, "fixedlen");
  return false;
}

bool fill_varlen(TBranch* branch, TLeaf* leaf, std::vector<int> &dims, std::vector<std::string> &counters, PyObject* tofill, Long64_t startingEntry) {
  // TODO: fill a single Numpy array with just the data or a 2-tuple of arrays with data and counter
  PyErr_SetString(PyExc_NotImplementedError, "varlen");
  return false;
}

bool fill_unileaf(TBranch* branch, TLeaf* leaf, PyObject* tofill, Long64_t startingEntry) {
  std::vector<int> dims;
  std::vector<std::string> counters;
  getdim(leaf, dims, counters);

  if (has_varlen(counters))
    return fill_varlen(branch, leaf, dims, counters, tofill, startingEntry);
  else if (!dims.empty())
    return fill_fixedlen(branch, leaf, dims, tofill, startingEntry);
  else
    return fill_flat(branch, leaf, tofill, startingEntry);
}

bool fill_multileaf(TBranch* branch, TObjArray* leaves, PyObject* tofill, Long64_t startingEntry) {
  // TODO: fill a Numpy recarray
  PyErr_SetString(PyExc_NotImplementedError, "multileaf");
  return false;
}

bool fill_unibranch(TBranch* branch, PyObject* tofill, Long64_t startingEntry) {
  TObjArray* leaves = branch->GetListOfLeaves();
  if (leaves->GetEntries() == 1)
    return fill_unileaf(branch, dynamic_cast<TLeaf*>(leaves->First()), tofill, startingEntry);
  else
    return fill_multileaf(branch, leaves, tofill, startingEntry);
}

bool fill_multibranch(TObjArray* branches, PyObject* tofill, Long64_t startingEntry) {
  // TODO: fill a Python dict of arrays
  PyErr_SetString(PyExc_NotImplementedError, "multibranch");
  return false;
}

bool fill_branch(TBranch* branch, PyObject* tofill, Long64_t startingEntry) {
  TObjArray* subbranches = branch->GetListOfBranches();
  if (subbranches->GetEntries() == 0)
    return fill_unibranch(branch, tofill, startingEntry);
  else
    return fill_multibranch(subbranches, tofill, startingEntry);
}

static PyObject* fill(PyObject* self, PyObject* args) {
  char* filePath;
  char* treePath;
  char* branchName;
  // PyObject* pyroot_tbranch;
  PyObject* tofill;
  Long64_t startingEntry = 0;
  
  if (PyArg_ParseTuple(args, "sssO|L", &filePath, &treePath, &branchName, &tofill, &startingEntry)) {
    TFile* file;
    if (!getfile(file, filePath)) return NULL;

    TTree* tree;
    if (!gettree(tree, file, filePath, treePath)) return NULL;

    TBranch* branch;
    if (!getbranch(branch, tree, filePath, treePath, branchName)) return NULL;

    if (fill_branch(branch, tofill, startingEntry))
      return Py_BuildValue("O", Py_None);
    else
      return NULL;
  }

  // TODO: check PyArg_ParseTuple(args, "OO|L", &pyroot_tbranch, &array, &startingEntry) for a PyROOT tbranch
  // else if () { }

  PyErr_SetString(PyExc_TypeError, "either supply\n    filePath (str), treePath (str), branchName (str), tofill (Numpy array, tuple, dict), and optional startingEntry (default 0)\nor\n    tbranch (PyROOT), tofill (Numpy array, tuple, dict), and optional startingEntry (default 0)");
  return NULL;
}

/////////////////////////////////////////////////////// iterator variants

