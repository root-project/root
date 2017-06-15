#include <iostream>
#include <vector>
#include <string>

#include <Python.h>
#include <numpy/arrayobject.h>

#include <TBranch.h>
#include <TBufferFile.h>
#include <TFile.h>
#include <TLeaf.h>
#include <TObjArray.h>
#include <TTree.h>
#include <ROOT/TBulkBranchRead.hxx>

/////////////////////////////////////////////////////// module

static char module_docstring[] = "Tools for quickly loading TBranch contents into Numpy arrays.";
static char clusters_docstring[] = "Returns a list of the first entry in each cluster followed by the total number of entries (the length of this list is the number of clusters + 1).";
static char fill_docstring[] = "Fills a given array (or set of arrays) with TBranch data.";

static PyObject* clusters(PyObject* self, PyObject* args);
static PyObject* fill(PyObject* self, PyObject* args);

static PyMethodDef module_methods[] = {
  {"clusters", (PyCFunction)clusters, METH_VARARGS, clusters_docstring},
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
    TFile* file = TFile::Open(filePath);
    TTree* tree;
    file->GetObject(treePath, tree);
    return clusters_impl(tree);
  }

  // TODO: check PyArg_ParseTuple(args, "O", &pyroot_ttree) for a PyROOT ttree
  // else if () { }

  PyErr_SetString(PyExc_TypeError, "either supply\n    filePath (str), treePath (str)\nor\n    tbranch (PyROOT)");
  return NULL;
}

/////////////////////////////////////////////////////// declare variants

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

bool fill_fixedlen(TBranch* branch, TLeaf* leaf, std::vector<int>& dims, PyObject* tofill, Long64_t startingEntry) {
  // TODO: fill a multidimensional Numpy array
  PyErr_SetString(PyExc_NotImplementedError, "fixedlen");
  return false;
}

bool fill_varlen(TBranch* branch, TLeaf* leaf, std::vector<int>& dims, std::vector<std::string>& counters, PyObject* tofill, Long64_t startingEntry) {
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
  else if (dims.size() > 0)
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
    TFile* file = TFile::Open(filePath);
    TTree* tree;
    file->GetObject(treePath, tree);
    TBranch* branch = tree->GetBranch(branchName);

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

