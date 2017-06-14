#include <iostream>

#include <Python.h>
#include <numpy/arrayobject.h>

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TLeaf.h>
#include <TBufferFile.h>
#include <ROOT/TBulkBranchRead.hxx>

static char module_docstring[] = "Tools for quickly filling Numpy arrays with TBranch contents.";
static char fill_docstring[] = "Fills a given array with TBranch data.";

static PyObject* fill(PyObject* self, PyObject* args);

static PyMethodDef module_methods[] = {
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

static PyObject* fill(PyObject* self, PyObject* args) {
  char* filePath;
  char* treePath;
  char* branchName;
  PyObject* array;
  Long64_t startingEntry = 0;
  
  if (!PyArg_ParseTuple(args, "sssO|L", &filePath, &treePath, &branchName, &array, &startingEntry)  ||  !PyArray_Check(array)) {
    PyErr_SetString(PyExc_TypeError, "arguments are: filePath (str), treePath (str), branchName (str), array (Numpy); optionally startingEntry (int, default is 0)");
    return NULL;
  }

  TFile* file = TFile::Open(filePath);
  TTree* tree;
  file->GetObject(treePath, tree);
  Long64_t numEntries = tree->GetEntries();
  TBranch* branch = tree->GetBranch(branchName);

  // error checking is for wimps

  char* arraydata = PyArray_BYTES(array);
  size_t arrayindex = 0;
  size_t arrayend = PyArray_NBYTES(array);

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

  return Py_BuildValue("O", Py_None);
}
