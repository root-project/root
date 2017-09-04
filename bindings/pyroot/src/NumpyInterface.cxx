// @(#)root/pyroot:$Id$
// Author: Jim Pivarski, Jul 2017

// Forwards and Python include
#include "NumpyInterface.h"

// Numpy include must be first
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// ROOT
#include <ROOT/TBulkBranchRead.hxx>
#include <TBasket.h>
#include <TBranch.h>
#include <TBufferFile.h>
#include <TBuffer.h>
#include <TTree.h>

// Bindings
#include "PyROOT.h"
#include "ObjectProxy.h"

namespace PyROOT {

  template<typename T>
  bool getpyroot(PyObject* in, T* &out) {
    if (!ObjectProxy_Check(in)) {
      PyErr_Format(PyExc_TypeError, "argument must be a %s object", T::Class()->GetName());
      return false;
    }
    PyROOT::ObjectProxy* objectProxy = reinterpret_cast<PyROOT::ObjectProxy*>(in);
    
    out = reinterpret_cast<T*>(TClass::GetClass(Cppyy::GetFinalName(objectProxy->ObjectIsA()).c_str())->DynamicCast(T::Class(), objectProxy->GetObject()));
    
    if (!out) {
      PyErr_Format(PyExc_TypeError, "argument must be a %s object", T::Class()->GetName());
      return false;
    }
    
    return true;
  }

#if PY_VERSION_HEX >= 0x03000000
  void* InitializeNumpy() {
    import_array();
    return 0;
  }
#else
  void InitializeNumpy() {
    import_array();
  }
#endif

  PyObject* FillNumpyArray(PyObject* self, PyObject* args) {
    TBranch* branch;
    if (!getpyroot(self, branch))
      return 0;

    PyObject* pyarray;
    Long64_t entry_start = 0;

    if (!PyArg_ParseTuple(args, "O|L", &pyarray, &entry_start))
      return 0;

    if (!PyArray_Check(pyarray)) {
      PyErr_SetString(PyExc_TypeError, "argument must be a Numpy array");
      return 0;
    }
    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(pyarray);

    Long64_t arraysize = 1;
    npy_intp* dims = PyArray_DIMS(array);
    for (int i = 0;  i < PyArray_NDIM(array);  i++)
      arraysize *= dims[i];

    Long64_t arraybytes = arraysize * PyArray_DESCR(array)->elsize;
    Long64_t byteindex = 0;
    char* arraydata = PyArray_BYTES(array);

    branch->DropBaskets();
    TBufferFile buffer(TBuffer::kWrite, 32*1024);

    Long64_t entryindex = entry_start;
    while (entryindex < branch->GetTree()->GetEntries()  &&  byteindex < arraybytes) {
      Long64_t num_entries = branch->GetBulkRead().GetEntriesSerialized(entryindex, buffer, false);
      if (num_entries <= 0) {
        PyErr_Format(PyExc_IOError, "GetBulkRead().GetEntriesSerialized failed at entry %lld", entryindex);
        return 0;
      }

      Long64_t header_size = reinterpret_cast<size_t>(buffer.GetCurrent()) - reinterpret_cast<size_t>(buffer.Buffer());

      TBasket* basket = reinterpret_cast<TBasket*>(branch->GetListOfBaskets()->Last());
      Long64_t num_bytes = basket->GetLast() - header_size;

      memcpy(&arraydata[byteindex], buffer.GetCurrent(), num_bytes);

      entryindex += num_entries;
      byteindex += num_bytes;
    }

    return Py_BuildValue("LL", entryindex, byteindex);
  }

} // namespace PyROOT
