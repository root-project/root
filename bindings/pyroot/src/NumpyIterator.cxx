// @(#)root/pyroot:$Id$
// Author: Jim Pivarski, Jul 2017

// Forwards and Python include
#include "NumpyIterator.h"

// ROOT
#include <ROOT/TBulkBranchRead.hxx>
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

// Bindings
#include "PyROOT.h"
#include "ObjectProxy.h"

// Standard
#include <string>

#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#endif // NPY_ARRAY_C_CONTIGUOUS

#ifndef NPY_ARRAY_ALIGNED
#define NPY_ARRAY_ALIGNED NPY_ALIGNED
#endif // NPY_ARRAY_ALIGNED

#define IS_ALIGNED(ptr) ((size_t)ptr % 8 == 0)     // FIXME: is there a better way to check for alignment?

namespace PyROOT {

/////////////////////////////////////////////////////// class methods

void ClusterBuffer::CopyToExtra(Long64_t keep_start) {
  // this is a safer algorithm than is necessary, and it could impact performance, but significantly less so than the other speed-ups

  // remove data from the start of the extra buffer to keep it from growing too much
  if (exEntryStart < keep_start) {
    const Long64_t offset = (keep_start - exEntryStart) * fItemSize;
    const Long64_t newsize = (exEntryEnd - keep_start) * fItemSize;

    memmove(fExtra.data(), &fExtra.data()[offset], newsize);
    fExtra.resize(newsize);
    exEntryStart = keep_start;
  }

  // append the BufferFile at the end of the extra buffer
  const Long64_t oldsize = fExtra.size();
  const Long64_t additional = (bfEntryEnd - bfEntryStart) * fItemSize;

  if (additional > 0) {
    fExtra.resize(oldsize + additional);
    memcpy(&fExtra.data()[oldsize], fBufferFile.GetCurrent(), additional);
    exEntryEnd = bfEntryEnd;
  }
}

// ReadOne asks ROOT to read one basket from the file
// and ClusterBuffer ensures that entries as old as keep_start are preserved
void ClusterBuffer::ReadOne(Long64_t keep_start, const char* &error_string) {
  if (!fUsingExtra  &&  bfEntryEnd > keep_start) {
    // need to overwrite the TBufferFile before we're done with it, so we need to start using extra now
    CopyToExtra(0);
    fUsingExtra = true;
  }

  // read in one more basket, starting at the old bfEntryEnd
  Long64_t numentries = fRequest.branch->GetBulkRead().GetEntriesSerialized(bfEntryEnd, fBufferFile);

  if (fSwapBytes) {
    switch (fItemSize) {
      case 8:
        {
          Long64_t* buffer64 = reinterpret_cast<Long64_t*>(fBufferFile.GetCurrent());
          for (Long64_t i = 0;  i < numentries;  i++)
            buffer64[i] = __builtin_bswap64(buffer64[i]);
          break;
        }

      case 4:
        {
          Int_t* buffer32 = reinterpret_cast<Int_t*>(fBufferFile.GetCurrent());
          for (Long64_t i = 0;  i < numentries;  i++)
            buffer32[i] = __builtin_bswap32(buffer32[i]);
          break;
        }

      case 2:
        {
          Short_t* buffer16 = reinterpret_cast<Short_t*>(fBufferFile.GetCurrent());
          for (Long64_t i = 0;  i < numentries;  i++)
            buffer16[i] = __builtin_bswap16(buffer16[i]);
          break;
        }

      default:
        error_string = "illegal fItemSize";
        return;
    }
  }

  // update the range
  bfEntryStart = bfEntryEnd;
  bfEntryEnd = bfEntryStart + numentries;

  // check for errors
  if (numentries <= 0) {
    bfEntryEnd = bfEntryStart;
    error_string = "failed to read TBasket into TBufferFile (using GetBulkRead().GetEntriesSerialized)";
  }

  // for now, always mirror to the extra buffer
  if (fUsingExtra)
    CopyToExtra(keep_start);
}

// GetBuffer returns a pointer to contiguous data with its size
// if you're lucky (and ask for it), this is performed without any copies
void* ClusterBuffer::GetBuffer(Long64_t &numbytes, Long64_t entry_start, Long64_t EntryEnd) {
  numbytes = (EntryEnd - entry_start) * fItemSize;

  if (fUsingExtra) {
    const Long64_t offset = (entry_start - exEntryStart) * fItemSize;
    return &fExtra.data()[offset];
  }
  else {
    const Long64_t offset = (entry_start - bfEntryStart) * fItemSize;
    return &fBufferFile.GetCurrent()[offset];
  }
}

Long64_t ClusterBuffer::EntryEnd() {
  return bfEntryEnd;  // hide the distinction between bf and extra
}

void ClusterBuffer::Reset() {
  fExtra.clear();
  bfEntryStart = 0;
  bfEntryEnd = 0;
  exEntryStart = 0;
  exEntryEnd = 0;
  // required for re-readability
  fRequest.branch->DropBaskets();
}

// step all ClusterBuffers forward, for all branches, returning true when done and setting error_string on any errors
bool NumpyIterator::StepForward(const char* &error_string) {
  // put your feet together for the next step
  fCurrentStart = fCurrentEnd;

  // check for done
  if (fCurrentEnd >= fNumEntries)
    return true;

  // increment the branches that are at the forefront
  for (unsigned int i = 0;  i < fClusterBuffers.size();  i++) {
    ClusterBuffer &buf = *fClusterBuffers[i];
    if (buf.EntryEnd() == fCurrentStart) {
      buf.ReadOne(fCurrentStart, error_string);
      if (error_string != nullptr)
        return true;
    }
  }

  // find the maximum EntryEnd
  fCurrentEnd = -1;
  for (unsigned int i = 0;  i < fClusterBuffers.size();  i++) {
    ClusterBuffer &buf = *fClusterBuffers[i];
    if (buf.EntryEnd() > fCurrentEnd)
      fCurrentEnd = buf.EntryEnd();
  }

  // bring all others up to at least fCurrentEnd
  for (unsigned int i = 0;  i < fClusterBuffers.size();  i++) {
    ClusterBuffer &buf = *fClusterBuffers[i];
    while (buf.EntryEnd() < fCurrentEnd) {
      buf.ReadOne(fCurrentStart, error_string);
      if (error_string != nullptr)
        return true;
    }
  }

  return false;
}

// get a Python tuple of arrays for all buffers
PyObject* NumpyIterator::arrays() {
  // step forward, handling errors
  const char* error_string = nullptr;
  if (StepForward(error_string)) {
    if (error_string != nullptr) {
      PyErr_SetString(PyExc_IOError, error_string);
      return 0;
    }
    else {
      PyErr_SetNone(PyExc_StopIteration);
      return 0;
    }
  }

  // create a tuple of results
  PyObject* out = PyTuple_New(2 + fClusterBuffers.size());
  PyTuple_SET_ITEM(out, 0, PyLong_FromLong(fCurrentStart));
  PyTuple_SET_ITEM(out, 1, PyLong_FromLong(fCurrentEnd));

  for (unsigned int i = 0;  i < fClusterBuffers.size();  i++) {
    ClusterBuffer &buf = *fClusterBuffers[i];
    const ArrayInfo &ai = fArrayInfo[i];

    Long64_t numbytes;
    void* ptr = buf.GetBuffer(numbytes, fCurrentStart, fCurrentEnd);

    npy_intp dims[ai.nd];
    dims[0] = numbytes / ai.dtype->elsize;
    for (unsigned int j = 0;  j < ai.dims.size();  j++) {
      dims[j + 1] = ai.dims[j];
    }

    Py_INCREF(ai.dtype);

    PyObject* array;
    if (fReturnNewBuffers) {
      array = PyArray_Empty(ai.nd, dims, ai.dtype, false);
      memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)), ptr, numbytes);
    }
    else {
      int flags = NPY_ARRAY_C_CONTIGUOUS;
      if (!IS_ALIGNED(ptr))
        flags |= NPY_ARRAY_ALIGNED;
      array = PyArray_NewFromDescr(&PyArray_Type, ai.dtype, ai.nd, dims, 0, ptr, flags, 0);
    }

    PyTuple_SET_ITEM(out, i + 2, array);
  }

  return out;
}

void NumpyIterator::Reset() {
  fCurrentStart = 0;
  fCurrentEnd = 0;
  for (unsigned int i = 0;  i < fClusterBuffers.size();  i++) {
    fClusterBuffers[i]->Reset();
  }
}

/////////////////////////////////////////////////////// helper functions

bool getrequest(Request &request, TTree* tree, const char* branchName) {
  request.branch = tree->GetBranch(branchName);

  if (request.branch != 0)
    request.leaf = std::string();

  else {
    request.branch = tree->GetLeaf(branchName)->GetBranch();

    if (request.branch == 0) {
      PyErr_Format(PyExc_IOError, "could not read branch \"%s\" from tree \"%s\"", branchName, tree->GetName());
      return false;
    }

    request.leaf = branchName;
  }
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
      default:
        if (std::string(expectedClass->GetName()) == std::string("TClonesArray"))
          return swap_bytes ? "<i4" : ">i4";
        else
          return 0;
    }
  }
  return 0;
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
  if (asstring == 0) {
    PyErr_Format(PyExc_ValueError, "cannot convert type of TLeaf \"%s\" to Numpy", leaf->GetName());
    return false;
  }

  if (!PyArray_DescrConverter(PyUnicode_FromString(asstring), &dtype)) {
    PyErr_SetString(PyExc_ValueError, "cannot create a dtype");
    return 0;
  }

  return true;
}

bool dtypedim_unileaf(ArrayInfo &arrayinfo, TLeaf* leaf, bool swap_bytes) {
  std::vector<std::string> counters;
  getdim(leaf, arrayinfo.dims, counters);
  arrayinfo.nd = 1 + arrayinfo.dims.size();    // first dimension is for the set of entries itself
  if (counters.empty())
    arrayinfo.counter = std::string("");
  else
    arrayinfo.counter = counters[0];

  return dtypedim(arrayinfo.dtype, leaf, swap_bytes);
}

bool dtypedim_multileaf(ArrayInfo &arrayinfo, TObjArray* leaves, bool swap_bytes) {
  // silence warnings until this placeholder is implemented
  (void)(arrayinfo);
  (void)(leaves);
  (void)(swap_bytes);
  // TODO: Numpy recarray dtype
  PyErr_SetString(PyExc_NotImplementedError, "multileaf");
  return false;
}

bool dtypedim_multibranch(ArrayInfo &arrayinfo, TObjArray* branches, bool swap_bytes) {
  // silence warnings until this placeholder is implemented
  (void)(arrayinfo);
  (void)(branches);
  (void)(swap_bytes);
  // TODO: dict of Numpy arrays (nested when this function is called recursively)
  PyErr_SetString(PyExc_NotImplementedError, "multibranch");
  return false;
}

bool dtypedim_branch(ArrayInfo &arrayinfo, Request request, bool swap_bytes) {
  TObjArray* leaves = request.branch->GetListOfLeaves();
  if (request.leaf != std::string(""))
    return dtypedim_unileaf(arrayinfo, request.branch->GetLeaf(request.leaf.c_str()), swap_bytes);
  else if (leaves->GetEntries() == 1)
    return dtypedim_unileaf(arrayinfo, dynamic_cast<TLeaf*>(leaves->First()), swap_bytes);
  else
    return dtypedim_multileaf(arrayinfo, leaves, swap_bytes);
}

bool dtypedim_request(ArrayInfo &arrayinfo, Request request, bool swap_bytes) {
  TObjArray* subbranches = request.branch->GetListOfBranches();
  if (subbranches->GetEntries() != 0  &&  request.leaf == std::string(""))
    return dtypedim_multibranch(arrayinfo, subbranches, swap_bytes);
  else
    return dtypedim_branch(arrayinfo, request, swap_bytes);
}

bool checkstring(PyObject* str) {
  return PyUnicode_Check(str)  ||  PyBytes_Check(str);
}

std::string getstring(PyObject* str) {
  if (PyUnicode_Check(str)) {
    PyObject* bytes = PyUnicode_AsEncodedString(str, "ascii", "backslashreplace");
    std::string out(PyBytes_AsString(bytes));
    Py_DECREF(bytes);
    return out;
  }
  else
    return std::string(PyBytes_AsString(str));
}

/////////////////////////////////////////////////////// Python functions

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

bool gettree(PyObject* self, TTree* &tree) {
  if (!ObjectProxy_Check(self)) {
    PyErr_SetString(PyExc_TypeError, "TTree::GetNumpyIterator must be called with a TTree instance as its implicit argument");
    return false;
  }
  PyROOT::ObjectProxy* pyobj = reinterpret_cast<PyROOT::ObjectProxy*>(self);

  tree = (TTree*)TClass::GetClass(Cppyy::GetFinalName(pyobj->ObjectIsA()).c_str())->DynamicCast(TTree::Class(), pyobj->GetObject());

  if (!tree) {
    PyErr_SetString(PyExc_TypeError, "TTree::GetNumpyIterator must be called with a TTree instance as its implicit argument");
    return false;
  }

  return true;
}

bool getleaf(PyObject* self, TLeaf* &leaf) {
  if (!ObjectProxy_Check(self)) {
    PyErr_SetString(PyExc_TypeError, "TLeaf::FillNumpy must be called with a TLeaf instance as its implicit argument");
    return false;
  }
  PyROOT::ObjectProxy* pyobj = reinterpret_cast<PyROOT::ObjectProxy*>(self);

  leaf = (TLeaf*)TClass::GetClass(Cppyy::GetFinalName(pyobj->ObjectIsA()).c_str())->DynamicCast(TLeaf::Class(), pyobj->GetObject());

  if (!leaf) {
    PyErr_SetString(PyExc_TypeError, "TLeaf::FillNumpy must be called with a TLeaf instance as its implicit argument");
    return false;
  }

  return true;
}

bool getrequests(PyObject* self, PyObject* args, TTree* &tree, std::vector<Request> &requests) {
  if (!gettree(self, tree))
    return false;

  if (PyTuple_GET_SIZE(args) < 1) {
    PyErr_SetString(PyExc_TypeError, "at least one argument is required");
    return false;
  }

  for (int i = 0;  i < PyTuple_GET_SIZE(args);  i++) {
    PyObject* obj = PyTuple_GET_ITEM(args, i);
    if (!checkstring(obj)) {
      PyErr_SetString(PyExc_TypeError, "all arguments must be strings (branch names)");
      return false;
    }
    std::string branchName = getstring(obj);

    Request request;
    if (!getrequest(request, tree, branchName.c_str())) return false;
    requests.push_back(request);
  }

  return true;
}

PyObject* GetNumpyIterator(PyObject* self, PyObject* args, PyObject* kwds) {
  TTree *tree;
  std::vector<Request> requests;
  if (!getrequests(self, args, tree, requests))
    return 0;

  bool return_new_buffers = true;
  bool swap_bytes = true;

  if (kwds != 0) {
    PyObject* key;
    PyObject* value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(kwds, &pos, &key, &value)) {
      if (checkstring(key)  &&  getstring(key) == std::string("return_new_buffers")) {
        if (PyObject_IsTrue(value))
          return_new_buffers = true;
        else
          return_new_buffers = false;
      }

      else if (checkstring(key)  &&  getstring(key) == std::string("swap_bytes")) {
        if (PyObject_IsTrue(value))
          swap_bytes = true;
        else
          swap_bytes = false;
      }

      else if (checkstring(key)) {
        PyErr_Format(PyExc_TypeError, "unrecognized option: %s", getstring(key).c_str());
        return 0;
      }

      else {
        PyErr_SetString(PyExc_TypeError, "unrecognized option");
        return 0;
      }
    }
  }

  std::vector<ArrayInfo> arrayinfo;

  for (unsigned int i = 0;  i < requests.size();  i++) {
    arrayinfo.push_back(ArrayInfo());
    if (!dtypedim_request(arrayinfo.back(), requests[i], swap_bytes))
      return 0;
  }

  PyNumpyIterator* out = PyObject_New(PyNumpyIterator, &PyNumpyIteratorType);
  out->iter = 0;

  if (!PyObject_Init(reinterpret_cast<PyObject*>(out), &PyNumpyIteratorType)) {
    Py_DECREF(out);
    return 0;
  }

  Long64_t num_entries = requests.back().branch->GetTree()->GetEntries();
  out->iter = new NumpyIterator(requests, arrayinfo, num_entries, return_new_buffers, swap_bytes);

  return reinterpret_cast<PyObject*>(out);
}

PyObject* GetNumpyIteratorInfo(PyObject* self, PyObject* args, PyObject* kwds) {
  TTree* tree;
  std::vector<Request> requests;
  if (!getrequests(self, args, tree, requests))
    return 0;

  bool swap_bytes = true;

  if (kwds != 0) {
    PyObject* key;
    PyObject* value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(kwds, &pos, &key, &value)) {
      if (checkstring(key)  &&  getstring(key) == std::string("swap_bytes")) {
        if (PyObject_IsTrue(value))
          swap_bytes = true;
        else
          swap_bytes = false;
      }

      else if (checkstring(key)) {
        PyErr_Format(PyExc_TypeError, "unrecognized option: %s", getstring(key).c_str());
        return 0;
      }

      else {
        PyErr_SetString(PyExc_TypeError, "unrecognized option");
        return 0;
      }
    }
  }

  PyObject* out = PyTuple_New(requests.size());

  for (unsigned int i = 0;  i < requests.size();  i++) {
    ArrayInfo arrayinfo;
    if (!dtypedim_request(arrayinfo, requests[i], swap_bytes)) {
      Py_DECREF(out);
      return 0;
    }

    PyObject* shape = PyTuple_New(arrayinfo.nd);
    PyTuple_SET_ITEM(shape, 0, PyLong_FromLong((int)ceil(1.0 * requests[i].branch->GetTotalSize() / arrayinfo.dtype->elsize)));

    for (unsigned int j = 0;  j < arrayinfo.dims.size();  j++)
      PyTuple_SET_ITEM(shape, j + 1, PyLong_FromLong(arrayinfo.dims[j]));

    PyObject* tupleitem = PyTuple_New(4);
    Py_INCREF(arrayinfo.dtype);
    PyTuple_SET_ITEM(tupleitem, 0, PyUnicode_FromString(requests[i].branch->GetName()));
    PyTuple_SET_ITEM(tupleitem, 1, reinterpret_cast<PyObject*>(arrayinfo.dtype));
    PyTuple_SET_ITEM(tupleitem, 2, shape);
    if (arrayinfo.counter == std::string(""))
      PyTuple_SET_ITEM(tupleitem, 3, Py_BuildValue("O", Py_None));
    else
      PyTuple_SET_ITEM(tupleitem, 3, Py_BuildValue("s", arrayinfo.counter.c_str()));

    PyTuple_SET_ITEM(out, i, tupleitem);
  }

  return out;
}

PyObject* FillNumpy(PyObject* self, PyObject* args) {
  TLeaf* leaf;
  if (!getleaf(self, leaf))
    return 0;

  TBranch* branch = leaf->GetBranch();
  TTree* tree = branch->GetTree();

  PyObject* pyarray;
  Long64_t entry_start = 0;
  if (!PyArg_ParseTuple(args, "O|l", &pyarray, &entry_start))
    return 0;

  Long64_t entry_end = tree->GetEntries();

  if (!PyArray_Check(pyarray)) {
    PyErr_SetString(PyExc_TypeError, "first argument must be a Numpy array");
    return 0;
  }

  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(pyarray);

  Long64_t arraylength = 1;
  for (int i = 0;  i < PyArray_NDIM(array);  i++)
    arraylength *= PyArray_DIM(array, i);

  char* arraydata = PyArray_BYTES(array);

  if (PyArray_DESCR(array)->elsize != leaf->GetLenType()) {
    PyErr_Format(PyExc_TypeError, "array expects %d-byte elements but leaf provides %d-byte elements", PyArray_DESCR(array)->elsize, leaf->GetLenType());
    return 0;
  }

  int makeclass_mode = tree->GetMakeClass();
  tree->SetMakeClass(1);

  int leaflength = leaf->GetLenType();
  char branchdata[leaflength];
  branch->SetAddress(branchdata);

  Long64_t i, j;
  for (i = 0;  i < arraylength  &&  i < entry_end - entry_start;  i++) {
    branch->GetEntry(entry_start + i);
    for (j = 0;  j < leaflength;  j++)
      arraydata[i*leaflength + j] = branchdata[j];
  }
  
  tree->SetMakeClass(makeclass_mode);

  return Py_BuildValue("i", i);
}

} // namespace PyROOT
