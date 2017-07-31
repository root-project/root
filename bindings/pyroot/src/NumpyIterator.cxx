// @(#)root/pyroot:$Id$
// Author: Jim Pivarski, Jul 2017

// Forwards and Python include
#include "NumpyIterator.h"

// Numpy include must be first
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// ROOT
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

// Standard
#include <string>
#include <vector>

#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#endif // NPY_ARRAY_C_CONTIGUOUS

#ifndef NPY_ARRAY_ALIGNED
#define NPY_ARRAY_ALIGNED NPY_ALIGNED
#endif // NPY_ARRAY_ALIGNED

#define IS_ALIGNED(ptr) ((size_t)ptr % 8 == 0)     // FIXME: is there a better way to check for alignment?

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
  const TBranch* fBranch;
  const Long64_t fItemSize;
  const bool fSwapBytes;
  TBufferFile fBufferFile;
  std::vector<char> fExtra;
  void* fOldExtra;
  bool fUsingExtra;

  // always numbers of entries (not bytes) and always inclusive on start, exclusive on end (like Python)
  // also, the TBufferFile is always ahead of the extra buffer and there's no gap between them
  Long64_t bfEntryStart;
  Long64_t bfEntryEnd;
  Long64_t exEntryStart;
  Long64_t exEntryEnd;

  void CopyToExtra(Long64_t keep_start);

  inline void CheckExtraAllocations() {
    if (fOldExtra != reinterpret_cast<void*>(fExtra.data())) {
      fOldExtra = reinterpret_cast<void*>(fExtra.data());
    }
  }

public:
  ClusterBuffer(const TBranch* branch, const Long64_t itemsize, const bool swap_bytes) :
    fBranch(branch), fItemSize(itemsize), fSwapBytes(swap_bytes), fBufferFile(TBuffer::kWrite, 32*1024), fOldExtra(nullptr), fUsingExtra(false),
    bfEntryStart(0), bfEntryEnd(0), exEntryStart(0), exEntryEnd(0)
  {
    CheckExtraAllocations();
  }

  void ReadOne(Long64_t keep_start, const char* &error_string);
  void* GetBuffer(Long64_t &numbytes, Long64_t entry_start, Long64_t entry_end);

  Long64_t EntryEnd() {
    return bfEntryEnd;  // hide the distinction between bf and extra
  }

  void Reset() {
    fExtra.clear();
    bfEntryStart = 0;
    bfEntryEnd = 0;
    exEntryStart = 0;
    exEntryEnd = 0;
  }
};

class NumpyIterator {
private:
  std::vector<std::unique_ptr<ClusterBuffer>> fRequested;
  const std::vector<ArrayInfo> fArrayInfo;   // has the same length as fRequested
  const Long64_t fNumEntries;
  const bool fReturnNewBuffers;
  Long64_t fCurrentStart;
  Long64_t fCurrentEnd;

  bool StepForward(const char* &error_string);

public:
  NumpyIterator(const std::vector<TBranch*> &requested_branches, const std::vector<ArrayInfo> arrayinfo, Long64_t num_entries, bool return_new_buffers, bool swap_bytes) :
    fArrayInfo(arrayinfo), fNumEntries(num_entries), fReturnNewBuffers(return_new_buffers), fCurrentStart(0), fCurrentEnd(0) {
    for (unsigned int i = 0;  i < fArrayInfo.size();  i++)
      fRequested.push_back(std::unique_ptr<ClusterBuffer>(new ClusterBuffer(requested_branches[i], fArrayInfo[i].dtype->elsize, swap_bytes)));
  }

  PyObject* arrays();

  void Reset() {
    fCurrentStart = 0;
    fCurrentEnd = 0;
    for (unsigned int i = 0;  i < fRequested.size();  i++) {
      fRequested[i]->Reset();
    }
  }
};    

void ClusterBuffer::CopyToExtra(Long64_t keep_start) {
  // this is a safer algorithm than is necessary, and it could impact performance, but significantly less so than the other speed-ups

  // remove data from the start of the extra buffer to keep it from growing too much
  if (exEntryStart < keep_start) {
    const Long64_t offset = (keep_start - exEntryStart) * fItemSize;
    const Long64_t newsize = (exEntryEnd - keep_start) * fItemSize;

    memmove(fExtra.data(), &fExtra.data()[offset], newsize);

    fExtra.resize(newsize);
    CheckExtraAllocations();

    exEntryStart = keep_start;
  }

  // append the BufferFile at the end of the extra buffer
  const Long64_t oldsize = fExtra.size();
  const Long64_t additional = (bfEntryEnd - bfEntryStart) * fItemSize;

  if (additional > 0) {
    fExtra.resize(oldsize + additional);
    CheckExtraAllocations();

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
  Long64_t numentries = fBranch->GetBulkRead().GetEntriesSerialized(bfEntryEnd, fBufferFile);

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

// step all ClusterBuffers forward, for all branches, returning true when done and setting error_string on any errors
bool NumpyIterator::StepForward(const char* &error_string) {
  // put your feet together for the next step
  fCurrentStart = fCurrentEnd;

  // check for done
  if (fCurrentEnd >= fNumEntries)
    return true;

  // increment the branches that are at the forefront
  for (unsigned int i = 0;  i < fRequested.size();  i++) {
    ClusterBuffer &buf = *fRequested[i];
    if (buf.EntryEnd() == fCurrentStart) {
      buf.ReadOne(fCurrentStart, error_string);
      if (error_string != nullptr)
        return true;
    }
  }

  // find the maximum EntryEnd
  fCurrentEnd = -1;
  for (unsigned int i = 0;  i < fRequested.size();  i++) {
    ClusterBuffer &buf = *fRequested[i];
    if (buf.EntryEnd() > fCurrentEnd)
      fCurrentEnd = buf.EntryEnd();
  }

  // bring all others up to at least fCurrentEnd
  for (unsigned int i = 0;  i < fRequested.size();  i++) {
    ClusterBuffer &buf = *fRequested[i];
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
      return NULL;
    }
    else {
      PyErr_SetNone(PyExc_StopIteration);
      return NULL;
    }
  }

  // create a tuple of results
  PyObject* out = PyTuple_New(2 + fRequested.size());
  PyTuple_SET_ITEM(out, 0, PyLong_FromLong(fCurrentStart));
  PyTuple_SET_ITEM(out, 1, PyLong_FromLong(fCurrentEnd));

  for (unsigned int i = 0;  i < fRequested.size();  i++) {
    ClusterBuffer &buf = *fRequested[i];
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

/////////////////////////////////////////////////////// Python functions

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

static PyObject* GetNumpyIterator(PyObject* self, PyObject* args, PyObject* kwds) {
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

  std::vector<ArrayInfo> arrayinfo;

  for (unsigned int i = 0;  i < requested_branches.size();  i++) {
    arrayinfo.push_back(ArrayInfo());
    if (!dtypedim_branch(arrayinfo.back(), requested_branches[i], swap_bytes))
      return NULL;
  }

  PyNumpyIterator* out = PyObject_New(PyNumpyIterator, &PyNumpyIteratorType);

  if (!PyObject_Init(reinterpret_cast<PyObject*>(out), &PyNumpyIteratorType)) {
    Py_DECREF(out);
    return NULL;
  }

  Long64_t num_entries = requested_branches.back()->GetTree()->GetEntries();
  out->iter = new NumpyIterator(requested_branches, arrayinfo, num_entries, return_new_buffers, swap_bytes);

  return reinterpret_cast<PyObject*>(out);
}

static PyObject* GetNumpyTypeAndSize(PyObject* self, PyObject* args, PyObject* kwds) {
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
