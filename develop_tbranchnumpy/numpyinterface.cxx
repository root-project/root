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

#define ALIGNMENT 8;    // if a pointer % ALIGNMENT == 0, declare it "aligned"

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
  const TBufferFile bf;
  const std::vector<char> extra;
  const Long64_t itemsize;
  Long64_t bf_entry_start;
  Long64_t bf_entry_end;
  Long64_t ex_entry_start;
  Long64_t ex_entry_end;
  bool spilled_to_extra;

public:
  ClusterBuffer(const TBranch* branch, const Long64_t itemsize) :
    branch(branch), bf(TBuffer::kWrite, 32*1024), itemsize(itemsize),
    bf_entry_start(0), bf_entry_end(0), ex_entry_start(0), ex_entry_end(0), spilled_to_extra(false) { }

  bool readmore(Long64_t target_start, Long64_t target_end);
  void* getbuffer(Long64_t &size, bool &iscopy);
};

enum CopyMode {ALWAYS, IF_UNALIGNED, TRY_NOT_TO};

class ClusterIterator {
private:
  std::vector<std::unique_ptr<ClusterBuffer>> requested;
  const std::vector<ArrayInfo> arrayinfo;
  const std::vector<std::unique_ptr<ClusterBuffer>> extra_counters;
  const Long64_t num_entries;
  const Long64_t current_start;
  const Long64_t current_end;

public:
  ClusterIterator(const std::vector<TBranch*> &branches, const std::vector<ArrayInfo> arrayinfo, Long64_t num_entries) :
    arrayinfo(arrayinfo), num_entries(num_entries), current_start(0), current_end(0) {
    for (unsigned int i = 0;  i < arrayinfo.size();  i++)
      requested.push_back(std::unique_ptr<ClusterBuffer>(new ClusterBuffer(branches[i], arrayinfo[i].dtype->elsize)));
  }

  bool stepforward(const char* &error_string);
  PyObject* arrays(CopyMode copy);
};    

// readmore asks ROOT to read from the file until reaching entry target_end
// and ClusterBuffer ensures that entries as old as target_start are preserved
bool ClusterBuffer::readmore(Long64_t target_start, Long64_t target_end) {
  return false;
}

// getbuffer returns a pointer to contiguous data with its size
// if you're lucky, this is performed without any copies
void* ClusterBuffer::getbuffer(Long64_t &size, bool &iscopy) {
  return nullptr;
}

// step all ClusterBuffers forward, for all branches
bool ClusterIterator::stepforward(const char* &error_string) {
  return false;
}

// get a Python tuple of arrays for all buffers
PyObject* ClusterIterator::arrays(CopyMode copy) {
  return nullptr;
}
