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
  Long64_t current_start;
  Long64_t current_end;

public:
  ClusterIterator(const std::vector<TBranch*> &requested_branches, const std::vector<TBranch*> &unrequested_counters, const std::vector<ArrayInfo> arrayinfo, Long64_t num_entries) :
    arrayinfo(arrayinfo), num_entries(num_entries), current_start(0), current_end(0) {
    for (unsigned int i = 0;  i < arrayinfo.size();  i++)
      requested.push_back(std::unique_ptr<ClusterBuffer>(new ClusterBuffer(requested_branches[i], arrayinfo[i].dtype->elsize)));
  }

  bool stepforward(const char* &error_string);
  PyObject* arrays(bool return_new_buffers, bool require_alignment);
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
      if (error_string != NULL)
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
      if (error_string != NULL)
        return true;
    }
  }

  return false;
}

// get a Python tuple of arrays for all buffers
PyObject* ClusterIterator::arrays(bool return_new_buffers, bool require_alignment) {
  return nullptr;
}
