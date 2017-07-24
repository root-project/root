// Author: Enrico Guiraud, Danilo Piparo CERN  12/2016

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDFOPERATIONS
#define ROOT_TDFOPERATIONS

#include "ROOT/TBufferMerger.hxx" // for SnapshotHelper
#include "ROOT/TypeTraits.hxx"
#include "ROOT/TDFUtils.hxx"
#include "ROOT/TThreadedObject.hxx"
#include "TH1.h"
#include "TTreeReader.h" // for SnapshotHelper
#include "TFile.h" // for SnapshotHelper

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

/// \cond HIDDEN_SYMBOLS

namespace ROOT {
namespace Internal {
namespace TDF {
using namespace ROOT::TypeTraits;

using Count_t = unsigned long;
using Hist_t = ::TH1D;

template <typename F>
class ForeachSlotHelper {
   F fCallable;

public:
   using BranchTypes_t = RemoveFirstParameter_t<typename CallableTraits<F>::arg_types>;
   ForeachSlotHelper(F &&f) : fCallable(f) {}

   void InitSlot(TTreeReader*, unsigned int) {}

   template <typename... Args>
   void Exec(unsigned int slot, Args &&... args)
   {
      // check that the decayed types of Args are the same as the branch types
      static_assert(std::is_same<TypeList<typename std::decay<Args>::type...>, BranchTypes_t>::value, "");
      fCallable(slot, std::forward<Args>(args)...);
   }

   void Finalize() { /* noop */}
};

class CountHelper {
   const std::shared_ptr<unsigned int> fResultCount;
   std::vector<Count_t> fCounts;

public:
   using BranchTypes_t = TypeList<>;
   CountHelper(const std::shared_ptr<unsigned int> &resultCount, unsigned int nSlots);
   void InitSlot(TTreeReader*, unsigned int) {}
   void Exec(unsigned int slot);
   void Finalize();
};

class FillHelper {
   // this sets a total initial size of 16 MB for the buffers (can increase)
   static constexpr unsigned int fgTotalBufSize = 2097152;
   using BufEl_t = double;
   using Buf_t = std::vector<BufEl_t>;

   std::vector<Buf_t> fBuffers;
   std::vector<Buf_t> fWBuffers;
   const std::shared_ptr<Hist_t> fResultHist;
   unsigned int fNSlots;
   unsigned int fBufSize;
   Buf_t fMin;
   Buf_t fMax;

   void UpdateMinMax(unsigned int slot, double v);

public:
   FillHelper(const std::shared_ptr<Hist_t> &h, unsigned int nSlots);
   void InitSlot(TTreeReader*, unsigned int) {}
   void Exec(unsigned int slot, double v);
   void Exec(unsigned int slot, double v, double w);

   template <typename T, typename std::enable_if<IsContainer<T>::value, int>::type = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      auto &thisBuf = fBuffers[slot];
      for (auto &v : vs) {
         UpdateMinMax(slot, v);
         thisBuf.emplace_back(v); // TODO: Can be optimised in case T == BufEl_t
      }
   }

   template <typename T, typename W,
             typename std::enable_if<IsContainer<T>::value && IsContainer<W>::value, int>::type = 0>
   void Exec(unsigned int slot, const T &vs, const W &ws)
   {
      auto &thisBuf = fBuffers[slot];
      for (auto &v : vs) {
         UpdateMinMax(slot, v);
         thisBuf.emplace_back(v); // TODO: Can be optimised in case T == BufEl_t
      }

      auto &thisWBuf = fWBuffers[slot];
      for (auto &w : ws) {
         thisWBuf.emplace_back(w); // TODO: Can be optimised in case T == BufEl_t
      }
   }

   void Finalize();
};

extern template void FillHelper::Exec(unsigned int, const std::vector<float> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<double> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<char> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<int> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<unsigned int> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<float> &, const std::vector<float> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<double> &, const std::vector<double> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<char> &, const std::vector<char> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<int> &, const std::vector<int> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<unsigned int> &,
                                      const std::vector<unsigned int> &);

template <typename HIST = Hist_t>
class FillTOHelper {
   std::unique_ptr<TThreadedObject<HIST>> fTo;

public:
   FillTOHelper(FillTOHelper &&) = default;

   FillTOHelper(const std::shared_ptr<HIST> &h, unsigned int nSlots) : fTo(new TThreadedObject<HIST>(*h))
   {
      fTo->SetAtSlot(0, h);
      // Initialise all other slots
      for (unsigned int i = 0; i < nSlots; ++i) {
         fTo->GetAtSlot(i);
      }
   }

   void InitSlot(TTreeReader*, unsigned int) {}

   void Exec(unsigned int slot, double x0) // 1D histos
   {
      fTo->GetAtSlotUnchecked(slot)->Fill(x0);
   }

   void Exec(unsigned int slot, double x0, double x1) // 1D weighted and 2D histos
   {
      fTo->GetAtSlotUnchecked(slot)->Fill(x0, x1);
   }

   void Exec(unsigned int slot, double x0, double x1, double x2) // 2D weighted and 3D histos
   {
      fTo->GetAtSlotUnchecked(slot)->Fill(x0, x1, x2);
   }

   void Exec(unsigned int slot, double x0, double x1, double x2, double x3) // 3D weighted histos
   {
      fTo->GetAtSlotUnchecked(slot)->Fill(x0, x1, x2, x3);
   }

   template <typename X0, typename std::enable_if<IsContainer<X0>::value, int>::type = 0>
   void Exec(unsigned int slot, const X0 &x0s)
   {
      auto thisSlotH = fTo->GetAtSlotUnchecked(slot);
      for (auto &x0 : x0s) {
         thisSlotH->Fill(x0); // TODO: Can be optimised in case T == vector<double>
      }
   }

   template <typename X0, typename X1,
             typename std::enable_if<IsContainer<X0>::value && IsContainer<X1>::value, int>::type = 0>
   void Exec(unsigned int slot, const X0 &x0s, const X1 &x1s)
   {
      auto thisSlotH = fTo->GetAtSlotUnchecked(slot);
      if (x0s.size() != x1s.size()) {
         throw std::runtime_error("Cannot fill histogram with values in containers of different sizes.");
      }
      auto x0sIt = std::begin(x0s);
      const auto x0sEnd = std::end(x0s);
      auto x1sIt = std::begin(x1s);
      for (; x0sIt != x0sEnd; x0sIt++, x1sIt++) {
         thisSlotH->Fill(*x0sIt, *x1sIt); // TODO: Can be optimised in case T == vector<double>
      }
   }

   template <typename X0, typename X1, typename X2,
             typename std::enable_if<IsContainer<X0>::value && IsContainer<X1>::value && IsContainer<X2>::value,
                                     int>::type = 0>
   void Exec(unsigned int slot, const X0 &x0s, const X1 &x1s, const X2 &x2s)
   {
      auto thisSlotH = fTo->GetAtSlotUnchecked(slot);
      if (!(x0s.size() == x1s.size() && x1s.size() == x2s.size())) {
         throw std::runtime_error("Cannot fill histogram with values in containers of different sizes.");
      }
      auto x0sIt = std::begin(x0s);
      const auto x0sEnd = std::end(x0s);
      auto x1sIt = std::begin(x1s);
      auto x2sIt = std::begin(x2s);
      for (; x0sIt != x0sEnd; x0sIt++, x1sIt++, x2sIt++) {
         thisSlotH->Fill(*x0sIt, *x1sIt, *x2sIt); // TODO: Can be optimised in case T == vector<double>
      }
   }
   template <typename X0, typename X1, typename X2, typename X3,
             typename std::enable_if<IsContainer<X0>::value && IsContainer<X1>::value && IsContainer<X2>::value &&
                                        IsContainer<X3>::value,
                                     int>::type = 0>
   void Exec(unsigned int slot, const X0 &x0s, const X1 &x1s, const X2 &x2s, const X3 &x3s)
   {
      auto thisSlotH = fTo->GetAtSlotUnchecked(slot);
      if (!(x0s.size() == x1s.size() && x1s.size() == x2s.size() && x1s.size() == x3s.size())) {
         throw std::runtime_error("Cannot fill histogram with values in containers of different sizes.");
      }
      auto x0sIt = std::begin(x0s);
      const auto x0sEnd = std::end(x0s);
      auto x1sIt = std::begin(x1s);
      auto x2sIt = std::begin(x2s);
      auto x3sIt = std::begin(x3s);
      for (; x0sIt != x0sEnd; x0sIt++, x1sIt++, x2sIt++, x3sIt++) {
         thisSlotH->Fill(*x0sIt, *x1sIt, *x2sIt, *x3sIt); // TODO: Can be optimised in case T == vector<double>
      }
   }
   void Finalize() { fTo->Merge(); }
};

// note: changes to this class should probably be replicated in its partial
// specialization below
template <typename T, typename COLL>
class TakeHelper {
   std::vector<std::shared_ptr<COLL>> fColls;

public:
   using BranchTypes_t = TypeList<T>;
   TakeHelper(const std::shared_ptr<COLL> &resultColl, unsigned int nSlots)
   {
      fColls.emplace_back(resultColl);
      for (unsigned int i = 1; i < nSlots; ++i) fColls.emplace_back(std::make_shared<COLL>());
   }

   void InitSlot(TTreeReader*, unsigned int) {}

   void Exec(unsigned int slot, T v)
   {
      fColls[slot]->emplace_back(v);
   }

   void Finalize()
   {
      auto rColl = fColls[0];
      for (unsigned int i = 1; i < fColls.size(); ++i) {
         auto &coll = fColls[i];
         for (T &v : *coll) {
            rColl->emplace_back(v);
         }
      }
   }
};

// note: changes to this class should probably be replicated in its unspecialized
// declaration above
template <typename T>
class TakeHelper<T, std::vector<T>> {
   std::vector<std::shared_ptr<std::vector<T>>> fColls;

public:
   using BranchTypes_t = TypeList<T>;
   TakeHelper(const std::shared_ptr<std::vector<T>> &resultColl, unsigned int nSlots)
   {
      fColls.emplace_back(resultColl);
      for (unsigned int i = 1; i < nSlots; ++i) {
         auto v = std::make_shared<std::vector<T>>();
         v->reserve(1024);
         fColls.emplace_back(v);
      }
   }

   void InitSlot(TTreeReader*, unsigned int) {}

   void Exec(unsigned int slot, T v)
   {
      fColls[slot]->emplace_back(v);
   }

   void Finalize()
   {
      ULong64_t totSize = 0;
      for (auto &coll : fColls) totSize += coll->size();
      auto rColl = fColls[0];
      rColl->reserve(totSize);
      for (unsigned int i = 1; i < fColls.size(); ++i) {
         auto &coll = fColls[i];
         rColl->insert(rColl->end(), coll->begin(), coll->end());
      }
   }
};

template <typename F, typename T>
class ReduceHelper {
   F fReduceFun;
   const std::shared_ptr<T> fReduceRes;
   std::vector<T> fReduceObjs;

public:
   using BranchTypes_t = TypeList<T>;
   ReduceHelper(F &&f, const std::shared_ptr<T> &reduceRes, unsigned int nSlots)
      : fReduceFun(std::move(f)), fReduceRes(reduceRes), fReduceObjs(nSlots, *reduceRes)
   {
   }

   void InitSlot(TTreeReader*, unsigned int) {}

   void Exec(unsigned int slot, const T &value) { fReduceObjs[slot] = fReduceFun(fReduceObjs[slot], value); }

   void Finalize()
   {
      for (auto &t : fReduceObjs) *fReduceRes = fReduceFun(*fReduceRes, t);
   }
};

class MinHelper {
   const std::shared_ptr<double> fResultMin;
   std::vector<double> fMins;

public:
   MinHelper(const std::shared_ptr<double> &minVPtr, unsigned int nSlots);

   void InitSlot(TTreeReader*, unsigned int) {}

   void Exec(unsigned int slot, double v);

   template <typename T, typename std::enable_if<IsContainer<T>::value, int>::type = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      for (auto &&v : vs) fMins[slot] = std::min((double)v, fMins[slot]);
   }

   void Finalize();
};

extern template void MinHelper::Exec(unsigned int, const std::vector<float> &);
extern template void MinHelper::Exec(unsigned int, const std::vector<double> &);
extern template void MinHelper::Exec(unsigned int, const std::vector<char> &);
extern template void MinHelper::Exec(unsigned int, const std::vector<int> &);
extern template void MinHelper::Exec(unsigned int, const std::vector<unsigned int> &);

class MaxHelper {
   const std::shared_ptr<double> fResultMax;
   std::vector<double> fMaxs;

public:
   MaxHelper(const std::shared_ptr<double> &maxVPtr, unsigned int nSlots);
   void InitSlot(TTreeReader*, unsigned int) {}
   void Exec(unsigned int slot, double v);

   template <typename T, typename std::enable_if<IsContainer<T>::value, int>::type = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      for (auto &&v : vs) fMaxs[slot] = std::max((double)v, fMaxs[slot]);
   }

   void Finalize();
};

extern template void MaxHelper::Exec(unsigned int, const std::vector<float> &);
extern template void MaxHelper::Exec(unsigned int, const std::vector<double> &);
extern template void MaxHelper::Exec(unsigned int, const std::vector<char> &);
extern template void MaxHelper::Exec(unsigned int, const std::vector<int> &);
extern template void MaxHelper::Exec(unsigned int, const std::vector<unsigned int> &);

class MeanHelper {
   const std::shared_ptr<double> fResultMean;
   std::vector<Count_t> fCounts;
   std::vector<double> fSums;

public:
   MeanHelper(const std::shared_ptr<double> &meanVPtr, unsigned int nSlots);
   void InitSlot(TTreeReader*, unsigned int) {}
   void Exec(unsigned int slot, double v);

   template <typename T, typename std::enable_if<IsContainer<T>::value, int>::type = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      for (auto &&v : vs) {
         fSums[slot] += v;
         fCounts[slot]++;
      }
   }

   void Finalize();
};

extern template void MeanHelper::Exec(unsigned int, const std::vector<float> &);
extern template void MeanHelper::Exec(unsigned int, const std::vector<double> &);
extern template void MeanHelper::Exec(unsigned int, const std::vector<char> &);
extern template void MeanHelper::Exec(unsigned int, const std::vector<int> &);
extern template void MeanHelper::Exec(unsigned int, const std::vector<unsigned int> &);

/// Helper object for a single-thread Snapshot action
template <typename... BranchTypes>
class SnapshotHelper {
   std::unique_ptr<TFile> fOutputFile;
   std::unique_ptr<TTree> fOutputTree; // must be a ptr because TTrees are not copy/move constructible
   bool fIsFirstEvent{true};
   const ColumnNames_t fBranchNames;
public:
   SnapshotHelper(const std::string &filename, const std::string &dirname, const std::string &treename,
                  const ColumnNames_t &bnames)
      : fOutputFile(TFile::Open(filename.c_str(), "RECREATE")), fBranchNames(bnames)
   {
      if (!dirname.empty()) {
         fOutputFile->mkdir(dirname.c_str());
         fOutputFile->cd(dirname.c_str());
      }
      fOutputTree.reset(new TTree(treename.c_str(), treename.c_str(), /*splitlevel=*/99, /*dir=*/fOutputFile.get()));
   }

   SnapshotHelper(const SnapshotHelper &) = delete;
   SnapshotHelper(SnapshotHelper &&) = default;
   ~SnapshotHelper() = default;

   void InitSlot(TTreeReader *r, unsigned int /* slot */)
   {
      if (!r) // empty source, nothing to do
         return;
      auto tree = r->GetTree();
      // AddClone guarantees that if the input file changes the branches of the output tree are updated with the new
      // addresses of the branch values
      tree->AddClone(fOutputTree.get());
   }

   void Exec(unsigned int /* slot */, BranchTypes &... values)
   {
      if (fIsFirstEvent) {
         using ind_t = GenStaticSeq_t<sizeof...(BranchTypes)>;
         SetBranches(&values..., ind_t());
      }
      fOutputTree->Fill();
   }

   template <int... S>
   void SetBranches(BranchTypes *... branchAddresses, StaticSeq<S...> /*dummy*/)
   {
      // hack to call TTree::Branch on all variadic template arguments
      std::initializer_list<int> expander = {(fOutputTree->Branch(fBranchNames[S].c_str(), branchAddresses), 0)..., 0};
      (void)expander; // avoid unused variable warnings for older compilers such as gcc 4.9
      fIsFirstEvent = false;
   }

   void Finalize() { fOutputTree->Write(); }
};

/// Helper object for a multi-thread Snapshot action
template <typename... BranchTypes>
class SnapshotHelperMT {
   const unsigned int fNSlots;
   std::unique_ptr<ROOT::Experimental::TBufferMerger> fMerger; // must use a ptr because TBufferMerger is not movable
   std::vector<std::shared_ptr<ROOT::Experimental::TBufferMergerFile>> fOutputFiles;
   std::vector<TTree *> fOutputTrees; // ROOT will own/manage these TTrees, must not delete
   std::vector<int> fIsFirstEvent; // vector<bool> is evil
   const std::string fDirName; // name of TFile subdirectory in which output must be written (possibly empty)
   const std::string fTreeName; // name of output tree
   const ColumnNames_t fBranchNames;
public:
   using BranchTypes_t = TypeList<BranchTypes...>;
   SnapshotHelperMT(unsigned int nSlots, const std::string &filename, const std::string &dirname,
                    const std::string &treename, const ColumnNames_t &bnames)
      : fNSlots(nSlots), fMerger(new ROOT::Experimental::TBufferMerger(filename.c_str(), "RECREATE")),
        fOutputFiles(fNSlots), fOutputTrees(fNSlots, nullptr), fIsFirstEvent(fNSlots, 1), fDirName(dirname),
        fTreeName(treename), fBranchNames(bnames)
   {
   }
   SnapshotHelperMT(const SnapshotHelperMT &) = delete;
   SnapshotHelperMT(SnapshotHelperMT &&) = default;
   ~SnapshotHelperMT() = default;

   void InitSlot(TTreeReader *r, unsigned int slot)
   {
      ::TDirectory::TContext c; // do not let tasks change the thread-local gDirectory
      if (!fOutputTrees[slot]) {
         // first time this thread executes something, let's create a TBufferMerger output directory
         fOutputFiles[slot] = fMerger->GetFile();
      } else {
         // this thread is now re-executing the task, let's flush the current contents of the TBufferMergerFile
         fOutputFiles[slot]->Write();
      }
      TDirectory *treeDirectory = fOutputFiles[slot].get();
      if (!fDirName.empty()) {
         treeDirectory = fOutputFiles[slot]->mkdir(fDirName.c_str());
      }
      fOutputTrees[slot] = new TTree(fTreeName.c_str(), fTreeName.c_str(), /*splitlvl=*/99, /*dir=*/treeDirectory);
      fOutputTrees[slot]->ResetBit(kMustCleanup); // do not mingle with the thread-unsafe gListOfCleanups
      fOutputTrees[slot]->SetImplicitMT(false);
      if (r) {
         // not an empty-source TDF
         auto inputTree = r->GetTree();
         // AddClone guarantees that if the input file changes the branches of the output tree are updated with the new
         // addresses of the branch values
         inputTree->AddClone(fOutputTrees[slot]);
      }
      fIsFirstEvent[slot] = 1; // reset first event flag for this slot
   }

   void Exec(unsigned int slot, BranchTypes &... values)
   {
      if (fIsFirstEvent[slot]) {
         using ind_t = GenStaticSeq_t<sizeof...(BranchTypes)>;
         SetBranches(slot, &values..., ind_t());
         fIsFirstEvent[slot] = 0;
      }
      fOutputTrees[slot]->Fill();
      auto entries = fOutputTrees[slot]->GetEntries();
      auto autoflush = fOutputTrees[slot]->GetAutoFlush();
      if ((autoflush > 0) && (entries % autoflush == 0)) fOutputFiles[slot]->Write();
   }

   template <int... S>
   void SetBranches(unsigned int slot, BranchTypes *... branchAddresses, StaticSeq<S...> /*dummy*/)
   {
      // hack to call TTree::Branch on all variadic template arguments
      std::initializer_list<int> expander = {
         (fOutputTrees[slot]->Branch(fBranchNames[S].c_str(), branchAddresses), 0)..., 0};
      (void)expander; // avoid unused variable warnings for older compilers such as gcc 4.9
   }

   void Finalize()
   {
      for (auto &file : fOutputFiles) {
         if (file) file->Write();
      }
   }
};

} // end of NS TDF
} // end of NS Internal
} // end of NS ROOT

/// \endcond

#endif
