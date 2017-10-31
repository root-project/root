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
#include "ROOT/TArrayBranch.hxx"
#include "TH1.h"
#include "TTreeReader.h" // for SnapshotHelper
#include "TFile.h"       // for SnapshotHelper

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
using namespace ROOT::Experimental::TDF;

using Hist_t = ::TH1D;

template <typename F>
class ForeachSlotHelper {
   F fCallable;

public:
   using BranchTypes_t = RemoveFirstParameter_t<typename CallableTraits<F>::arg_types>;
   ForeachSlotHelper(F &&f) : fCallable(f) {}
   ForeachSlotHelper(ForeachSlotHelper &&) = default;
   ForeachSlotHelper(const ForeachSlotHelper &) = delete;

   void InitSlot(TTreeReader *, unsigned int) {}

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
   const std::shared_ptr<ULong64_t> fResultCount;
   std::vector<ULong64_t> fCounts;

public:
   using BranchTypes_t = TypeList<>;
   CountHelper(const std::shared_ptr<ULong64_t> &resultCount, const unsigned int nSlots);
   CountHelper(CountHelper &&) = default;
   CountHelper(const CountHelper &) = delete;
   void InitSlot(TTreeReader *, unsigned int) {}
   void Exec(unsigned int slot);
   void Finalize();
   ULong64_t &PartialUpdate(unsigned int slot);
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
   /// Histograms containing "snapshots" of partial results. Non-null only if a registered callback requires it.
   std::vector<std::unique_ptr<Hist_t>> fPartialHists;
   Buf_t fMin;
   Buf_t fMax;

   void UpdateMinMax(unsigned int slot, double v);

public:
   FillHelper(const std::shared_ptr<Hist_t> &h, const unsigned int nSlots);
   FillHelper(FillHelper &&) = default;
   FillHelper(const FillHelper &) = delete;
   void InitSlot(TTreeReader *, unsigned int) {}
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

   Hist_t &PartialUpdate(unsigned int);

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
extern template void
FillHelper::Exec(unsigned int, const std::vector<unsigned int> &, const std::vector<unsigned int> &);

template <typename HIST = Hist_t>
class FillTOHelper {
   std::unique_ptr<TThreadedObject<HIST>> fTo;

public:
   FillTOHelper(FillTOHelper &&) = default;
   FillTOHelper(const FillTOHelper &) = delete;

   FillTOHelper(const std::shared_ptr<HIST> &h, const unsigned int nSlots) : fTo(new TThreadedObject<HIST>(*h))
   {
      fTo->SetAtSlot(0, h);
      // Initialise all other slots
      for (unsigned int i = 0; i < nSlots; ++i) {
         fTo->GetAtSlot(i);
      }
   }

   void InitSlot(TTreeReader *, unsigned int) {}

   void Exec(unsigned int slot, double x0) // 1D histos
   {
      fTo->GetAtSlotRaw(slot)->Fill(x0);
   }

   void Exec(unsigned int slot, double x0, double x1) // 1D weighted and 2D histos
   {
      fTo->GetAtSlotRaw(slot)->Fill(x0, x1);
   }

   void Exec(unsigned int slot, double x0, double x1, double x2) // 2D weighted and 3D histos
   {
      fTo->GetAtSlotRaw(slot)->Fill(x0, x1, x2);
   }

   void Exec(unsigned int slot, double x0, double x1, double x2, double x3) // 3D weighted histos
   {
      fTo->GetAtSlotRaw(slot)->Fill(x0, x1, x2, x3);
   }

   template <typename X0, typename std::enable_if<IsContainer<X0>::value, int>::type = 0>
   void Exec(unsigned int slot, const X0 &x0s)
   {
      auto thisSlotH = fTo->GetAtSlotRaw(slot);
      for (auto &x0 : x0s) {
         thisSlotH->Fill(x0); // TODO: Can be optimised in case T == vector<double>
      }
   }

   template <typename X0, typename X1,
             typename std::enable_if<IsContainer<X0>::value && IsContainer<X1>::value, int>::type = 0>
   void Exec(unsigned int slot, const X0 &x0s, const X1 &x1s)
   {
      auto thisSlotH = fTo->GetAtSlotRaw(slot);
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
      auto thisSlotH = fTo->GetAtSlotRaw(slot);
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
      auto thisSlotH = fTo->GetAtSlotRaw(slot);
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

   HIST &PartialUpdate(unsigned int slot) { return *fTo->GetAtSlotRaw(slot); }
};

// In case of the take helper we have 4 cases:
// 1. The column is not an TArrayBranch, the collection is not a vector
// 2. The column is not an TArrayBranch, the collection is a vector
// 3. The column is an TArrayBranch, the collection is not a vector
// 4. The column is an TArrayBranch, the collection is a vector

// Case 1.: The column is not an TArrayBranch, the collection is not a vector
// No optimisations, no transformations: just copies.
template <typename RealT_t, typename T, typename COLL>
class TakeHelper {
   std::vector<std::shared_ptr<COLL>> fColls;

public:
   using BranchTypes_t = TypeList<T>;
   TakeHelper(const std::shared_ptr<COLL> &resultColl, const unsigned int nSlots)
   {
      fColls.emplace_back(resultColl);
      for (unsigned int i = 1; i < nSlots; ++i)
         fColls.emplace_back(std::make_shared<COLL>());
   }
   TakeHelper(TakeHelper &&) = default;
   TakeHelper(const TakeHelper &) = delete;

   void InitSlot(TTreeReader *, unsigned int) {}

   void Exec(unsigned int slot, T &v) { fColls[slot]->emplace_back(v); }

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

   COLL &PartialUpdate(unsigned int slot) { return *fColls[slot].get(); }
};

// Case 2.: The column is not an TArrayBranch, the collection is a vector
// Optimisations, no transformations: just copies.
template <typename RealT_t, typename T>
class TakeHelper<RealT_t, T, std::vector<T>> {
   std::vector<std::shared_ptr<std::vector<T>>> fColls;

public:
   using BranchTypes_t = TypeList<T>;
   TakeHelper(const std::shared_ptr<std::vector<T>> &resultColl, const unsigned int nSlots)
   {
      fColls.emplace_back(resultColl);
      for (unsigned int i = 1; i < nSlots; ++i) {
         auto v = std::make_shared<std::vector<T>>();
         v->reserve(1024);
         fColls.emplace_back(v);
      }
   }
   TakeHelper(TakeHelper &&) = default;
   TakeHelper(const TakeHelper &) = delete;

   void InitSlot(TTreeReader *, unsigned int) {}

   void Exec(unsigned int slot, T &v) { fColls[slot]->emplace_back(v); }

   // This is optimised to treat vectors
   void Finalize()
   {
      ULong64_t totSize = 0;
      for (auto &coll : fColls)
         totSize += coll->size();
      auto rColl = fColls[0];
      rColl->reserve(totSize);
      for (unsigned int i = 1; i < fColls.size(); ++i) {
         auto &coll = fColls[i];
         rColl->insert(rColl->end(), coll->begin(), coll->end());
      }
   }

   std::vector<T> &PartialUpdate(unsigned int slot) { return *fColls[slot]; }
};

// Case 3.: The column is a TArrayBranch, the collection is not a vector
// No optimisations, transformations from TArrayBranchs to vectors
template <typename RealT_t, typename COLL>
class TakeHelper<RealT_t, TArrayBranch<RealT_t>, COLL> {
   std::vector<std::shared_ptr<COLL>> fColls;

public:
   using BranchTypes_t = TypeList<TArrayBranch<RealT_t>>;
   TakeHelper(const std::shared_ptr<COLL> &resultColl, const unsigned int nSlots)
   {
      fColls.emplace_back(resultColl);
      for (unsigned int i = 1; i < nSlots; ++i)
         fColls.emplace_back(std::make_shared<COLL>());
   }
   TakeHelper(TakeHelper &&) = default;
   TakeHelper(const TakeHelper &) = delete;

   void InitSlot(TTreeReader *, unsigned int) {}

   void Exec(unsigned int slot, TArrayBranch<RealT_t> av) { fColls[slot]->emplace_back(av.begin(), av.end()); }

   void Finalize()
   {
      auto rColl = fColls[0];
      for (unsigned int i = 1; i < fColls.size(); ++i) {
         auto &coll = fColls[i];
         for (auto &v : *coll) {
            rColl->emplace_back(v);
         }
      }
   }
};

// Case 4.: The column is an TArrayBranch, the collection is a vector
// Optimisations, transformations from TArrayBranchs to vectors
template <typename RealT_t>
class TakeHelper<RealT_t, TArrayBranch<RealT_t>, std::vector<RealT_t>> {
   std::vector<std::shared_ptr<std::vector<std::vector<RealT_t>>>> fColls;

public:
   using BranchTypes_t = TypeList<TArrayBranch<RealT_t>>;
   TakeHelper(const std::shared_ptr<std::vector<std::vector<RealT_t>>> &resultColl, const unsigned int nSlots)
   {
      fColls.emplace_back(resultColl);
      for (unsigned int i = 1; i < nSlots; ++i) {
         auto v = std::make_shared<std::vector<RealT_t>>();
         v->reserve(1024);
         fColls.emplace_back(v);
      }
   }
   TakeHelper(TakeHelper &&) = default;
   TakeHelper(const TakeHelper &) = delete;

   void InitSlot(TTreeReader *, unsigned int) {}

   void Exec(unsigned int slot, TArrayBranch<RealT_t> av) { fColls[slot]->emplace_back(av.begin(), av.end()); }

   // This is optimised to treat vectors
   void Finalize()
   {
      ULong64_t totSize = 0;
      for (auto &coll : fColls)
         totSize += coll->size();
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
   ReduceHelper(F &&f, const std::shared_ptr<T> &reduceRes, const unsigned int nSlots)
      : fReduceFun(std::move(f)), fReduceRes(reduceRes), fReduceObjs(nSlots, *reduceRes)
   {
   }
   ReduceHelper(ReduceHelper &&) = default;
   ReduceHelper(const ReduceHelper &) = delete;

   void InitSlot(TTreeReader *, unsigned int) {}

   void Exec(unsigned int slot, const T &value) { fReduceObjs[slot] = fReduceFun(fReduceObjs[slot], value); }

   void Finalize()
   {
      for (auto &t : fReduceObjs)
         *fReduceRes = fReduceFun(*fReduceRes, t);
   }

   T &PartialUpdate(unsigned int slot) { return fReduceObjs[slot]; }
};

template <typename ResultType>
class MinHelper {
   const std::shared_ptr<ResultType> fResultMin;
   std::vector<ResultType> fMins;

public:
   MinHelper(MinHelper &&) = default;
   MinHelper(const std::shared_ptr<ResultType> &minVPtr, const unsigned int nSlots)
      : fResultMin(minVPtr), fMins(nSlots, std::numeric_limits<ResultType>::max()) {}

   void Exec(unsigned int slot, ResultType v) { fMins[slot] = std::min(v, fMins[slot]); }

   void InitSlot(TTreeReader *, unsigned int) {}

   template <typename T, typename std::enable_if<IsContainer<T>::value, int>::type = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      for (auto &&v : vs)
         fMins[slot] = std::min(v, fMins[slot]);
   }

   void Finalize()
   {
      *fResultMin = std::numeric_limits<ResultType>::max();
      for (auto &m : fMins)
         *fResultMin = std::min(m, *fResultMin);
   }

   ResultType &PartialUpdate(unsigned int slot) { return fMins[slot]; }
};

// TODO
// extern template void MinHelper::Exec(unsigned int, const std::vector<float> &);
// extern template void MinHelper::Exec(unsigned int, const std::vector<double> &);
// extern template void MinHelper::Exec(unsigned int, const std::vector<char> &);
// extern template void MinHelper::Exec(unsigned int, const std::vector<int> &);
// extern template void MinHelper::Exec(unsigned int, const std::vector<unsigned int> &);

template <typename ResultType>
class MaxHelper {
   const std::shared_ptr<ResultType> fResultMax;
   std::vector<ResultType> fMaxs;

public:
   MaxHelper(MaxHelper &&) = default;
   MaxHelper(const MaxHelper &) = delete;
   MaxHelper(const std::shared_ptr<ResultType> &maxVPtr, const unsigned int nSlots)
      : fResultMax(maxVPtr), fMaxs(nSlots, std::numeric_limits<ResultType>::lowest()) {}

   void InitSlot(TTreeReader *, unsigned int) {}
   void Exec(unsigned int slot, ResultType v) { fMaxs[slot] = std::max(v, fMaxs[slot]); }

   template <typename T, typename std::enable_if<IsContainer<T>::value, int>::type = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      for (auto &&v : vs)
         fMaxs[slot] = std::max((ResultType)v, fMaxs[slot]);
   }

   void Finalize()
   {
      *fResultMax = std::numeric_limits<ResultType>::lowest();
      for (auto &m : fMaxs) {
         *fResultMax = std::max(m, *fResultMax);
      }
   }

   ResultType &PartialUpdate(unsigned int slot) { return fMaxs[slot]; }
};

// TODO
// extern template void MaxHelper::Exec(unsigned int, const std::vector<float> &);
// extern template void MaxHelper::Exec(unsigned int, const std::vector<double> &);
// extern template void MaxHelper::Exec(unsigned int, const std::vector<char> &);
// extern template void MaxHelper::Exec(unsigned int, const std::vector<int> &);
// extern template void MaxHelper::Exec(unsigned int, const std::vector<unsigned int> &);

template <typename ResultType>
class SumHelper {
   const std::shared_ptr<ResultType> fResultSum;
   std::vector<ResultType> fSums;

public:
   SumHelper(SumHelper &&) = default;
   SumHelper(const SumHelper &) = delete;
   SumHelper(const std::shared_ptr<ResultType> &sumVPtr, const unsigned int nSlots)
      : fResultSum(sumVPtr), fSums(nSlots, *sumVPtr - *sumVPtr) {}

   void InitSlot(TTreeReader *, unsigned int) {}
   void Exec(unsigned int slot, ResultType v) { fSums[slot] += v; }

   template <typename T, typename std::enable_if<IsContainer<T>::value, int>::type = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      for (auto &&v : vs)
         fSums[slot] += static_cast<ResultType>(v);
   }

   void Finalize()
   {
      for (auto &m : fSums)
         *fResultSum += m;
   }

   ResultType &PartialUpdate(unsigned int slot) { return fSums[slot]; }
};

class MeanHelper {
   const std::shared_ptr<double> fResultMean;
   std::vector<ULong64_t> fCounts;
   std::vector<double> fSums;
   std::vector<double> fPartialMeans;

public:
   MeanHelper(const std::shared_ptr<double> &meanVPtr, const unsigned int nSlots);
   MeanHelper(MeanHelper &&) = default;
   MeanHelper(const MeanHelper &) = delete;
   void InitSlot(TTreeReader *, unsigned int) {}
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

   double &PartialUpdate(unsigned int slot);
};

extern template void MeanHelper::Exec(unsigned int, const std::vector<float> &);
extern template void MeanHelper::Exec(unsigned int, const std::vector<double> &);
extern template void MeanHelper::Exec(unsigned int, const std::vector<char> &);
extern template void MeanHelper::Exec(unsigned int, const std::vector<int> &);
extern template void MeanHelper::Exec(unsigned int, const std::vector<unsigned int> &);

template<typename T>
struct AddRefIfNotArrayBranch {
   using type = T&;
};

template<typename T>
struct AddRefIfNotArrayBranch<TArrayBranch<T>> {
   using type = TArrayBranch<T>;
};

template<typename T>
using AddRefIfNotArrayBranch_t = typename AddRefIfNotArrayBranch<T>::type;

/// Helper object for a single-thread Snapshot action
template <typename... BranchTypes>
class SnapshotHelper {
   std::unique_ptr<TFile> fOutputFile;
   std::unique_ptr<TTree> fOutputTree; // must be a ptr because TTrees are not copy/move constructible
   bool fIsFirstEvent{true};
   const ColumnNames_t fBranchNames;
   TTree *fInputTree = nullptr; // Current input tree. Set at initialization time (`InitSlot`)

public:
   SnapshotHelper(std::string_view filename, std::string_view dirname, std::string_view treename,
                  const ColumnNames_t &bnames, const TSnapshotOptions &options)
      : fOutputFile(TFile::Open(std::string(filename).c_str(), options.fMode.c_str(), /*ftitle=*/"",
                                ROOT::CompressionSettings(options.fCompressionAlgorithm, options.fCompressionLevel))),
        fBranchNames(bnames)
   {
      if (!dirname.empty()) {
         std::string dirnameStr(dirname);
         fOutputFile->mkdir(dirnameStr.c_str());
         fOutputFile->cd(dirnameStr.c_str());
      }
      std::string treenameStr(treename);
      fOutputTree.reset(
         new TTree(treenameStr.c_str(), treenameStr.c_str(), options.fSplitLevel, /*dir=*/fOutputFile.get()));

      if (options.fAutoFlush)
         fOutputTree->SetAutoFlush(options.fAutoFlush);
   }

   SnapshotHelper(const SnapshotHelper &) = delete;
   SnapshotHelper(SnapshotHelper &&) = default;

   void InitSlot(TTreeReader *r, unsigned int /* slot */)
   {
      if (!r) // empty source, nothing to do
         return;
      fInputTree = r->GetTree();
      // AddClone guarantees that if the input file changes the branches of the output tree are updated with the new
      // addresses of the branch values
      fInputTree->AddClone(fOutputTree.get());
   }

   void Exec(unsigned int /* slot */, AddRefIfNotArrayBranch_t<BranchTypes>... values)
   {
      if (fIsFirstEvent) {
         using ind_t = GenStaticSeq_t<sizeof...(BranchTypes)>;
         SetBranches(values..., ind_t());
      }
      fOutputTree->Fill();
   }

   template <int... S>
   void SetBranches(AddRefIfNotArrayBranch_t<BranchTypes>... values, StaticSeq<S...> /*dummy*/)
   {
      // hack to call TTree::Branch on all variadic template arguments
      int expander[] = {(SetBranchesHelper(fBranchNames[S], &values), 0)..., 0};
      (void)expander; // avoid unused variable warnings for older compilers such as gcc 4.9
      fIsFirstEvent = false;
   }

   template <typename T>
   void SetBranchesHelper(const std::string &name, T *address)
   {
      fOutputTree->Branch(name.c_str(), address);
   }

   // This overload is called for columns of type `TArrayBranch<T>`. For TDF, these represent c-style arrays in ROOT
   // files, so we are sure that there are input trees to which we can ask the correct branch title
   template <typename T>
   void SetBranchesHelper(const std::string &name, TArrayBranch<T> *ab)
   {
      fOutputTree->Branch(name.c_str(), ab->GetData(), fInputTree->GetBranch(name.c_str())->GetTitle());
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
   std::vector<int> fIsFirstEvent;    // vector<bool> is evil
   const std::string fDirName;        // name of TFile subdirectory in which output must be written (possibly empty)
   const std::string fTreeName;       // name of output tree
   const TSnapshotOptions fOptions;   // struct holding options to pass down to TFile and TTree in this action
   const ColumnNames_t fBranchNames;
   std::vector<TTree *> fInputTrees; // Current input trees. Set at initialization time (`InitSlot`)

public:
   using BranchTypes_t = TypeList<BranchTypes...>;
   SnapshotHelperMT(const unsigned int nSlots, std::string_view filename, std::string_view dirname,
                    std::string_view treename, const ColumnNames_t &bnames, const TSnapshotOptions &options)
      : fNSlots(nSlots), fMerger(new ROOT::Experimental::TBufferMerger(
                            std::string(filename).c_str(), options.fMode.c_str(),
                            ROOT::CompressionSettings(options.fCompressionAlgorithm, options.fCompressionLevel))),
        fOutputFiles(fNSlots), fOutputTrees(fNSlots, nullptr), fIsFirstEvent(fNSlots, 1), fDirName(dirname),
        fTreeName(treename), fOptions(options), fBranchNames(bnames), fInputTrees(fNSlots)
   {
   }
   SnapshotHelperMT(const SnapshotHelperMT &) = delete;
   SnapshotHelperMT(SnapshotHelperMT &&) = default;

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
      // re-create output tree as we need to create its branches again, with new input variables
      // TODO we could instead create the output tree and its branches, change addresses of input variables in each task
      fOutputTrees[slot] = new TTree(fTreeName.c_str(), fTreeName.c_str(), fOptions.fSplitLevel, /*dir=*/treeDirectory);
      fOutputTrees[slot]->ResetBit(kMustCleanup); // do not mingle with the thread-unsafe gListOfCleanups
      if (fOptions.fAutoFlush)
         fOutputTrees[slot]->SetAutoFlush(fOptions.fAutoFlush);
      if (r) {
         // not an empty-source TDF
         fInputTrees[slot] = r->GetTree();
         // AddClone guarantees that if the input file changes the branches of the output tree are updated with the new
         // addresses of the branch values
         fInputTrees[slot]->AddClone(fOutputTrees[slot]);
      }
      fIsFirstEvent[slot] = 1; // reset first event flag for this slot
   }

   void Exec(unsigned int slot, AddRefIfNotArrayBranch_t<BranchTypes>... values)
   {
      if (fIsFirstEvent[slot]) {
         using ind_t = GenStaticSeq_t<sizeof...(BranchTypes)>;
         SetBranches(slot, values..., ind_t());
         fIsFirstEvent[slot] = 0;
      }
      fOutputTrees[slot]->Fill();
      auto entries = fOutputTrees[slot]->GetEntries();
      auto autoFlush = fOutputTrees[slot]->GetAutoFlush();
      if ((autoFlush > 0) && (entries % autoFlush == 0))
         fOutputFiles[slot]->Write();
   }

   template <int... S>
   void SetBranches(unsigned int slot, AddRefIfNotArrayBranch_t<BranchTypes>... values, StaticSeq<S...> /*dummy*/)
   {
      // hack to call TTree::Branch on all variadic template arguments
      int expander[] = {(SetBranchesHelper(*fOutputTrees[slot], *fInputTrees[slot], fBranchNames[S], &values), 0)...,
                        0};
      (void)expander; // avoid unused variable warnings for older compilers such as gcc 4.9
   }

   template <typename T>
   void SetBranchesHelper(TTree &t, TTree &, const std::string &name, T *address)
   {
      t.Branch(name.c_str(), address);
   }

   // This overload is called for columns of type `TArrayBranch<T>`. For TDF, these represent c-style arrays in ROOT
   // files, so we are sure that there are input trees to which we can ask the correct branch title
   template <typename T>
   void SetBranchesHelper(TTree &outputTree, TTree &inputTree, const std::string &name, TArrayBranch<T> *ab)
   {
      outputTree.Branch(name.c_str(), ab->GetData(), inputTree.GetBranch(name.c_str())->GetTitle());
   }

   void Finalize()
   {
      for (auto &file : fOutputFiles) {
         if (file)
            file->Write();
      }
   }
};

} // end of NS TDF
} // end of NS Internal
} // end of NS ROOT

/// \endcond

#endif
