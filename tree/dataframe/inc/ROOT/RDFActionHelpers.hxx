// Author: Enrico Guiraud, Danilo Piparo CERN  12/2016

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDFOPERATIONS
#define ROOT_RDFOPERATIONS

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "Compression.h"
#include "ROOT/RIntegerSequence.hxx"
#include "ROOT/RStringView.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/TBufferMerger.hxx" // for SnapshotHelper
#include "ROOT/RCutFlowReport.hxx"
#include "ROOT/RDFUtils.hxx"
#include "ROOT/RSnapshotOptions.hxx"
#include "ROOT/TThreadedObject.hxx"
#include "ROOT/TypeTraits.hxx"
#include "RtypesCore.h"
#include "TBranch.h"
#include "TClassEdit.h"
#include "TDirectory.h"
#include "TFile.h" // for SnapshotHelper
#include "TH1.h"
#include "TLeaf.h"
#include "TObjArray.h"
#include "TObject.h"
#include "TTree.h"
#include "TTreeReader.h" // for SnapshotHelper

/// \cond HIDDEN_SYMBOLS

namespace ROOT {
namespace Detail {
namespace RDF {

template <typename Helper>
class RActionImpl {
public:
   void InitSlot(TTreeReader *r, unsigned int slot) { static_cast<Helper &>(*this).InitTask(r, slot); }
};

} // namespace RDF
} // namespace Detail

namespace Internal {
namespace RDF {
using namespace ROOT::TypeTraits;
using namespace ROOT::VecOps;
using namespace ROOT::RDF;

using Hist_t = ::TH1D;

template <typename F>
class ForeachSlotHelper {
   F fCallable;

public:
   using ColumnTypes_t = RemoveFirstParameter_t<typename CallableTraits<F>::arg_types>;
   ForeachSlotHelper(F &&f) : fCallable(f) {}
   ForeachSlotHelper(ForeachSlotHelper &&) = default;
   ForeachSlotHelper(const ForeachSlotHelper &) = delete;

   void InitSlot(TTreeReader *, unsigned int) {}

   template <typename... Args>
   void Exec(unsigned int slot, Args &&... args)
   {
      // check that the decayed types of Args are the same as the branch types
      static_assert(std::is_same<TypeList<typename std::decay<Args>::type...>, ColumnTypes_t>::value, "");
      fCallable(slot, std::forward<Args>(args)...);
   }

   void Initialize() { /* noop */}

   void Finalize() { /* noop */}
};

class CountHelper {
   const std::shared_ptr<ULong64_t> fResultCount;
   std::vector<ULong64_t> fCounts;

public:
   using ColumnTypes_t = TypeList<>;
   CountHelper(const std::shared_ptr<ULong64_t> &resultCount, const unsigned int nSlots);
   CountHelper(CountHelper &&) = default;
   CountHelper(const CountHelper &) = delete;
   void InitSlot(TTreeReader *, unsigned int) {}
   void Exec(unsigned int slot);
   void Initialize() { /* noop */}
   void Finalize();
   ULong64_t &PartialUpdate(unsigned int slot);
};

template <typename ProxiedVal_t>
class ReportHelper {
   const std::shared_ptr<RCutFlowReport> fReport;
   // Here we have a weak pointer since we need to keep track of the validity
   // of the proxied node. It can happen that the user does not trigger the
   // event loop by looking into the RResultPtr and the chain goes out of scope
   // before the Finalize method is invoked.
   std::weak_ptr<ProxiedVal_t> fProxiedWPtr;
   bool fReturnEmptyReport;

public:
   using ColumnTypes_t = TypeList<>;
   ReportHelper(const std::shared_ptr<RCutFlowReport> &report, const std::shared_ptr<ProxiedVal_t> &pp, bool emptyRep)
      : fReport(report), fProxiedWPtr(pp), fReturnEmptyReport(emptyRep){};
   ReportHelper(ReportHelper &&) = default;
   ReportHelper(const ReportHelper &) = delete;
   void InitSlot(TTreeReader *, unsigned int) {}
   void Exec(unsigned int /* slot */) {}
   void Initialize() { /* noop */}
   void Finalize()
   {
      // We need the weak_ptr in order to avoid crashes at tear down
      if (!fReturnEmptyReport && !fProxiedWPtr.expired())
         fProxiedWPtr.lock()->Report(*fReport);
   }
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

   void Initialize() { /* noop */}

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

   void Initialize() { /* noop */}

   void Finalize() { fTo->Merge(); }

   HIST &PartialUpdate(unsigned int slot) { return *fTo->GetAtSlotRaw(slot); }
};

// In case of the take helper we have 4 cases:
// 1. The column is not an RVec, the collection is not a vector
// 2. The column is not an RVec, the collection is a vector
// 3. The column is an RVec, the collection is not a vector
// 4. The column is an RVec, the collection is a vector

// Case 1.: The column is not an RVec, the collection is not a vector
// No optimisations, no transformations: just copies.
template <typename RealT_t, typename T, typename COLL>
class TakeHelper {
   std::vector<std::shared_ptr<COLL>> fColls;

public:
   using ColumnTypes_t = TypeList<T>;
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

   void Initialize() { /* noop */}

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

// Case 2.: The column is not an RVec, the collection is a vector
// Optimisations, no transformations: just copies.
template <typename RealT_t, typename T>
class TakeHelper<RealT_t, T, std::vector<T>> {
   std::vector<std::shared_ptr<std::vector<T>>> fColls;

public:
   using ColumnTypes_t = TypeList<T>;
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

   void Initialize() { /* noop */}

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

// Case 3.: The column is a RVec, the collection is not a vector
// No optimisations, transformations from RVecs to vectors
template <typename RealT_t, typename COLL>
class TakeHelper<RealT_t, RVec<RealT_t>, COLL> {
   std::vector<std::shared_ptr<COLL>> fColls;

public:
   using ColumnTypes_t = TypeList<RVec<RealT_t>>;
   TakeHelper(const std::shared_ptr<COLL> &resultColl, const unsigned int nSlots)
   {
      fColls.emplace_back(resultColl);
      for (unsigned int i = 1; i < nSlots; ++i)
         fColls.emplace_back(std::make_shared<COLL>());
   }
   TakeHelper(TakeHelper &&) = default;
   TakeHelper(const TakeHelper &) = delete;

   void InitSlot(TTreeReader *, unsigned int) {}

   void Exec(unsigned int slot, RVec<RealT_t> av) { fColls[slot]->emplace_back(av.begin(), av.end()); }

   void Initialize() { /* noop */}

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

// Case 4.: The column is an RVec, the collection is a vector
// Optimisations, transformations from RVecs to vectors
template <typename RealT_t>
class TakeHelper<RealT_t, RVec<RealT_t>, std::vector<RealT_t>> {
   std::vector<std::shared_ptr<std::vector<std::vector<RealT_t>>>> fColls;

public:
   using ColumnTypes_t = TypeList<RVec<RealT_t>>;
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

   void Exec(unsigned int slot, RVec<RealT_t> av) { fColls[slot]->emplace_back(av.begin(), av.end()); }

   void Initialize() { /* noop */}

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

template <typename ResultType>
class MinHelper {
   const std::shared_ptr<ResultType> fResultMin;
   std::vector<ResultType> fMins;

public:
   MinHelper(MinHelper &&) = default;
   MinHelper(const std::shared_ptr<ResultType> &minVPtr, const unsigned int nSlots)
      : fResultMin(minVPtr), fMins(nSlots, std::numeric_limits<ResultType>::max())
   {
   }

   void Exec(unsigned int slot, ResultType v) { fMins[slot] = std::min(v, fMins[slot]); }

   void InitSlot(TTreeReader *, unsigned int) {}

   template <typename T, typename std::enable_if<IsContainer<T>::value, int>::type = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      for (auto &&v : vs)
         fMins[slot] = std::min(v, fMins[slot]);
   }

   void Initialize() { /* noop */}

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
      : fResultMax(maxVPtr), fMaxs(nSlots, std::numeric_limits<ResultType>::lowest())
   {
   }

   void InitSlot(TTreeReader *, unsigned int) {}
   void Exec(unsigned int slot, ResultType v) { fMaxs[slot] = std::max(v, fMaxs[slot]); }

   template <typename T, typename std::enable_if<IsContainer<T>::value, int>::type = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      for (auto &&v : vs)
         fMaxs[slot] = std::max((ResultType)v, fMaxs[slot]);
   }

   void Initialize() { /* noop */}

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
      : fResultSum(sumVPtr), fSums(nSlots, *sumVPtr - *sumVPtr)
   {
   }

   void InitSlot(TTreeReader *, unsigned int) {}
   void Exec(unsigned int slot, ResultType v) { fSums[slot] += v; }

   template <typename T, typename std::enable_if<IsContainer<T>::value, int>::type = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      for (auto &&v : vs)
         fSums[slot] += static_cast<ResultType>(v);
   }

   void Initialize() { /* noop */}

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

   void Initialize() { /* noop */}

   void Finalize();

   double &PartialUpdate(unsigned int slot);
};

extern template void MeanHelper::Exec(unsigned int, const std::vector<float> &);
extern template void MeanHelper::Exec(unsigned int, const std::vector<double> &);
extern template void MeanHelper::Exec(unsigned int, const std::vector<char> &);
extern template void MeanHelper::Exec(unsigned int, const std::vector<int> &);
extern template void MeanHelper::Exec(unsigned int, const std::vector<unsigned int> &);

/// Helper function for SnapshotHelper and SnapshotHelperMT. It creates new branches for the output TTree of a Snapshot.
template <typename T>
void SetBranchesHelper(TTree * /*inputTree*/, TTree &outputTree, const std::string & /*validName*/,
                       const std::string &name, T *address)
{
   outputTree.Branch(name.c_str(), address);
}

/// Helper function for SnapshotHelper and SnapshotHelperMT. It creates new branches for the output TTree of a Snapshot.
/// This overload is called for columns of type `RVec<T>`. For RDF, these can represent:
/// 1. c-style arrays in ROOT files, so we are sure that there are input trees to which we can ask the correct branch title
/// 2. RVecs coming from a custom column or a source
/// 3. vectors coming from ROOT files
template <typename T>
void SetBranchesHelper(TTree *inputTree, TTree &outputTree, const std::string &validName, const std::string &name,
                       RVec<T> *ab)
{
   // Treat 2. and 3.:
   // 2. RVec coming from a custom column or a source
   // 3. RVec coming from a column on disk of type vector (the RVec is adopting the data of that vector)
   auto *const inputBranch = inputTree ? inputTree->GetBranch(validName.c_str()) : nullptr;
   auto mustWriteRVec =
      !inputBranch || ROOT::ESTLType::kSTLvector == TClassEdit::IsSTLCont(inputBranch->GetClassName());
   if (mustWriteRVec) {
      outputTree.Branch(name.c_str(), reinterpret_cast<typename RVec<T>::Impl_t *>(ab));
      return;
   }

   // Treat 1, the C-array case
   auto *const leaf = static_cast<TLeaf *>(inputBranch->GetListOfLeaves()->UncheckedAt(0));
   const auto bname = leaf->GetName();
   const auto counterStr =
      leaf->GetLeafCount() ? std::string(leaf->GetLeafCount()->GetName()) : std::to_string(leaf->GetLenStatic());
   const auto btype = leaf->GetTypeName();
   const auto rootbtype = TypeName2ROOTTypeName(btype);
   const auto leaflist = std::string(bname) + "[" + counterStr + "]/" + rootbtype;
   auto *const outputBranch = outputTree.Branch(name.c_str(), ab->data(), leaflist.c_str());
   outputBranch->SetTitle(inputBranch->GetTitle());
}

/// Helper object for a single-thread Snapshot action
template <typename... BranchTypes>
class SnapshotHelper {
   const std::string fFileName;
   const std::string fDirName;
   const std::string fTreeName;
   const RSnapshotOptions fOptions;
   std::unique_ptr<TFile> fOutputFile;
   std::unique_ptr<TTree> fOutputTree; // must be a ptr because TTrees are not copy/move constructible
   bool fIsFirstEvent{true};
   const ColumnNames_t fInputBranchNames; // This contains the resolved aliases
   const ColumnNames_t fOutputBranchNames;
   TTree *fInputTree = nullptr; // Current input tree. Set at initialization time (`InitSlot`)

public:
   SnapshotHelper(std::string_view filename, std::string_view dirname, std::string_view treename,
                  const ColumnNames_t &vbnames, const ColumnNames_t &bnames, const RSnapshotOptions &options)
      : fFileName(filename), fDirName(dirname), fTreeName(treename), fOptions(options), fInputBranchNames(vbnames),
        fOutputBranchNames(ReplaceDotWithUnderscore(bnames))
   {
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

   void Exec(unsigned int /* slot */, BranchTypes &... values)
   {
      if (fIsFirstEvent) {
         using ind_t = std::index_sequence_for<BranchTypes...>;
         SetBranches(values..., ind_t());
      }
      fOutputTree->Fill();
   }

   template <std::size_t... S>
   void SetBranches(BranchTypes &... values, std::index_sequence<S...> /*dummy*/)
   {
      // call TTree::Branch on all variadic template arguments
      int expander[] = {
         (SetBranchesHelper(fInputTree, *fOutputTree, fInputBranchNames[S], fOutputBranchNames[S], &values), 0)..., 0};
      (void)expander; // avoid unused variable warnings for older compilers such as gcc 4.9
      fIsFirstEvent = false;
   }

   void Initialize()
   {
      fOutputFile.reset(
         TFile::Open(fFileName.c_str(), fOptions.fMode.c_str(), /*ftitle=*/"",
                     ROOT::CompressionSettings(fOptions.fCompressionAlgorithm, fOptions.fCompressionLevel)));

      if (!fDirName.empty()) {
         fOutputFile->mkdir(fDirName.c_str());
         fOutputFile->cd(fDirName.c_str());
      }

      fOutputTree.reset(
         new TTree(fTreeName.c_str(), fTreeName.c_str(), fOptions.fSplitLevel, /*dir=*/fOutputFile.get()));

      if (fOptions.fAutoFlush)
         fOutputTree->SetAutoFlush(fOptions.fAutoFlush);
   }

   void Finalize()
   {
      ::TDirectory::TContext ctxt(fOutputFile->GetDirectory(fDirName.c_str()));
      fOutputTree->Write();
   }
};

/// Helper object for a multi-thread Snapshot action
template <typename... BranchTypes>
class SnapshotHelperMT {
   const unsigned int fNSlots;
   std::unique_ptr<ROOT::Experimental::TBufferMerger> fMerger; // must use a ptr because TBufferMerger is not movable
   std::vector<std::shared_ptr<ROOT::Experimental::TBufferMergerFile>> fOutputFiles;
   std::vector<TTree *> fOutputTrees;     // ROOT will own/manage these TTrees, must not delete
   std::vector<int> fIsFirstEvent;        // vector<bool> is evil
   const std::string fFileName;           // name of the output file name
   const std::string fDirName;            // name of TFile subdirectory in which output must be written (possibly empty)
   const std::string fTreeName;           // name of output tree
   const RSnapshotOptions fOptions;       // struct holding options to pass down to TFile and TTree in this action
   const ColumnNames_t fInputBranchNames; // This contains the resolved aliases
   const ColumnNames_t fOutputBranchNames;
   std::vector<TTree *> fInputTrees; // Current input trees. Set at initialization time (`InitSlot`)

public:
   using ColumnTypes_t = TypeList<BranchTypes...>;
   SnapshotHelperMT(const unsigned int nSlots, std::string_view filename, std::string_view dirname,
                    std::string_view treename, const ColumnNames_t &vbnames, const ColumnNames_t &bnames,
                    const RSnapshotOptions &options)
      : fNSlots(nSlots), fOutputFiles(fNSlots), fOutputTrees(fNSlots, nullptr), fIsFirstEvent(fNSlots, 1),
        fFileName(filename), fDirName(dirname), fTreeName(treename), fOptions(options), fInputBranchNames(vbnames),
        fOutputBranchNames(ReplaceDotWithUnderscore(bnames)), fInputTrees(fNSlots)
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
         // not an empty-source RDF
         fInputTrees[slot] = r->GetTree();
         // AddClone guarantees that if the input file changes the branches of the output tree are updated with the new
         // addresses of the branch values
         fInputTrees[slot]->AddClone(fOutputTrees[slot]);
      }
      fIsFirstEvent[slot] = 1; // reset first event flag for this slot
   }

   void Exec(unsigned int slot, BranchTypes &... values)
   {
      if (fIsFirstEvent[slot]) {
         using ind_t = std::index_sequence_for<BranchTypes...>;
         SetBranches(slot, values..., ind_t());
         fIsFirstEvent[slot] = 0;
      }
      fOutputTrees[slot]->Fill();
      auto entries = fOutputTrees[slot]->GetEntries();
      auto autoFlush = fOutputTrees[slot]->GetAutoFlush();
      if ((autoFlush > 0) && (entries % autoFlush == 0))
         fOutputFiles[slot]->Write();
   }

   template <std::size_t... S>
   void SetBranches(unsigned int slot, BranchTypes &... values, std::index_sequence<S...> /*dummy*/)
   {
      // hack to call TTree::Branch on all variadic template arguments
      int expander[] = {(SetBranchesHelper(fInputTrees[slot], *fOutputTrees[slot], fInputBranchNames[S],
                                           fOutputBranchNames[S], &values),
                         0)...,
                        0};
      (void)expander; // avoid unused variable warnings for older compilers such as gcc 4.9
      (void)slot;     // avoid unused variable warnings in gcc6.2
   }

   void Initialize()
   {
      const auto cs = ROOT::CompressionSettings(fOptions.fCompressionAlgorithm, fOptions.fCompressionLevel);
      fMerger.reset(new ROOT::Experimental::TBufferMerger(fFileName.c_str(), fOptions.fMode.c_str(), cs));
   }

   void Finalize()
   {
      for (auto &file : fOutputFiles) {
         if (file)
            file->Write();
      }
   }
};

template <typename Acc, typename Merge, typename R, typename T, typename U,
          bool MustCopyAssign = std::is_same<R, U>::value>
class AggregateHelper {
   Acc fAggregate;
   Merge fMerge;
   const std::shared_ptr<U> fResult;
   std::vector<U> fAggregators;

public:
   using ColumnTypes_t = TypeList<T>;
   AggregateHelper(Acc &&f, Merge &&m, const std::shared_ptr<U> &result, const unsigned int nSlots)
      : fAggregate(std::move(f)), fMerge(std::move(m)), fResult(result), fAggregators(nSlots, *result)
   {
   }
   AggregateHelper(AggregateHelper &&) = default;
   AggregateHelper(const AggregateHelper &) = delete;

   void InitSlot(TTreeReader *, unsigned int) {}

   template <bool MustCopyAssign_ = MustCopyAssign, typename std::enable_if<MustCopyAssign_, int>::type = 0>
   void Exec(unsigned int slot, const T &value)
   {
      fAggregators[slot] = fAggregate(fAggregators[slot], value);
   }

   template <bool MustCopyAssign_ = MustCopyAssign, typename std::enable_if<!MustCopyAssign_, int>::type = 0>
   void Exec(unsigned int slot, const T &value)
   {
      fAggregate(fAggregators[slot], value);
   }

   void Initialize() { /* noop */}

   template <typename MergeRet = typename CallableTraits<Merge>::ret_type,
             bool MergeAll = std::is_same<void, MergeRet>::value>
   typename std::enable_if<MergeAll, void>::type Finalize()
   {
      fMerge(fAggregators);
      *fResult = fAggregators[0];
   }

   template <typename MergeRet = typename CallableTraits<Merge>::ret_type,
             bool MergeTwoByTwo = std::is_same<U, MergeRet>::value>
   typename std::enable_if<MergeTwoByTwo, void>::type Finalize(...) // ... needed to let compiler distinguish overloads
   {
      for (auto &acc : fAggregators)
         *fResult = fMerge(*fResult, acc);
   }

   U &PartialUpdate(unsigned int slot) { return fAggregators[slot]; }
};

} // end of NS RDF
} // end of NS Internal
} // end of NS ROOT

/// \endcond

#endif
