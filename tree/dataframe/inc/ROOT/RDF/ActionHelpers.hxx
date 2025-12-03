/**
 \file ROOT/RDF/ActionHelpers.hxx
 \ingroup dataframe
 \author Enrico Guiraud, CERN
 \author Danilo Piparo, CERN
 \date 2016-12
 \author Vincenzo Eduardo Padulano
 \date 2020-06
*/

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDFOPERATIONS
#define ROOT_RDFOPERATIONS

#include "ROOT/RVec.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "ROOT/TypeTraits.hxx"
#include "ROOT/RDF/RDisplay.hxx"
#include "RtypesCore.h"
#include "TH1.h"
#include "TH3.h"
#include "TGraph.h"
#include "TGraphAsymmErrors.h"
#include "TObject.h"
#include "ROOT/RDF/RActionImpl.hxx"
#include "ROOT/RDF/RMergeableValue.hxx"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility> // std::index_sequence
#include <vector>
#include <numeric> // std::accumulate in MeanHelper

class TCollection;
class TStatistic;
class TTreeReader;
namespace ROOT::RDF {
class RCutFlowReport;
} // namespace ROOT::RDF

/// \cond HIDDEN_SYMBOLS

namespace ROOT {
namespace Internal {
namespace RDF {
using namespace ROOT::TypeTraits;
using namespace ROOT::VecOps;
using namespace ROOT::RDF;
using namespace ROOT::Detail::RDF;

using Hist_t = ::TH1D;

/// The container type for each thread's partial result in an action helper
// We have to avoid to instantiate std::vector<bool> as that makes it impossible to return a reference to one of
// the thread-local results. In addition, a common definition for the type of the container makes it easy to swap
// the type of the underlying container if e.g. we see problems with false sharing of the thread-local results..
template <typename T>
using Results = std::conditional_t<std::is_same<T, bool>::value, std::deque<T>, std::vector<T>>;

template <typename F>
class R__CLING_PTRCHECK(off) ForeachSlotHelper : public RActionImpl<ForeachSlotHelper<F>> {
   F fCallable;

public:
   using ColumnTypes_t = RemoveFirstParameter_t<typename CallableTraits<F>::arg_types>;
   ForeachSlotHelper(F &&f) : fCallable(f) {}
   ForeachSlotHelper(ForeachSlotHelper &&) = default;
   ForeachSlotHelper(const ForeachSlotHelper &) = delete;

   void InitTask(TTreeReader *, unsigned int) {}

   template <typename... Args>
   void Exec(unsigned int slot, Args &&... args)
   {
      // check that the decayed types of Args are the same as the branch types
      static_assert(std::is_same<TypeList<std::decay_t<Args>...>, ColumnTypes_t>::value, "");
      fCallable(slot, std::forward<Args>(args)...);
   }

   void Initialize() { /* noop */}

   void Finalize() { /* noop */}

   std::string GetActionName() { return "ForeachSlot"; }
};

class R__CLING_PTRCHECK(off) CountHelper : public RActionImpl<CountHelper> {
   std::shared_ptr<ULong64_t> fResultCount;
   Results<ULong64_t> fCounts;

public:
   using ColumnTypes_t = TypeList<>;
   CountHelper(const std::shared_ptr<ULong64_t> &resultCount, const unsigned int nSlots);
   CountHelper(CountHelper &&) = default;
   CountHelper(const CountHelper &) = delete;
   void InitTask(TTreeReader *, unsigned int) {}
   void Exec(unsigned int slot);
   void Initialize() { /* noop */}
   void Finalize();

   // Helper functions for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      return std::make_unique<RMergeableCount>(*fResultCount);
   }

   ULong64_t &PartialUpdate(unsigned int slot);

   std::string GetActionName() { return "Count"; }

   CountHelper MakeNew(void *newResult, std::string_view /*variation*/ = "nominal")
   {
      auto &result = *static_cast<std::shared_ptr<ULong64_t> *>(newResult);
      return CountHelper(result, fCounts.size());
   }
};

template <typename RNode_t>
class R__CLING_PTRCHECK(off) ReportHelper : public RActionImpl<ReportHelper<RNode_t>> {
   std::shared_ptr<RCutFlowReport> fReport;
   /// Non-owning pointer, never null. As usual, the node is owned by its children nodes (and therefore indirectly by
   /// the RAction corresponding to this action helper).
   RNode_t *fNode;
   bool fReturnEmptyReport;

public:
   using ColumnTypes_t = TypeList<>;
   ReportHelper(const std::shared_ptr<RCutFlowReport> &report, RNode_t *node, bool emptyRep)
      : fReport(report), fNode(node), fReturnEmptyReport(emptyRep){};
   ReportHelper(ReportHelper &&) = default;
   ReportHelper(const ReportHelper &) = delete;
   void InitTask(TTreeReader *, unsigned int) {}
   void Exec(unsigned int /* slot */) {}
   void Initialize() { /* noop */}
   void Finalize()
   {
      if (!fReturnEmptyReport)
         fNode->Report(*fReport);
   }

   std::string GetActionName() { return "Report"; }

   ReportHelper MakeNew(void *newResult, std::string_view variation = "nominal")
   {
      auto &&result = *static_cast<std::shared_ptr<RCutFlowReport> *>(newResult);
      return ReportHelper{result,
                          std::static_pointer_cast<RNode_t>(fNode->GetVariedFilter(std::string(variation))).get(),
                          fReturnEmptyReport};
   }
};

/// This helper fills TH1Ds for which no axes were specified by buffering the fill values to pick good axes limits.
///
/// TH1Ds have an automatic mechanism to pick good limits based on the first N entries they were filled with, but
/// that does not work in multi-thread event loops as it might yield histograms with incompatible binning in each
/// thread, making it impossible to merge the per-thread results.
/// Instead, this helper delays the decision on the axes limits until all threads have done processing, synchronizing
/// the decision on the limits as part of the merge operation.
class R__CLING_PTRCHECK(off) BufferedFillHelper : public RActionImpl<BufferedFillHelper> {
   // this sets a total initial size of 16 MB for the buffers (can increase)
   static constexpr unsigned int fgTotalBufSize = 2097152;
   using BufEl_t = double;
   using Buf_t = std::vector<BufEl_t>;

   std::vector<Buf_t> fBuffers;
   std::vector<Buf_t> fWBuffers;
   std::shared_ptr<Hist_t> fResultHist;
   unsigned int fNSlots;
   unsigned int fBufSize;
   /// Histograms containing "snapshots" of partial results. Non-null only if a registered callback requires it.
   Results<std::unique_ptr<Hist_t>> fPartialHists;
   Buf_t fMin;
   Buf_t fMax;

   void UpdateMinMax(unsigned int slot, double v);

public:
   BufferedFillHelper(const std::shared_ptr<Hist_t> &h, const unsigned int nSlots);
   BufferedFillHelper(BufferedFillHelper &&) = default;
   BufferedFillHelper(const BufferedFillHelper &) = delete;
   void InitTask(TTreeReader *, unsigned int) {}
   void Exec(unsigned int slot, double v);
   void Exec(unsigned int slot, double v, double w);

   template <typename T, std::enable_if_t<IsDataContainer<T>::value, int> = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      auto &thisBuf = fBuffers[slot];
      // range-based for results in warnings on some compilers due to vector<bool>'s custom reference type
      for (auto v = vs.begin(); v != vs.end(); ++v) {
         UpdateMinMax(slot, *v);
         thisBuf.emplace_back(*v); // TODO: Can be optimised in case T == BufEl_t
      }
   }

   template <typename T, typename W, std::enable_if_t<IsDataContainer<T>::value && IsDataContainer<W>::value, int> = 0>
   void Exec(unsigned int slot, const T &vs, const W &ws)
   {
      auto &thisBuf = fBuffers[slot];

      for (auto &v : vs) {
         UpdateMinMax(slot, v);
         thisBuf.emplace_back(v);
      }

      auto &thisWBuf = fWBuffers[slot];
      for (auto &w : ws) {
         thisWBuf.emplace_back(w); // TODO: Can be optimised in case T == BufEl_t
      }
   }

   template <typename T, typename W, std::enable_if_t<IsDataContainer<T>::value && !IsDataContainer<W>::value, int> = 0>
   void Exec(unsigned int slot, const T &vs, const W w)
   {
      auto &thisBuf = fBuffers[slot];
      for (auto &v : vs) {
         UpdateMinMax(slot, v);
         thisBuf.emplace_back(v); // TODO: Can be optimised in case T == BufEl_t
      }

      auto &thisWBuf = fWBuffers[slot];
      thisWBuf.insert(thisWBuf.end(), vs.size(), w);
   }

   template <typename T, typename W, std::enable_if_t<IsDataContainer<W>::value && !IsDataContainer<T>::value, int> = 0>
   void Exec(unsigned int slot, const T v, const W &ws)
   {
      UpdateMinMax(slot, v);
      auto &thisBuf = fBuffers[slot];
      thisBuf.insert(thisBuf.end(), ws.size(), v);

      auto &thisWBuf = fWBuffers[slot];
      thisWBuf.insert(thisWBuf.end(), ws.begin(), ws.end());
   }

   Hist_t &PartialUpdate(unsigned int);

   void Initialize() { /* noop */}

   void Finalize();

   // Helper functions for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      return std::make_unique<RMergeableFill<Hist_t>>(*fResultHist);
   }

   std::string GetActionName()
   {
      return std::string(fResultHist->IsA()->GetName()) + "\\n" + std::string(fResultHist->GetName());
   }

   BufferedFillHelper MakeNew(void *newResult, std::string_view /*variation*/ = "nominal")
   {
      auto &result = *static_cast<std::shared_ptr<Hist_t> *>(newResult);
      result->Reset();
      result->SetDirectory(nullptr);
      return BufferedFillHelper(result, fNSlots);
   }
};

// class which wraps a pointer and implements a no-op increment operator
template <typename T>
class ScalarConstIterator {
   const T *obj_;

public:
   using iterator_category = std::forward_iterator_tag;
   using difference_type = std::ptrdiff_t;
   using value_type = T;
   using pointer = T *;
   using reference = T &;
   ScalarConstIterator(const T *obj) : obj_(obj) {}
   const T &operator*() const { return *obj_; }
   ScalarConstIterator<T> &operator++() { return *this; }
};

// return unchanged value for scalar
template <typename T>
auto MakeBegin(const T &val)
{
   if constexpr (IsDataContainer<T>::value) {
      return std::begin(val);
   } else {
      return ScalarConstIterator<T>(&val);
   }
}

// return container size for containers, and 1 for scalars
template <typename T>
std::size_t GetSize(const T &val)
{
   if constexpr (IsDataContainer<T>::value) {
      return std::size(val);
   } else {
      return 1;
   }
}

// Helpers for dealing with histograms and similar:
template <typename H, typename = decltype(std::declval<H>().Reset())>
void ResetIfPossible(H *h)
{
   h->Reset();
}

void ResetIfPossible(TStatistic *h);
void ResetIfPossible(...);

void UnsetDirectoryIfPossible(TH1 *h);
void UnsetDirectoryIfPossible(...);

/// The generic Fill helper: it calls Fill on per-thread objects and then Merge to produce a final result.
/// For one-dimensional histograms, if no axes are specified, RDataFrame uses BufferedFillHelper instead.
template <typename HIST = Hist_t>
class R__CLING_PTRCHECK(off) FillHelper : public RActionImpl<FillHelper<HIST>> {
   std::vector<HIST *> fObjects;

   // Merge overload for types with Merge(TCollection*), like TH1s
   template <typename H, typename = std::enable_if_t<std::is_base_of<TObject, H>::value, int>>
   auto Merge(std::vector<H *> &objs, int /*toincreaseoverloadpriority*/)
      -> decltype(objs[0]->Merge((TCollection *)nullptr), void())
   {
      TList l;
      for (auto it = ++objs.begin(); it != objs.end(); ++it)
         l.Add(*it);
      objs[0]->Merge(&l);
   }

   // Merge overload for types with Merge(const std::vector&)
   template <typename H>
   auto Merge(std::vector<H *> &objs, double /*toloweroverloadpriority*/)
      -> decltype(objs[0]->Merge(std::vector<HIST *>{}), void())
   {
      objs[0]->Merge({++objs.begin(), objs.end()});
   }

   // Merge overload to error out in case no valid HIST::Merge method was detected
   template <typename T>
   void Merge(T, ...)
   {
      static_assert(sizeof(T) < 0,
                    "The type passed to Fill does not provide a Merge(TCollection*) or Merge(const std::vector&) method.");
   }

   template <std::size_t ColIdx, typename End_t, typename... Its>
   void ExecLoop(unsigned int slot, End_t end, Its... its)
   {
      for (auto *thisSlotH = fObjects[slot]; GetNthElement<ColIdx>(its...) != end; (std::advance(its, 1), ...)) {
         thisSlotH->Fill(*its...);
      }
   }

public:
   FillHelper(FillHelper &&) = default;
   FillHelper(const FillHelper &) = delete;

   FillHelper(const std::shared_ptr<HIST> &h, const unsigned int nSlots) : fObjects(nSlots, nullptr)
   {
      fObjects[0] = h.get();
      // Initialize all other slots
      for (unsigned int i = 1; i < nSlots; ++i) {
         fObjects[i] = new HIST(*fObjects[0]);
         UnsetDirectoryIfPossible(fObjects[i]);
      }
   }

   void InitTask(TTreeReader *, unsigned int) {}

   // no container arguments
   template <typename... ValTypes, std::enable_if_t<!Disjunction<IsDataContainer<ValTypes>...>::value, int> = 0>
   auto Exec(unsigned int slot, const ValTypes &...x) -> decltype(fObjects[slot]->Fill(x...), void())
   {
      fObjects[slot]->Fill(x...);
   }

   // at least one container argument
   template <typename... Xs, std::enable_if_t<Disjunction<IsDataContainer<Xs>...>::value, int> = 0>
   auto Exec(unsigned int slot, const Xs &...xs) -> decltype(fObjects[slot]->Fill(*MakeBegin(xs)...), void())
   {
      // array of bools keeping track of which inputs are containers
      constexpr std::array<bool, sizeof...(Xs)> isContainer{IsDataContainer<Xs>::value...};

      // index of the first container input
      constexpr std::size_t colidx = FindIdxTrue(isContainer);
      // if this happens, there is a bug in the implementation
      static_assert(colidx < sizeof...(Xs), "Error: index of collection-type argument not found.");

      // get the end iterator to the first container
      auto const xrefend = std::end(GetNthElement<colidx>(xs...));

      // array of container sizes (1 for scalars)
      std::array<std::size_t, sizeof...(xs)> sizes = {{GetSize(xs)...}};

      for (std::size_t i = 0; i < sizeof...(xs); ++i) {
         if (isContainer[i] && sizes[i] != sizes[colidx]) {
            throw std::runtime_error("Cannot fill histogram with values in containers of different sizes.");
         }
      }

      ExecLoop<colidx>(slot, xrefend, MakeBegin(xs)...);
   }

   template <typename T = HIST>
   void Exec(...)
   {
      static_assert(sizeof(T) < 0,
                    "When filling an object with RDataFrame (e.g. via a Fill action) the number or types of the "
                    "columns passed did not match the signature of the object's `Fill` method.");
   }

   void Initialize() { /* noop */}

   void Finalize()
   {
      if (fObjects.size() == 1)
         return;

      Merge(fObjects, /*toselectcorrectoverload=*/0);

      // delete the copies we created for the slots other than the first
      for (auto it = ++fObjects.begin(); it != fObjects.end(); ++it)
         delete *it;
   }

   HIST &PartialUpdate(unsigned int slot) { return *fObjects[slot]; }

   // Helper functions for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      return std::make_unique<RMergeableFill<HIST>>(*fObjects[0]);
   }

   // if the fObjects vector type is derived from TObject, return the name of the object
   template <typename T = HIST, std::enable_if_t<std::is_base_of<TObject, T>::value, int> = 0>
   std::string GetActionName()
   {
      return std::string(fObjects[0]->IsA()->GetName()) + "\\n" + std::string(fObjects[0]->GetName());
   }

   // if fObjects is not derived from TObject, indicate it is some other object
   template <typename T = HIST, std::enable_if_t<!std::is_base_of<TObject, T>::value, int> = 0>
   std::string GetActionName()
   {
      return "Fill custom object";
   }

   template <typename H = HIST>
   FillHelper MakeNew(void *newResult, std::string_view /*variation*/ = "nominal")
   {
      auto &result = *static_cast<std::shared_ptr<H> *>(newResult);
      ResetIfPossible(result.get());
      UnsetDirectoryIfPossible(result.get());
      return FillHelper(result, fObjects.size());
   }
};

class R__CLING_PTRCHECK(off) FillTGraphHelper : public ROOT::Detail::RDF::RActionImpl<FillTGraphHelper> {
public:
   using Result_t = ::TGraph;

private:
   std::vector<::TGraph *> fGraphs;

public:
   FillTGraphHelper(FillTGraphHelper &&) = default;
   FillTGraphHelper(const FillTGraphHelper &) = delete;

   FillTGraphHelper(const std::shared_ptr<::TGraph> &g, const unsigned int nSlots) : fGraphs(nSlots, nullptr)
   {
      fGraphs[0] = g.get();
      // Initialize all other slots
      for (unsigned int i = 1; i < nSlots; ++i) {
         fGraphs[i] = new TGraph(*fGraphs[0]);
      }
   }

   void Initialize() {}
   void InitTask(TTreeReader *, unsigned int) {}

   // case: both types are container types
   template <typename X0, typename X1,
             std::enable_if_t<IsDataContainer<X0>::value && IsDataContainer<X1>::value, int> = 0>
   void Exec(unsigned int slot, const X0 &x0s, const X1 &x1s)
   {
      if (x0s.size() != x1s.size()) {
         throw std::runtime_error("Cannot fill Graph with values in containers of different sizes.");
      }
      auto *thisSlotG = fGraphs[slot];
      auto x0sIt = std::begin(x0s);
      const auto x0sEnd = std::end(x0s);
      auto x1sIt = std::begin(x1s);
      for (; x0sIt != x0sEnd; x0sIt++, x1sIt++) {
         thisSlotG->SetPoint(thisSlotG->GetN(), *x0sIt, *x1sIt);
      }
   }

   // case: both types are non-container types, e.g. scalars
   template <typename X0, typename X1,
             std::enable_if_t<!IsDataContainer<X0>::value && !IsDataContainer<X1>::value, int> = 0>
   void Exec(unsigned int slot, X0 x0, X1 x1)
   {
      auto thisSlotG = fGraphs[slot];
      thisSlotG->SetPoint(thisSlotG->GetN(), x0, x1);
   }

   // case: types are combination of containers and non-containers
   // this is not supported, error out
   template <typename X0, typename X1, typename... ExtraArgsToLowerPriority>
   void Exec(unsigned int, X0, X1, ExtraArgsToLowerPriority...)
   {
      throw std::runtime_error("Graph was applied to a mix of scalar values and collections. This is not supported.");
   }

   void Finalize()
   {
      const auto nSlots = fGraphs.size();
      auto resGraph = fGraphs[0];
      TList l;
      l.SetOwner(); // The list will free the memory associated to its elements upon destruction
      for (unsigned int slot = 1; slot < nSlots; ++slot) {
         l.Add(fGraphs[slot]);
      }
      resGraph->Merge(&l);
   }

   // Helper functions for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      return std::make_unique<RMergeableFill<Result_t>>(*fGraphs[0]);
   }

   std::string GetActionName() { return "Graph"; }

   Result_t &PartialUpdate(unsigned int slot) { return *fGraphs[slot]; }

   FillTGraphHelper MakeNew(void *newResult, std::string_view /*variation*/ = "nominal")
   {
      auto &result = *static_cast<std::shared_ptr<TGraph> *>(newResult);
      result->Set(0);
      return FillTGraphHelper(result, fGraphs.size());
   }
};

class R__CLING_PTRCHECK(off) FillTGraphAsymmErrorsHelper
   : public ROOT::Detail::RDF::RActionImpl<FillTGraphAsymmErrorsHelper> {
public:
   using Result_t = ::TGraphAsymmErrors;

private:
   std::vector<::TGraphAsymmErrors *> fGraphAsymmErrors;

public:
   FillTGraphAsymmErrorsHelper(FillTGraphAsymmErrorsHelper &&) = default;
   FillTGraphAsymmErrorsHelper(const FillTGraphAsymmErrorsHelper &) = delete;

   FillTGraphAsymmErrorsHelper(const std::shared_ptr<::TGraphAsymmErrors> &g, const unsigned int nSlots)
      : fGraphAsymmErrors(nSlots, nullptr)
   {
      fGraphAsymmErrors[0] = g.get();
      // Initialize all other slots
      for (unsigned int i = 1; i < nSlots; ++i) {
         fGraphAsymmErrors[i] = new TGraphAsymmErrors(*fGraphAsymmErrors[0]);
      }
   }

   void Initialize() {}
   void InitTask(TTreeReader *, unsigned int) {}

   // case: all types are container types
   template <
      typename X, typename Y, typename EXL, typename EXH, typename EYL, typename EYH,
      std::enable_if_t<IsDataContainer<X>::value && IsDataContainer<Y>::value && IsDataContainer<EXL>::value &&
                          IsDataContainer<EXH>::value && IsDataContainer<EYL>::value && IsDataContainer<EYH>::value,
                       int> = 0>
   void
   Exec(unsigned int slot, const X &xs, const Y &ys, const EXL &exls, const EXH &exhs, const EYL &eyls, const EYH &eyhs)
   {
      if ((xs.size() != ys.size()) || (xs.size() != exls.size()) || (xs.size() != exhs.size()) ||
          (xs.size() != eyls.size()) || (xs.size() != eyhs.size())) {
         throw std::runtime_error("Cannot fill GraphAsymmErrors with values in containers of different sizes.");
      }
      auto *thisSlotG = fGraphAsymmErrors[slot];
      auto xsIt = std::begin(xs);
      auto ysIt = std::begin(ys);
      auto exlsIt = std::begin(exls);
      auto exhsIt = std::begin(exhs);
      auto eylsIt = std::begin(eyls);
      auto eyhsIt = std::begin(eyhs);
      while (xsIt != std::end(xs)) {
         const auto n = thisSlotG->GetN(); // must use the same `n` for SetPoint and SetPointError
         thisSlotG->SetPoint(n, *xsIt++, *ysIt++);
         thisSlotG->SetPointError(n, *exlsIt++, *exhsIt++, *eylsIt++, *eyhsIt++);
      }
   }

   // case: all types are non-container types, e.g. scalars
   template <
      typename X, typename Y, typename EXL, typename EXH, typename EYL, typename EYH,
      std::enable_if_t<!IsDataContainer<X>::value && !IsDataContainer<Y>::value && !IsDataContainer<EXL>::value &&
                          !IsDataContainer<EXH>::value && !IsDataContainer<EYL>::value && !IsDataContainer<EYH>::value,
                       int> = 0>
   void Exec(unsigned int slot, X x, Y y, EXL exl, EXH exh, EYL eyl, EYH eyh)
   {
      auto thisSlotG = fGraphAsymmErrors[slot];
      const auto n = thisSlotG->GetN();
      thisSlotG->SetPoint(n, x, y);
      thisSlotG->SetPointError(n, exl, exh, eyl, eyh);
   }

   // case: types are combination of containers and non-containers
   // this is not supported, error out
   template <typename X, typename Y, typename EXL, typename EXH, typename EYL, typename EYH,
             typename... ExtraArgsToLowerPriority>
   void Exec(unsigned int, X, Y, EXL, EXH, EYL, EYH, ExtraArgsToLowerPriority...)
   {
      throw std::runtime_error(
         "GraphAsymmErrors was applied to a mix of scalar values and collections. This is not supported.");
   }

   void Finalize()
   {
      const auto nSlots = fGraphAsymmErrors.size();
      auto resGraphAsymmErrors = fGraphAsymmErrors[0];
      TList l;
      l.SetOwner(); // The list will free the memory associated to its elements upon destruction
      for (unsigned int slot = 1; slot < nSlots; ++slot) {
         l.Add(fGraphAsymmErrors[slot]);
      }
      resGraphAsymmErrors->Merge(&l);
   }

   // Helper functions for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      return std::make_unique<RMergeableFill<Result_t>>(*fGraphAsymmErrors[0]);
   }

   std::string GetActionName() { return "GraphAsymmErrors"; }

   Result_t &PartialUpdate(unsigned int slot) { return *fGraphAsymmErrors[slot]; }

   FillTGraphAsymmErrorsHelper MakeNew(void *newResult, std::string_view /*variation*/ = "nominal")
   {
      auto &result = *static_cast<std::shared_ptr<TGraphAsymmErrors> *>(newResult);
      result->Set(0);
      return FillTGraphAsymmErrorsHelper(result, fGraphAsymmErrors.size());
   }
};

/// A FillHelper for classes supporting the FillThreadSafe function.
template <typename HIST>
class R__CLING_PTRCHECK(off) ThreadSafeFillHelper : public RActionImpl<ThreadSafeFillHelper<HIST>> {
   std::vector<std::shared_ptr<HIST>> fObjects;
   std::vector<std::unique_ptr<std::mutex>> fMutexPtrs;

   // This overload matches if the function exists:
   template <typename T, typename... Args>
   auto TryCallFillThreadSafe(T &object, std::mutex &, int /*dummy*/, Args... args)
      -> decltype(ROOT::Internal::FillThreadSafe(object, args...), void())
   {
      ROOT::Internal::FillThreadSafe(object, args...);
   }
   // This one has lower precedence because of the dummy argument, and uses a lock
   template <typename T, typename... Args>
   auto TryCallFillThreadSafe(T &object, std::mutex &mutex, char /*dummy*/, Args... args)
   {
      std::scoped_lock lock{mutex};
      object.Fill(args...);
   }

   template <std::size_t ColIdx, typename End_t, typename... Its>
   void ExecLoop(unsigned int slot, End_t end, Its... its)
   {
      const auto localSlot = slot % fObjects.size();
      for (; GetNthElement<ColIdx>(its...) != end; (std::advance(its, 1), ...)) {
         TryCallFillThreadSafe(*fObjects[localSlot], *fMutexPtrs[localSlot], 0, *its...);
      }
   }

public:
   ThreadSafeFillHelper(ThreadSafeFillHelper &&) = default;
   ThreadSafeFillHelper(const ThreadSafeFillHelper &) = delete;

   ThreadSafeFillHelper(const std::shared_ptr<HIST> &h, const unsigned int nSlots)
   {
      fObjects.resize(nSlots);
      fObjects.front() = h;

      std::generate(fObjects.begin() + 1, fObjects.end(), [h]() {
         auto hist = std::make_shared<HIST>(*h);
         UnsetDirectoryIfPossible(hist.get());
         return hist;
      });
      fMutexPtrs.resize(nSlots);
      std::generate(fMutexPtrs.begin(), fMutexPtrs.end(), []() { return std::make_unique<std::mutex>(); });
   }

   void InitTask(TTreeReader *, unsigned int) {}

   // no container arguments
   template <typename... ValTypes, std::enable_if_t<!Disjunction<IsDataContainer<ValTypes>...>::value, int> = 0>
   void Exec(unsigned int slot, const ValTypes &...x)
   {
      const auto localSlot = slot % fObjects.size();
      TryCallFillThreadSafe(*fObjects[localSlot], *fMutexPtrs[localSlot], 0, x...);
   }

   // at least one container argument
   template <typename... Xs, std::enable_if_t<Disjunction<IsDataContainer<Xs>...>::value, int> = 0>
   void Exec(unsigned int slot, const Xs &...xs)
   {
      // array of bools keeping track of which inputs are containers
      constexpr std::array<bool, sizeof...(Xs)> isContainer{IsDataContainer<Xs>::value...};

      // index of the first container input
      constexpr std::size_t colidx = FindIdxTrue(isContainer);
      // if this happens, there is a bug in the implementation
      static_assert(colidx < sizeof...(Xs), "Error: index of collection-type argument not found.");

      // get the end iterator to the first container
      auto const xrefend = std::end(GetNthElement<colidx>(xs...));

      // array of container sizes (1 for scalars)
      std::array<std::size_t, sizeof...(xs)> sizes = {{GetSize(xs)...}};

      for (std::size_t i = 0; i < sizeof...(xs); ++i) {
         if (isContainer[i] && sizes[i] != sizes[colidx]) {
            throw std::runtime_error("Cannot fill histogram with values in containers of different sizes.");
         }
      }

      ExecLoop<colidx>(slot, xrefend, MakeBegin(xs)...);
   }

   template <typename T = HIST>
   void Exec(...)
   {
      static_assert(sizeof(T) < 0,
                    "When filling an object with RDataFrame (e.g. via a Fill action) the number or types of the "
                    "columns passed did not match the signature of the object's `FillThreadSafe` method.");
   }

   void Initialize() { /* noop */ }

   void Finalize()
   {
      if (fObjects.size() > 1) {
         TList list;
         for (auto it = fObjects.cbegin() + 1; it != fObjects.end(); ++it) {
            list.Add(it->get());
         }
         fObjects[0]->Merge(&list);
      }

      fObjects.resize(1);
      fMutexPtrs.clear();
   }

   // Helper function for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      return std::make_unique<RMergeableFill<HIST>>(*fObjects[0]);
   }

   // if the fObjects vector type is derived from TObject, return the name of the object
   template <typename T = HIST, std::enable_if_t<std::is_base_of<TObject, T>::value, int> = 0>
   std::string GetActionName()
   {
      return std::string(fObjects[0]->IsA()->GetName()) + "\\n" + std::string(fObjects[0]->GetName());
   }

   template <typename H = HIST>
   ThreadSafeFillHelper MakeNew(void *newResult, std::string_view /*variation*/ = "nominal")
   {
      auto &result = *static_cast<std::shared_ptr<H> *>(newResult);
      ResetIfPossible(result.get());
      UnsetDirectoryIfPossible(result.get());
      return ThreadSafeFillHelper(result, fObjects.size());
   }
};

// In case of the take helper we have 4 cases:
// 1. The column is not an RVec, the collection is not a vector
// 2. The column is not an RVec, the collection is a vector
// 3. The column is an RVec, the collection is not a vector
// 4. The column is an RVec, the collection is a vector

template <typename V, typename COLL>
void FillColl(V&& v, COLL& c) {
   c.emplace_back(v);
}

// Use push_back for bool since some compilers do not support emplace_back.
template <typename COLL>
void FillColl(bool v, COLL& c) {
   c.push_back(v);
}

// Case 1.: The column is not an RVec, the collection is not a vector
// No optimisations, no transformations: just copies.
template <typename RealT_t, typename T, typename COLL>
class R__CLING_PTRCHECK(off) TakeHelper : public RActionImpl<TakeHelper<RealT_t, T, COLL>> {
   Results<std::shared_ptr<COLL>> fColls;

public:
   using ColumnTypes_t = TypeList<T>;
   TakeHelper(const std::shared_ptr<COLL> &resultColl, const unsigned int nSlots)
   {
      fColls.emplace_back(resultColl);
      for (unsigned int i = 1; i < nSlots; ++i)
         fColls.emplace_back(std::make_shared<COLL>());
   }
   TakeHelper(TakeHelper &&);
   TakeHelper(const TakeHelper &) = delete;

   void InitTask(TTreeReader *, unsigned int) {}

   void Exec(unsigned int slot, T &v) { FillColl(v, *fColls[slot]); }

   void Initialize() { /* noop */}

   void Finalize()
   {
      auto rColl = fColls[0];
      for (unsigned int i = 1; i < fColls.size(); ++i) {
         const auto &coll = fColls[i];
         const auto end = coll->end();
         // Use an explicit loop here to prevent compiler warnings introduced by
         // clang's range-based loop analysis and vector<bool> references.
         for (auto j = coll->begin(); j != end; j++) {
            FillColl(*j, *rColl);
         }
      }
   }

   COLL &PartialUpdate(unsigned int slot) { return *fColls[slot].get(); }

   std::string GetActionName() { return "Take"; }

   TakeHelper MakeNew(void *newResult, std::string_view /*variation*/ = "nominal")
   {
      auto &result = *static_cast<std::shared_ptr<COLL> *>(newResult);
      result->clear();
      return TakeHelper(result, fColls.size());
   }
};

// Case 2.: The column is not an RVec, the collection is a vector
// Optimisations, no transformations: just copies.
template <typename RealT_t, typename T>
class R__CLING_PTRCHECK(off) TakeHelper<RealT_t, T, std::vector<T>>
   : public RActionImpl<TakeHelper<RealT_t, T, std::vector<T>>> {
   Results<std::shared_ptr<std::vector<T>>> fColls;

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
   TakeHelper(TakeHelper &&);
   TakeHelper(const TakeHelper &) = delete;

   void InitTask(TTreeReader *, unsigned int) {}

   void Exec(unsigned int slot, T &v) { FillColl(v, *fColls[slot]); }

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

   std::string GetActionName() { return "Take"; }

   TakeHelper MakeNew(void *newResult, std::string_view /*variation*/ = "nominal")
   {
      auto &result = *static_cast<std::shared_ptr<std::vector<T>> *>(newResult);
      result->clear();
      return TakeHelper(result, fColls.size());
   }
};

// Case 3.: The column is a RVec, the collection is not a vector
// No optimisations, transformations from RVecs to vectors
template <typename RealT_t, typename COLL>
class R__CLING_PTRCHECK(off) TakeHelper<RealT_t, RVec<RealT_t>, COLL>
   : public RActionImpl<TakeHelper<RealT_t, RVec<RealT_t>, COLL>> {
   Results<std::shared_ptr<COLL>> fColls;

public:
   using ColumnTypes_t = TypeList<RVec<RealT_t>>;
   TakeHelper(const std::shared_ptr<COLL> &resultColl, const unsigned int nSlots)
   {
      fColls.emplace_back(resultColl);
      for (unsigned int i = 1; i < nSlots; ++i)
         fColls.emplace_back(std::make_shared<COLL>());
   }
   TakeHelper(TakeHelper &&);
   TakeHelper(const TakeHelper &) = delete;

   void InitTask(TTreeReader *, unsigned int) {}

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

   std::string GetActionName() { return "Take"; }

   TakeHelper MakeNew(void *newResult, std::string_view /*variation*/ = "nominal")
   {
      auto &result = *static_cast<std::shared_ptr<COLL> *>(newResult);
      result->clear();
      return TakeHelper(result, fColls.size());
   }
};

// Case 4.: The column is an RVec, the collection is a vector
// Optimisations, transformations from RVecs to vectors
template <typename RealT_t>
class R__CLING_PTRCHECK(off) TakeHelper<RealT_t, RVec<RealT_t>, std::vector<RealT_t>>
   : public RActionImpl<TakeHelper<RealT_t, RVec<RealT_t>, std::vector<RealT_t>>> {

   Results<std::shared_ptr<std::vector<std::vector<RealT_t>>>> fColls;

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
   TakeHelper(TakeHelper &&);
   TakeHelper(const TakeHelper &) = delete;

   void InitTask(TTreeReader *, unsigned int) {}

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

   std::string GetActionName() { return "Take"; }

   TakeHelper MakeNew(void *newResult, std::string_view /*variation*/ = "nominal")
   {
      auto &result = *static_cast<typename decltype(fColls)::value_type *>(newResult);
      result->clear();
      return TakeHelper(result, fColls.size());
   }
};

// Extern templates for TakeHelper
// NOTE: The move-constructor of specializations declared as extern templates
// must be defined out of line, otherwise cling fails to find its symbol.
template <typename RealT_t, typename T, typename COLL>
TakeHelper<RealT_t, T, COLL>::TakeHelper(TakeHelper<RealT_t, T, COLL> &&) = default;
template <typename RealT_t, typename T>
TakeHelper<RealT_t, T, std::vector<T>>::TakeHelper(TakeHelper<RealT_t, T, std::vector<T>> &&) = default;
template <typename RealT_t, typename COLL>
TakeHelper<RealT_t, RVec<RealT_t>, COLL>::TakeHelper(TakeHelper<RealT_t, RVec<RealT_t>, COLL> &&) = default;
template <typename RealT_t>
TakeHelper<RealT_t, RVec<RealT_t>, std::vector<RealT_t>>::TakeHelper(TakeHelper<RealT_t, RVec<RealT_t>, std::vector<RealT_t>> &&) = default;

// External templates are disabled for gcc5 since this version wrongly omits the C++11 ABI attribute
#if __GNUC__ > 5
extern template class TakeHelper<bool, bool, std::vector<bool>>;
extern template class TakeHelper<unsigned int, unsigned int, std::vector<unsigned int>>;
extern template class TakeHelper<unsigned long, unsigned long, std::vector<unsigned long>>;
extern template class TakeHelper<unsigned long long, unsigned long long, std::vector<unsigned long long>>;
extern template class TakeHelper<int, int, std::vector<int>>;
extern template class TakeHelper<long, long, std::vector<long>>;
extern template class TakeHelper<long long, long long, std::vector<long long>>;
extern template class TakeHelper<float, float, std::vector<float>>;
extern template class TakeHelper<double, double, std::vector<double>>;
#endif

template <typename ResultType>
class R__CLING_PTRCHECK(off) MinHelper : public RActionImpl<MinHelper<ResultType>> {
   std::shared_ptr<ResultType> fResultMin;
   Results<ResultType> fMins;

public:
   MinHelper(MinHelper &&) = default;
   MinHelper(const std::shared_ptr<ResultType> &minVPtr, const unsigned int nSlots)
      : fResultMin(minVPtr), fMins(nSlots, std::numeric_limits<ResultType>::max())
   {
   }

   void Exec(unsigned int slot, ResultType v) { fMins[slot] = std::min(v, fMins[slot]); }

   void InitTask(TTreeReader *, unsigned int) {}

   template <typename T, std::enable_if_t<IsDataContainer<T>::value, int> = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      for (auto &&v : vs)
         fMins[slot] = std::min(static_cast<ResultType>(v), fMins[slot]);
   }

   void Initialize() { /* noop */}

   void Finalize()
   {
      *fResultMin = std::numeric_limits<ResultType>::max();
      for (auto &m : fMins)
         *fResultMin = std::min(m, *fResultMin);
   }

   // Helper functions for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      return std::make_unique<RMergeableMin<ResultType>>(*fResultMin);
   }

   ResultType &PartialUpdate(unsigned int slot) { return fMins[slot]; }

   std::string GetActionName() { return "Min"; }

   MinHelper MakeNew(void *newResult, std::string_view /*variation*/ = "nominal")
   {
      auto &result = *static_cast<std::shared_ptr<ResultType> *>(newResult);
      return MinHelper(result, fMins.size());
   }
};

template <typename ResultType>
class R__CLING_PTRCHECK(off) MaxHelper : public RActionImpl<MaxHelper<ResultType>> {
   std::shared_ptr<ResultType> fResultMax;
   Results<ResultType> fMaxs;

public:
   MaxHelper(MaxHelper &&) = default;
   MaxHelper(const MaxHelper &) = delete;
   MaxHelper(const std::shared_ptr<ResultType> &maxVPtr, const unsigned int nSlots)
      : fResultMax(maxVPtr), fMaxs(nSlots, std::numeric_limits<ResultType>::lowest())
   {
   }

   void InitTask(TTreeReader *, unsigned int) {}
   void Exec(unsigned int slot, ResultType v) { fMaxs[slot] = std::max(v, fMaxs[slot]); }

   template <typename T, std::enable_if_t<IsDataContainer<T>::value, int> = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      for (auto &&v : vs)
         fMaxs[slot] = std::max(static_cast<ResultType>(v), fMaxs[slot]);
   }

   void Initialize() { /* noop */}

   void Finalize()
   {
      *fResultMax = std::numeric_limits<ResultType>::lowest();
      for (auto &m : fMaxs) {
         *fResultMax = std::max(m, *fResultMax);
      }
   }

   // Helper functions for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      return std::make_unique<RMergeableMax<ResultType>>(*fResultMax);
   }

   ResultType &PartialUpdate(unsigned int slot) { return fMaxs[slot]; }

   std::string GetActionName() { return "Max"; }

   MaxHelper MakeNew(void *newResult, std::string_view /*variation*/ = "nominal")
   {
      auto &result = *static_cast<std::shared_ptr<ResultType> *>(newResult);
      return MaxHelper(result, fMaxs.size());
   }
};

template <typename ResultType>
class R__CLING_PTRCHECK(off) SumHelper : public RActionImpl<SumHelper<ResultType>> {
   std::shared_ptr<ResultType> fResultSum;
   Results<ResultType> fSums;
   Results<ResultType> fCompensations;

   /// Evaluate neutral element for this type and the sum operation.
   /// This is assumed to be any_value - any_value if operator- is defined
   /// for the type, otherwise a default-constructed ResultType{} is used.
   template <typename T = ResultType>
   auto NeutralElement(const T &v, int /*overloadresolver*/) -> decltype(v - v)
   {
      return v - v;
   }

   template <typename T = ResultType, typename Dummy = int>
   ResultType NeutralElement(const T &, Dummy) // this overload has lower priority thanks to the template arg
   {
      return ResultType{};
   }

public:
   SumHelper(SumHelper &&) = default;
   SumHelper(const SumHelper &) = delete;
   SumHelper(const std::shared_ptr<ResultType> &sumVPtr, const unsigned int nSlots)
      : fResultSum(sumVPtr), fSums(nSlots, NeutralElement(*sumVPtr, -1)),
        fCompensations(nSlots, NeutralElement(*sumVPtr, -1))
   {
   }
   void InitTask(TTreeReader *, unsigned int) {}

   void Exec(unsigned int slot, ResultType x)
   {
      // Kahan Sum:
      ResultType y = x - fCompensations[slot];
      ResultType t = fSums[slot] + y;
      fCompensations[slot] = (t - fSums[slot]) - y;
      fSums[slot] = t;
   }

   template <typename T, std::enable_if_t<IsDataContainer<T>::value, int> = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      for (auto &&v : vs) {
         Exec(slot, v);
      }
   }

   void Initialize() { /* noop */}

   void Finalize()
   {
      ResultType sum(NeutralElement(ResultType{}, -1));
      ResultType compensation(NeutralElement(ResultType{}, -1));
      ResultType y(NeutralElement(ResultType{}, -1));
      ResultType t(NeutralElement(ResultType{}, -1));
      for (auto &m : fSums) {
         // Kahan Sum:
         y = m - compensation;
         t = sum + y;
         compensation = (t - sum) - y;
         sum = t;
      }
      *fResultSum += sum;
   }

   // Helper functions for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      return std::make_unique<RMergeableSum<ResultType>>(*fResultSum);
   }

   ResultType &PartialUpdate(unsigned int slot) { return fSums[slot]; }

   std::string GetActionName() { return "Sum"; }

   SumHelper MakeNew(void *newResult, std::string_view /*variation*/ = "nominal")
   {
      auto &result = *static_cast<std::shared_ptr<ResultType> *>(newResult);
      *result = NeutralElement(*result, -1);
      return SumHelper(result, fSums.size());
   }
};

class R__CLING_PTRCHECK(off) MeanHelper : public RActionImpl<MeanHelper> {
   std::shared_ptr<double> fResultMean;
   std::vector<ULong64_t> fCounts;
   std::vector<double> fSums;
   std::vector<double> fPartialMeans;
   std::vector<double> fCompensations;

public:
   MeanHelper(const std::shared_ptr<double> &meanVPtr, const unsigned int nSlots);
   MeanHelper(MeanHelper &&) = default;
   MeanHelper(const MeanHelper &) = delete;
   void InitTask(TTreeReader *, unsigned int) {}
   void Exec(unsigned int slot, double v);

   template <typename T, std::enable_if_t<IsDataContainer<T>::value, int> = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      for (auto &&v : vs) {

         fCounts[slot]++;
         // Kahan Sum:
         double y = v - fCompensations[slot];
         double t = fSums[slot] + y;
         fCompensations[slot] = (t - fSums[slot]) - y;
         fSums[slot] = t;
      }
   }

   void Initialize() { /* noop */}

   void Finalize();

   // Helper functions for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      const ULong64_t counts = std::accumulate(fCounts.begin(), fCounts.end(), 0ull);
      return std::make_unique<RMergeableMean>(*fResultMean, counts);
   }

   double &PartialUpdate(unsigned int slot);

   std::string GetActionName() { return "Mean"; }

   MeanHelper MakeNew(void *newResult, std::string_view /*variation*/ = "nominal")
   {
      auto &result = *static_cast<std::shared_ptr<double> *>(newResult);
      return MeanHelper(result, fSums.size());
   }
};

class R__CLING_PTRCHECK(off) StdDevHelper : public RActionImpl<StdDevHelper> {
   // Number of subsets of data
   unsigned int fNSlots;
   std::shared_ptr<double> fResultStdDev;
   // Number of element for each slot
   std::vector<ULong64_t> fCounts;
   // Mean of each slot
   std::vector<double> fMeans;
   // Squared distance from the mean
   std::vector<double> fDistancesfromMean;

public:
   StdDevHelper(const std::shared_ptr<double> &meanVPtr, const unsigned int nSlots);
   StdDevHelper(StdDevHelper &&) = default;
   StdDevHelper(const StdDevHelper &) = delete;
   void InitTask(TTreeReader *, unsigned int) {}
   void Exec(unsigned int slot, double v);

   template <typename T, std::enable_if_t<IsDataContainer<T>::value, int> = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      for (auto &&v : vs) {
         Exec(slot, v);
      }
   }

   void Initialize() { /* noop */}

   void Finalize();

   // Helper functions for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      const ULong64_t counts = std::accumulate(fCounts.begin(), fCounts.end(), 0ull);
      const Double_t mean =
         std::inner_product(fMeans.begin(), fMeans.end(), fCounts.begin(), 0.) / static_cast<Double_t>(counts);
      return std::make_unique<RMergeableStdDev>(*fResultStdDev, counts, mean);
   }

   std::string GetActionName() { return "StdDev"; }

   StdDevHelper MakeNew(void *newResult, std::string_view /*variation*/ = "nominal")
   {
      auto &result = *static_cast<std::shared_ptr<double> *>(newResult);
      return StdDevHelper(result, fCounts.size());
   }
};

template <typename PrevNodeType>
class R__CLING_PTRCHECK(off) DisplayHelper : public RActionImpl<DisplayHelper<PrevNodeType>> {
private:
   using Display_t = ROOT::RDF::RDisplay;
   std::shared_ptr<Display_t> fDisplayerHelper;
   std::shared_ptr<PrevNodeType> fPrevNode;
   size_t fEntriesToProcess;

public:
   DisplayHelper(size_t nRows, const std::shared_ptr<Display_t> &d, const std::shared_ptr<PrevNodeType> &prevNode)
      : fDisplayerHelper(d), fPrevNode(prevNode), fEntriesToProcess(nRows)
   {
   }
   DisplayHelper(DisplayHelper &&) = default;
   DisplayHelper(const DisplayHelper &) = delete;
   void InitTask(TTreeReader *, unsigned int) {}

   template <typename... Columns>
   void Exec(unsigned int, Columns &... columns)
   {
      if (fEntriesToProcess == 0)
         return;

      fDisplayerHelper->AddRow(columns...);
      --fEntriesToProcess;

      if (fEntriesToProcess == 0) {
         // No more entries to process. Send a one-time signal that this node
         // of the graph is done. It is important that the 'StopProcessing'
         // method is only called once from this helper, otherwise it would seem
         // like more than one operation has completed its work.
         fPrevNode->StopProcessing();
      }
   }

   void Initialize() {}

   void Finalize() {}

   std::string GetActionName() { return "Display"; }
};

template <typename Acc, typename Merge, typename R, typename T, typename U,
          bool MustCopyAssign = std::is_same<R, U>::value>
class R__CLING_PTRCHECK(off) AggregateHelper
   : public RActionImpl<AggregateHelper<Acc, Merge, R, T, U, MustCopyAssign>> {
   Acc fAggregate;
   Merge fMerge;
   std::shared_ptr<U> fResult;
   Results<U> fAggregators;

public:
   using ColumnTypes_t = TypeList<T>;

   AggregateHelper(Acc &&f, Merge &&m, const std::shared_ptr<U> &result, const unsigned int nSlots)
      : fAggregate(std::move(f)), fMerge(std::move(m)), fResult(result), fAggregators(nSlots, *result)
   {
   }

   AggregateHelper(Acc &f, Merge &m, const std::shared_ptr<U> &result, const unsigned int nSlots)
      : fAggregate(f), fMerge(m), fResult(result), fAggregators(nSlots, *result)
   {
   }

   AggregateHelper(AggregateHelper &&) = default;
   AggregateHelper(const AggregateHelper &) = delete;

   void InitTask(TTreeReader *, unsigned int) {}

   template <bool MustCopyAssign_ = MustCopyAssign, std::enable_if_t<MustCopyAssign_, int> = 0>
   void Exec(unsigned int slot, const T &value)
   {
      fAggregators[slot] = fAggregate(fAggregators[slot], value);
   }

   template <bool MustCopyAssign_ = MustCopyAssign, std::enable_if_t<!MustCopyAssign_, int> = 0>
   void Exec(unsigned int slot, const T &value)
   {
      fAggregate(fAggregators[slot], value);
   }

   void Initialize() { /* noop */}

   template <typename MergeRet = typename CallableTraits<Merge>::ret_type,
             bool MergeAll = std::is_same<void, MergeRet>::value>
   std::enable_if_t<MergeAll, void> Finalize()
   {
      fMerge(fAggregators);
      *fResult = fAggregators[0];
   }

   template <typename MergeRet = typename CallableTraits<Merge>::ret_type,
             bool MergeTwoByTwo = std::is_same<U, MergeRet>::value>
   std::enable_if_t<MergeTwoByTwo, void> Finalize(...) // ... needed to let compiler distinguish overloads
   {
      for (const auto &acc : fAggregators)
         *fResult = fMerge(*fResult, acc);
   }

   U &PartialUpdate(unsigned int slot) { return fAggregators[slot]; }

   std::string GetActionName() { return "Aggregate"; }

   AggregateHelper MakeNew(void *newResult, std::string_view /*variation*/ = "nominal")
   {
      auto &result = *static_cast<std::shared_ptr<U> *>(newResult);
      return AggregateHelper(fAggregate, fMerge, result, fAggregators.size());
   }
};

} // end of NS RDF
} // end of NS Internal
} // end of NS ROOT

/// \endcond

#endif
